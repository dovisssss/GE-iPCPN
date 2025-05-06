import numpy as np
import os
import torch
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.model.physics_network import PhysicsNetwork as PNetwork, PhysicsNetwork
from src.model.custom_scheduler import CustomScheduler as Schedule, CustomScheduler
from src.model.function_library import FunctionLibrary as Library
from src.visualization.plotting import plot_prediction, plot_loss_curve


class Trainer:
    def __init__(self, cfg, logger, output_dir, device):
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.library = Library()

        # directory setting
        checkpoint_dir = os.path.join(output_dir, cfg. dirs.checkpoints)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = checkpoint_dir + "model"
        self.figure_dir = os.path.join(output_dir, cfg. dirs.figures)
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)

        self._init_optimizers(cfg)
        self._init_loss_records()
    #tensorflow允许预定义优化器参数，而pytorch要在模型创建（实例化）后将参数与优化器绑定
    def _init_optimizers(self, cfg):
        network_scheduler = Schedule(
            optimizer=None,
            initial_learning_rate=cfg.training.network_initial_lr,
            decay_steps=cfg.training.network_decay_steps,
            decay_rate=cfg.training.network_decay_rate,
            min_lr=cfg.training.network_minimum_lr
        )

        physics_schduler = Schedule(
            optimizer=None,
            initial_learning_rate=cfg.training.physics_initial_lr,
            decay_steps=cfg.training.physics_decay_steps,
            decay_rate=cfg.training.physics_decay_rate,
            min_lr=cfg.training.physics_minimum_lr
        )

        self.pretrain_opt = Adam([], lr=cfg.training.pretrain_lr, betas=(0.9, 0.999))
        #未定义network_opt和physics_opt

    def _init_loss_records(self):
        self.pretrain_loss_min = float('inf')
        self.physics_loss_min = float('inf')
        self.network_loss_min = float('inf')
        self.lamda_velocity_int = self.cfg.training.lamda_velocity_int
        self.lamda_l1 = self.cfg.training.lamda_l1

    #calculate multi-tasks losses
    def compute_losses(self, model, input, targets):
        normalized_displacement_hat, normalized_velocity_error = model.step_forward(input)

        displacement_loss = torch.nn.functional.mse_loss(normalized_displacement_hat, targets)
        velocity_loss = torch.nn.functional.mse_loss(
            normalized_velocity_error, torch.zeros_like(normalized_velocity_error)
        )
        l1_loss = torch.sum(torch.stack([torch.abs(p) for p in model.physics_variables]))

        return displacement_loss, velocity_loss, l1_loss

    def network_train_step(self, model, dataloader, physics_flag=True):
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer = self.network_opt if physics_flag else self.pretrain_opt
            optimizer.zero_grad()

            displacement_loss, velocity_loss, l1_loss = self.compute_losses(model, inputs, targets)

            if not physics_flag:
                loss = displacement_loss
            else:
                loss = displacement_loss + self.lamda_velocity_int + velocity_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)  #对应源代码中average_training_loss


    def network_validate_step(self, model, dataloader, physics_flag=True):
        model.eval()
        total_disp_loss = 0.0
        total_vel_loss = 0.0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                disp_loss, vel_loss, _ = self.compute_losses(model, inputs, targets)

                total_disp_loss += disp_loss.item()
                total_vel_loss += vel_loss.item()
                if physics_flag:
                    total_loss += (disp_loss + self.lambda_velocity_int * vel_loss).item()
                else:
                    total_loss += disp_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_disp = total_disp_loss / len(dataloader)
        avg_vel = total_vel_loss / len(dataloader)
        return (avg_loss, avg_disp, avg_vel) if physics_flag else avg_loss

    def physics_train_step(self, model, dataloader):
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.physics_opt.zero_grad()

            _, velocity_loss, l1_loss = self.compute_losses(model, inputs, targets)

            loss = self.lamda_velocity_int * velocity_loss + self.lamda_l1 * l1_loss

            loss.backward()
            self.physics_opt.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def physics_validate_step(self, model, dataloader):
        model.eval()
        total_loss = 0.0
        total_vel_loss = 0.0
        total_l1_loss = 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                _, vel_loss, l1_loss = self.compute_losses(model, inputs, targets)

                total_vel_loss += vel_loss.item()
                total_l1_loss += l1_loss.item()
                total_loss += (self.lamda_velocity_int * vel_loss + self.lamda_l1 * l1_loss).item()

                avg_loss = total_loss / len(dataloader)
                avg_vel = total_vel_loss / len(dataloader)
                avg_l1 = total_l1_loss / len(dataloader)
                return avg_loss, avg_vel, avg_l1

    def load_checkpoint(self, model, path):
        #load checkpoints of model
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        self.network_opt.load_state_dict(checkpoint["optimizers"]['network'])
        self.physics_opt.load_state_dict(checkpoint["optimizers"]['physics'])
        self.pretrain_opt.load_state_dict(checkpoint["optimizers"]['pretrain'])

    def visualize_results(self, model, dataloader, clean_data, suffix):
        model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                disp_pred, vel_pred = model.predict(inputs)
                all_preds.append((disp_pred.cpu(), vel_pred.cpu()))

        disp_all = torch.cat(p[0] for p in all_preds)
        vel_all = torch.cat(p[1] for p in all_preds)

        plot_prediction(disp_all, clean_data['displacement_clean'], "Predicted Displacement",
                        f"{self.figure_dir}/displacement_hat_{suffix}.png")
        plot_prediction(vel_all, clean_data['velocity_clean'], "Predicted Velocity",
                        f"{self.figure_dir}/velocity_hat_{suffix}.png")
        plt.close('all')
        #return model, disp_all, vel_all

    def train(self, train_loader, val_loader, Phi_int, Phi_diff, clean_data, max_values):
        model = PhysicsNetwork(
            self.cfg, Phi_int, Phi_diff,
            self.library.terms_number,
            self.library.build_functions(np.ones(self.library.terms_number)),
            max_values
        ).to(self.device)

        #bind optimizer parameters
        self._bind_optimizers(model)

        #pretrain stage
        self.logger.info("Starting training...")
        self._training_phase(model, train_loader, val_loader,
                             epochs=self.cfg.training.pretrain_epochs,
                             phase="pretrain")

        for iter in range(self.cfg.training.alternate_number):
            self.logger.info(f"Alternation Iteration {iter+1}")

            self._training_phase(model, train_loader, val_loader,
                                 epochs=self.cfg.training.physics_epochs,
                                 phase="physics")

            self._training_phase(model, train_loader, val_loader,
                                 epochs=self.cfg.training.network_epochs,
                                 phase="network")

            coeffs = [p.detach().cpu().numpy() for p in model.physics_variables]
            func = self.library.get_functions(coeffs)
            model.update_function(func)

            # visualize current result
            self.visualize_results(model, val_loader, clean_data, f"iter_{iter + 1}")

        return model

    def _bind_optimizers(self, model):
        network_params = model.displacement_model.parameters()
        physics_params = model.physics_variables

        self.network_opt = Adam(network_params, lr=self.cfg.training.network_initial_lr,
                                betas=(0.9,0.999))
        self.physics_opt = Adam(physics_params, lr=self.cfg.training.physics_initial_lr,
                                betas=(0.9,0.999))

        self.network_scheduler = CustomScheduler(
            self.network_opt,
            initial_learning_rate=self.cfg.training.network_initial_lr,
            decay_steps=self.cfg.training.network_epochs,
            decay_rate=self.cfg.training.network_decay_rate,
            min_lr=self.cfg.training.network_minimum_lr
        )

        self.physics_scheduler = CustomScheduler(
            self.physics_opt,
            initial_learning_rate=self.cfg.training.physics_initial_lr,
            decay_steps=self.cfg.training.physics_decay_steps,
            decay_rate=self.cfg.training.physics_decay_rate,
            min_lr=self.cfg.training.physics_minimum_lr
        )

    def _training_phase(self, model, train_loader, val_loader, epochs, phase):
        train_losses = []
        val_losses = []
        best_loss = float("inf")

        for epoch in range(epochs):
            if phase == "physics":
                train_loss = self.physics_train_step(model, train_loader)
                val_loss, val_vel, val_l1 = self.physics_validate_step(model, val_loader)
            elif phase == "network":
                train_loss = self.network_train_step(model, train_loader)
                val_loss, val_disp, val_vel = self.network_validate_step(model, val_loader)
            else:
                train_loss = self.network_train_step(model, train_loader, physics_flag=False)
                val_loss = self.network_validate_step(model, val_loader, physics_flag=False)

            # record losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(model, f"{self.checkpoint_path}/best_{phase}.pt")

            #output log
            if epoch % 100 == 0:
                log_msg = f"{phase.capitalize()} Epoch: {epoch+1}/{epochs} | "
                log_msg += f"LR:{self._get_current_lr(phase):.2e} | "
                log_msg += f"Train: {train_loss:.3e} | Val: {val_loss:.3e}"

                if phase == "physics":
                    log_msg += f"| Vel: {val_vel:.3e} | l1: {val_l1:.3e}"
                elif phase == "network":
                    log_msg += f"| Disp: {val_disp:.3e} | Vel: {val_loss:.3e}"

                self.logger.info(log_msg)

            if phase == "physics":
                self.physics_scheduler()
            else:
                self.network_scheduler()

        plot_loss_curve(
            train_losses, val_losses,
            title=f"{phase.capitalize()}  Training",
            save_path=f"{self.figure_dir}/{phase}_loss.png"
        )

    def _get_current_lr(self, phase):
        if phase == "physics":
            return self.physics_opt.param_groups[0]["lr"]
        else:
            return self.network_opt.param_groups[0]["lr"]





































