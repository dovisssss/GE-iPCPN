import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.model.physics_network import PhysicsNetwork as PNetwork
from src.model.custom_scheduler import CustomScheduler
from src.model.function_library import FunctionLibrary as Library
from src.model.grad_function_lib import gradFunctionLibrary as GradFunctionLibrary
from src.visualization.plotting import plot_prediction, plot_loss_curve


class Trainer:
    def __init__(self, cfg, logger, output_dir):
        self.cfg = cfg
        self.logger = logger
        self.library = Library()
        self.grad_library = GradFunctionLibrary()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # setting directory
        checkpoint_dir = os.path.join(output_dir, cfg.dirs.checkpoints)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, "model.pth")

        self.figure_dir = os.path.join(output_dir, cfg.dirs.figures)
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)

        # Loss function & optimizer Placeholder
        self.mse = nn.MSELoss()
        self.network_opt = None
        self.physics_opt = None
        self.physics_grad_opt = None
        self.pretrain_opt = None

        # learning rate scheduler
        self.network_scheduler = CustomScheduler(
            cfg.training.network_initial_lr,
            cfg.training.network_decay_steps,
            cfg.training.network_decay_rate,
            cfg.training.network_minimum_lr
        )
        self.physics_scheduler = CustomScheduler(
            cfg.training.physics_initial_lr,
            cfg.training.physics_decay_steps,
            cfg.training.physics_decay_rate,
            cfg.training.physics_minimum_lr
        )

        self.pretrain_loss_minimum = 1e10
        self.physics_loss_minimum = 1e10
        self.network_loss_minimum = 1e10
        self.lambda_velocity_int = cfg.training.lamda_velocity_int
        self.lambda_l1 = cfg.training.lamda_l1
        self.lambda_grad_l1 = cfg.training.lamda_grad_l1
        self.lambda_gradient = cfg.training.lamda_gradient_enhanced

    def compute_losses(self, model, input, output):
        normalized_displacement_hat, normalized_velocity_error, acc_error = model.step_forward(input)
        displacement_loss = self.mse(output, normalized_displacement_hat)
        velocity_loss = self.mse(normalized_velocity_error, torch.zeros_like(normalized_velocity_error))
        acc_loss = self.mse(acc_error, torch.zeros_like(acc_error))
        l1_loss = torch.sum(torch.abs(torch.stack(model.physics_variables)))
        grad_l1_loss = torch.sum(torch.abs(torch.stack(model.physics_grad_variables)))
        return displacement_loss, velocity_loss, l1_loss, acc_loss, grad_l1_loss

    def network_train_step(self, model, training_dataset, physics_flag=True):
        model.train()
        total_loss = 0.0
        for inputs, displacements in training_dataset:
            inputs = inputs.to(self.device)
            displacements = displacements.to(self.device)

            # Gradient zeroing
            if physics_flag:
                self.network_opt.zero_grad()
            else:
                self.pretrain_opt.zero_grad()

            # Forward propagation and loss calculation
            displacement_loss, velocity_loss, l1_loss, acc_loss, grad_l1_loss = self.compute_losses(model, inputs, displacements)

            # total loss
            if not physics_flag:
                total_loss_step = displacement_loss
            else:
                total_loss_step = displacement_loss + self.lambda_velocity_int * velocity_loss + self.lambda_gradient * acc_loss

            # Backpropagation and optimization
            total_loss_step.backward()
            if physics_flag:
                self.network_opt.step()
            else:
                self.pretrain_opt.step()

            total_loss += total_loss_step.item()

        return total_loss / len(training_dataset)

    def network_validate_step(self, model, validation_dataset, physics_flag=True):
        model.eval()
        total_loss = 0.0
        total_displacement_loss = 0.0
        total_velocity_loss = 0.0
        total_l1_loss = 0.0
        total_acc_loss = 0.0

        with torch.no_grad():
            for inputs, displacements in validation_dataset:
                inputs = inputs.to(self.device)
                displacements = displacements.to(self.device)

                displacement_loss, velocity_loss, l1_loss, acc_loss, grad_l1_loss = self.compute_losses(model, inputs, displacements)
                total_displacement_loss += displacement_loss.item()
                total_velocity_loss += velocity_loss.item()
                total_l1_loss += l1_loss.item()
                total_acc_loss += acc_loss.item()
                total_loss += displacement_loss + self.lambda_velocity_int * velocity_loss + self.lambda_gradient * acc_loss
            average_displacement_loss = total_displacement_loss / len(validation_dataset)
            average_velocity_loss = total_velocity_loss / len(validation_dataset)
            average_total_loss = total_loss / len(validation_dataset)

            if not physics_flag:
                return average_displacement_loss
            else:
                return average_total_loss

    def physics_train_step(self, model, training_dataset):
        model.train()
        total_loss = 0.0
        for inputs, displacements in training_dataset:
            inputs = inputs.to(self.device)
            displacements = displacements.to(self.device)

            self.physics_opt.zero_grad()
            self.physics_grad_opt.zero_grad()

            displacement_loss, velocity_loss, l1_loss, acc_loss, grad_l1_loss = self.compute_losses(model, inputs, displacements)
            total_loss_step = self.lambda_velocity_int * velocity_loss + self.lambda_l1 * l1_loss + self.lambda_gradient * acc_loss + self.lambda_grad_l1 * grad_l1_loss

            total_loss_step.backward()
            self.physics_opt.step()
            self.physics_grad_opt.step()

            total_loss += total_loss_step.item()

        return total_loss / len(training_dataset)

    def physics_validate_step(self, model, validation_dataset):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, displacements in validation_dataset:
                inputs = inputs.to(self.device)
                displacements = displacements.to(self.device)

                displacement_loss, velocity_loss, l1_loss, acc_loss, grad_l1_loss = self.compute_losses(model, inputs, displacements)
                total_loss_step = self.lambda_velocity_int * velocity_loss + self.lambda_l1 * l1_loss + self.lambda_gradient * acc_loss + self.lambda_grad_l1 * grad_l1_loss

                total_loss += total_loss_step.item()

        return total_loss / len(validation_dataset)

    def load_and_visualize_model(self, model, validation_dataset, clean_data, figure_suffix):
        # load the best model
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.eval()

        # collect predicted result
        excitation_all = []
        predicted_displacement_all = []
        predicted_velocity_all = []

        with torch.no_grad():
            for batch in validation_dataset:
                inputs = batch[0].to(self.device)
                displacement_hat, velocity_hat = model.predict(inputs)
                excitation_all.append(inputs.cpu())
                predicted_displacement_all.append(displacement_hat.cpu())
                predicted_velocity_all.append(velocity_hat.cpu())

        # transfer to numpy
        excitation_all = torch.cat(excitation_all, dim=0).numpy()
        predicted_displacement_all = torch.cat(predicted_displacement_all, dim=0).numpy()
        predicted_velocity_all = torch.cat(predicted_velocity_all, dim=0).numpy()

        # visualization
        plot_prediction(
            predicted_displacement_all,
            clean_data["displacement_clean"],
            "Predicted displacement",
            f"{self.figure_dir}/displacement_hat_{figure_suffix}.png"
        )
        plot_prediction(
            predicted_velocity_all,
            clean_data["velocity_clean"],
            "Predicted velocity",
            f"{self.figure_dir}/velocity_hat_{figure_suffix}.png"
        )
        plt.close('all')

        return model, excitation_all, predicted_displacement_all, predicted_velocity_all

    def train_iteration(self, iteration, model, function_acceleration,
                        train_dataset, validation_dataset, clean_data):
        physics_epochs = self.cfg.training.physics_epochs
        network_epochs = self.cfg.training.network_epochs

        physics_train_loss = []
        physics_val_loss = []
        network_train_loss = []
        network_val_loss = []

        # physics training phase
        self.logger.info(3 * "----------------------------------")
        self.logger.info(f"Iteration {iteration + 1}, Physics training")
        for epoch in range(physics_epochs):
            # update lr
            lr = self.physics_scheduler.get_lr(epoch)
            for param_group in self.physics_opt.param_groups:
                param_group['lr'] = lr
            for param_group in self.physics_grad_opt.param_groups:
                param_group['lr'] = lr

            # training steps
            train_loss = self.physics_train_step(model, train_dataset)
            val_loss = self.physics_validate_step(model, validation_dataset)

            physics_train_loss.append(train_loss)
            physics_val_loss.append(val_loss)

            # save the best model
            if val_loss < self.physics_loss_minimum:
                self.physics_loss_minimum = val_loss
                torch.save(model.state_dict(), self.checkpoint_path)

            # record logs
            if epoch % 100 == 0:
                self.logger.info(
                    f"Physics Epoch: {epoch + 1}/{physics_epochs} | "
                    f"LR: {lr:.6f} | "
                    f"Train Loss: {train_loss:.4e} | "
                    f"Val Loss: {val_loss:.4e}"
                )

        # plot loss curve
        plot_loss_curve(
            physics_train_loss, physics_val_loss,
            "Physics Training", f"{self.figure_dir}/physics_loss_{iteration + 1}.png"
        )

        # update the parameters of physics model
        model.clip_variables()
        lambda_acceleration = [param for name, param in model.named_parameters() if "cx" in name]
        function_acceleration = self.library.get_functions(lambda_acceleration)
        self.logger.info(f"Updated acceleration function: {function_acceleration}")
        function_acceleration = self.library.build_functions(lambda_acceleration)

        lambda_grad = [param for name, param in model.named_parameters() if "grad" in name]
        grad_expression = self.grad_library.get_functions(lambda_grad)
        self.logger.info(f"Updated acceleration_grad function: {grad_expression}")
        grad_expression = self.grad_library.build_functions(lambda_grad)
        model.update_function(function_acceleration, grad_expression)

        # network training phase
        self.logger.info(3 * "----------------------------------")
        self.logger.info(f"Iteration {iteration + 1}, Network training")
        for epoch in range(network_epochs):
            # update lr
            lr = self.network_scheduler.get_lr(epoch)
            for param_group in self.network_opt.param_groups:
                param_group['lr'] = lr

            # training steps
            train_loss = self.network_train_step(model, train_dataset)
            val_loss = self.network_validate_step(model, validation_dataset)

            network_train_loss.append(train_loss)
            network_val_loss.append(val_loss)

            # save the best model
            if val_loss < self.network_loss_minimum:
                self.network_loss_minimum = val_loss
                torch.save(model.state_dict(), self.checkpoint_path)

            # record logs
            if epoch % 100 == 0:
                self.logger.info(
                    f"Network Epoch: {epoch + 1}/{network_epochs} | "
                    f"LR: {lr:.6f} | "
                    f"Train Loss: {train_loss:.4e} | "
                    f"Val Loss: {val_loss:.4e}"
                )

        network_train_loss_cpu = [tensor.cpu() if torch.is_tensor(tensor) else tensor for tensor in
                                   network_train_loss]
        network_val_loss_cpu = [tensor.cpu() if torch.is_tensor(tensor) else tensor for tensor in
                                   network_val_loss]
        # plot loss curve
        plot_loss_curve(
            network_train_loss_cpu, network_val_loss_cpu,
            "Network Training", f"{self.figure_dir}/network_loss_{iteration + 1}.png"
        )

        # visualize current model
        model, excitation, displacement, velocity = self.load_and_visualize_model(
            model, validation_dataset, clean_data, f"AO_{iteration + 1}"
        )

        return model, excitation, displacement, velocity

    def training(self, train_dataset, validation_dataset,
                 Phi_int, Phi_diff, clean_data, max_values):
        # initialize model
        lambda_acceleration = np.ones(self.library.terms_number)
        function_acceleration = self.library.build_functions(lambda_acceleration)
        lambda_grad = np.ones(self.grad_library.terms_number)
        grad_expression = self.grad_library.build_functions(lambda_grad)
        model = PNetwork(
            self.cfg,
            Phi_int=Phi_int,
            Phi_diff=Phi_diff,
            number_library_terms=self.library.terms_number,
            number_grad_library_terms=self.grad_library.terms_number,
            function_acceleration=function_acceleration,
            grad_expression=grad_expression,
            max_values=max_values
        ).to(self.device)

        # initialize optimizer
        self.network_opt = optim.Adam(model.group_variables()[0], lr=self.cfg.training.network_initial_lr)
        self.physics_opt = optim.Adam(model.group_variables()[1], lr=self.cfg.training.physics_initial_lr)
        self.physics_grad_opt = optim.Adam(model.group_variables()[2], lr=self.cfg.training.physics_initial_lr)
        self.pretrain_opt = optim.Adam(model.group_variables()[0], lr=self.cfg.training.pretrain_lr)

        # pretraining phase
        self.logger.info("Pretraining the displacement network")
        pretrain_train_loss = []
        pretrain_val_loss = []
        for epoch in range(self.cfg.training.pretrain_epochs):
            train_loss = self.network_train_step(model, train_dataset, physics_flag=False)
            val_loss = self.network_validate_step(model, validation_dataset, physics_flag=False)

            pretrain_train_loss.append(train_loss)
            pretrain_val_loss.append(val_loss)

            if val_loss < self.pretrain_loss_minimum:
                self.pretrain_loss_minimum = val_loss
                torch.save(model.state_dict(), self.checkpoint_path)

            if epoch % 100 == 0:
                self.logger.info(
                    f"Pretrain Epoch: {epoch + 1}/{self.cfg.training.pretrain_epochs} | "
                    f"Train Loss: {train_loss:.4e} | "
                    f"Val Loss: {val_loss:.4e}"
                )

        plot_loss_curve(
            pretrain_train_loss, pretrain_val_loss,
            "Pretraining", f"{self.figure_dir}/pretrain_loss.png"
        )

        (model, excitation_all, displacement_all, velocity_hat_all) = (
            self.load_and_visualize_model(
                model,
                validation_dataset,
                clean_data,
                f"pretrain",
            )
        )
        for iteration in range(self.cfg.training.alternate_number):
            (
                model,
                excitation_all,
                displacement_all,
                velocity_hat_all,
            ) = self.train_iteration(
                iteration,
                model,
                function_acceleration,
                train_dataset,
                validation_dataset,
                clean_data,
            )

        return {
            "model": model,
            "excitation_all": excitation_all,
            "displacement_all": displacement_all,
            "velocity_hat_all": velocity_hat_all,
        }