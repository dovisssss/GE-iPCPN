import numpy as np
import os
import torch
from tensorflow_probability import optimizer
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.model.physics_network import PhysicsNetwork as PNetwork, PhysicsNetwork
from src.model.custom_scheduler import CustomScheduler as Schedule
from src.model.function_library import FunctionLibrary as Library


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
        for btach_idx, (inputs, targets) in enumerate(dataloader):
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
        for btach_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.physics_opt.zero_grad()

            _, velocity_loss, l1_loss = self.compute_losses(model, inputs, targets)

            loss = self.lamda_velocity_int * velocity_loss + self.lamda_l1 * l1_loss

            loss.backward()
            self.physics_opt.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

"""

    def validate(self, model, dataloader, mode='physics'):
        model.eval()
        disp_loss = 0.0
        vel_loss = 0.0
        l1_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                displacement_loss, velocity_loss, l1 = self.compute_losses(model, inputs, targets)
                disp_loss += displacement_loss.item()
                vel_loss += velocity_loss.item()
                l1_loss += l1.item()

        avg_disp_loss = disp_loss / len(dataloader)
        avg_vel_loss = vel_loss / len(dataloader)
        avg_l1_loss = l1_loss / len(dataloader)

        if mode == 'pretrain':
            return avg_disp_loss
        else:
            avg_loss = avg_disp_loss + self.lamda_velocity_int * avg_vel_loss
            return avg_loss, vel_loss, l1_loss
"""
































