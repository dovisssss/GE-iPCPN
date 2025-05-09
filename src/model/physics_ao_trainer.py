# physics_ao_trainer.py
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
from src.visualization.plotting import plot_prediction, plot_loss_curve


class Trainer:
    def __init__(self, cfg, logger, output_dir):
        self.cfg = cfg
        self.logger = logger
        self.library = Library()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 目录设置
        checkpoint_dir = os.path.join(output_dir, cfg.dirs.checkpoints)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, "model.pth")

        self.figure_dir = os.path.join(output_dir, cfg.dirs.figures)
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)

        # 损失函数和优化器占位符
        self.mse = nn.MSELoss()
        self.network_opt = None
        self.physics_opt = None
        self.pretrain_opt = None

        # 学习率调度器
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

        # 训练参数
        self.pretrain_loss_minimum = 1e10
        self.physics_loss_minimum = 1e10
        self.network_loss_minimum = 1e10
        self.lambda_velocity_int = cfg.training.lamda_velocity_int
        self.lambda_l1 = cfg.training.lamda_l1

    def compute_losses(self, model, input, output):
        normalized_displacement_hat, normalized_velocity_error = model.step_forward(input)
        displacement_loss = self.mse(output, normalized_displacement_hat)
        velocity_loss = self.mse(normalized_velocity_error, torch.zeros_like(normalized_velocity_error))
        l1_loss = torch.sum(torch.abs(torch.stack(model.physics_variables)))
        return displacement_loss, velocity_loss, l1_loss

    def network_train_step(self, model, training_dataset, physics_flag=True):
        model.train()
        total_loss = 0.0
        for inputs, displacements in training_dataset:
            inputs = inputs.to(self.device)
            displacements = displacements.to(self.device)

            # 梯度清零
            if physics_flag:
                self.network_opt.zero_grad()
            else:
                self.pretrain_opt.zero_grad()

            # 前向传播和损失计算
            displacement_loss, velocity_loss, l1_loss = self.compute_losses(model, inputs, displacements)

            # 计算总损失
            if not physics_flag:
                total_loss_step = displacement_loss
            else:
                total_loss_step = displacement_loss + self.lambda_velocity_int * velocity_loss

            # 反向传播和优化
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
        with torch.no_grad():
            for inputs, displacements in validation_dataset:
                inputs = inputs.to(self.device)
                displacements = displacements.to(self.device)

                displacement_loss, velocity_loss, l1_loss = self.compute_losses(model, inputs, displacements)

                if physics_flag:
                    total_loss_step = displacement_loss + self.lambda_velocity_int * velocity_loss
                else:
                    total_loss_step = displacement_loss

                total_loss += total_loss_step.item()

        return total_loss / len(validation_dataset)

    def physics_train_step(self, model, training_dataset):
        model.train()
        total_loss = 0.0
        for inputs, displacements in training_dataset:
            inputs = inputs.to(self.device)
            displacements = displacements.to(self.device)

            self.physics_opt.zero_grad()

            displacement_loss, velocity_loss, l1_loss = self.compute_losses(model, inputs, displacements)
            total_loss_step = self.lambda_velocity_int * velocity_loss + self.lambda_l1 * l1_loss

            total_loss_step.backward()
            self.physics_opt.step()

            total_loss += total_loss_step.item()

        return total_loss / len(training_dataset)

    def physics_validate_step(self, model, validation_dataset):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, displacements in validation_dataset:
                inputs = inputs.to(self.device)
                displacements = displacements.to(self.device)

                displacement_loss, velocity_loss, l1_loss = self.compute_losses(model, inputs, displacements)
                total_loss_step = self.lambda_velocity_int * velocity_loss + self.lambda_l1 * l1_loss

                total_loss += total_loss_step.item()

        return total_loss / len(validation_dataset)

    def load_and_visualize_model(self, model, validation_dataset, clean_data, figure_suffix):
        # 加载最优模型
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.eval()

        # 收集预测结果
        excitation_all = []
        predicted_displacement_all = []
        predicted_velocity_all = []

        with torch.no_grad():
            for batch in validation_dataset:
                inputs = batch[0].to(self.device)
                displacement_hat, velocity_hat = model.step_forward(inputs)
                excitation_all.append(inputs.cpu())
                predicted_displacement_all.append(displacement_hat.cpu())
                predicted_velocity_all.append(velocity_hat.cpu())

        # 转换为numpy数组
        excitation_all = torch.cat(excitation_all, dim=0).numpy()
        predicted_displacement_all = torch.cat(predicted_displacement_all, dim=0).numpy()
        predicted_velocity_all = torch.cat(predicted_velocity_all, dim=0).numpy()

        # 可视化
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

        # 物理训练阶段
        self.logger.info(3 * "----------------------------------")
        self.logger.info(f"Iteration {iteration + 1}, Physics training")
        for epoch in range(physics_epochs):
            # 更新学习率
            lr = self.physics_scheduler.get_lr(epoch)
            for param_group in self.physics_opt.param_groups:
                param_group['lr'] = lr

            # 训练步骤
            train_loss = self.physics_train_step(model, train_dataset)
            val_loss = self.physics_validate_step(model, validation_dataset)

            physics_train_loss.append(train_loss)
            physics_val_loss.append(val_loss)

            # 保存最佳模型
            if val_loss < self.physics_loss_minimum:
                self.physics_loss_minimum = val_loss
                torch.save(model.state_dict(), self.checkpoint_path)

            # 日志记录
            if epoch % 100 == 0:
                self.logger.info(
                    f"Physics Epoch: {epoch + 1}/{physics_epochs} | "
                    f"LR: {lr:.6f} | "
                    f"Train Loss: {train_loss:.4e} | "
                    f"Val Loss: {val_loss:.4e}"
                )

        # 绘制损失曲线
        plot_loss_curve(
            physics_train_loss, physics_val_loss,
            "Physics Training", f"{self.figure_dir}/physics_loss_{iteration + 1}.png"
        )

        # 更新物理模型参数
        model.clip_variables()
        lambda_acceleration = [param for name, param in model.named_parameters() if "cx" in name]
        function_acceleration = self.library.build_functions(lambda_acceleration)
        model.update_function(function_acceleration)

        # 网络训练阶段
        self.logger.info(3 * "----------------------------------")
        self.logger.info(f"Iteration {iteration + 1}, Network training")
        for epoch in range(network_epochs):
            # 更新学习率
            lr = self.network_scheduler.get_lr(epoch)
            for param_group in self.network_opt.param_groups:
                param_group['lr'] = lr

            # 训练步骤
            train_loss = self.network_train_step(model, train_dataset)
            val_loss, _, _ = self.network_validate_step(model, validation_dataset)

            network_train_loss.append(train_loss)
            network_val_loss.append(val_loss)

            # 保存最佳模型
            if val_loss < self.network_loss_minimum:
                self.network_loss_minimum = val_loss
                torch.save(model.state_dict(), self.checkpoint_path)

            # 日志记录
            if epoch % 100 == 0:
                self.logger.info(
                    f"Network Epoch: {epoch + 1}/{network_epochs} | "
                    f"LR: {lr:.6f} | "
                    f"Train Loss: {train_loss:.4e} | "
                    f"Val Loss: {val_loss:.4e}"
                )

        # 绘制损失曲线
        plot_loss_curve(
            network_train_loss, network_val_loss,
            "Network Training", f"{self.figure_dir}/network_loss_{iteration + 1}.png"
        )

        # 可视化当前模型
        model, excitation, displacement, velocity = self.load_and_visualize_model(
            model, validation_dataset, clean_data, f"AO_{iteration + 1}"
        )

        return model, excitation, displacement, velocity

    def training(self, train_dataset, validation_dataset,
                 Phi_int, Phi_diff, clean_data, max_values):
        # 初始化模型
        lambda_acceleration = np.ones(self.library.terms_number)
        function_acceleration = self.library.build_functions(lambda_acceleration)
        model = PNetwork(
            self.cfg,
            Phi_int=Phi_int,
            Phi_diff=Phi_diff,
            number_library_terms=self.library.terms_number,
            function_acceleration=function_acceleration,
            max_values=max_values
        ).to(self.device)

        # 初始化优化器
        self.network_opt = optim.Adam(model.network_params(), lr=self.cfg.training.network_initial_lr)
        self.physics_opt = optim.Adam(model.physics_params(), lr=self.cfg.training.physics_initial_lr)
        self.pretrain_opt = optim.Adam(model.network_params(), lr=self.cfg.training.pretrain_lr)

        # 预训练阶段
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