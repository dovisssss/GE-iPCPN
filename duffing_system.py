import os
import scipy
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import return_data
from src.data.numercial_operator import integration_operator, differentiation_operator
from src.model.physics_ao_trainer import Trainer
from src.visualization.plotting import plot_training_data

# set GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)

def system_training(cfg, output_dir, logger):
    data_path = (
        f"./data/{cfg.data.type}_{cfg.data.system}_"
        + (f"{cfg.data.noise_ratio:.2f}" if cfg.data.noise_ratio < 1 else f"{cfg.data.noise_ratio}")
        + ".mat"
    )

    data = return_data.return_data(data_path, cfg.data.split_ratio)

    fs = data["fs"]
    dt = 1/fs
    excitation_max = torch.tensor(data["excitation_max"], device=device)
    excitation_train = torch.tensor(data["excitation_train"], device=device)
    excitation_validation = torch.tensor(data["excitation_test"], device=device)

    #displacement
    displacement_noisy_max = torch.tensor(data["displacement_noisy_max"], device=device)
    displacement_train = torch.tensor(data["displacement_train"], device=device)
    displacement_train_clean = torch.tensor(data["displacement_train_label"], device=device)
    displacement_validation = torch.tensor(data["displacement_test"], device=device)
    displacement_validation_clean = torch.tensor(data["displacement_test_label"], device=device)

    #velocity
    velocity_noisy_max = torch.tensor(data["velocity_noisy_max"], device=device)
    velocity_test_clean = torch.tensor(data["velocity_test_label"], device=device)

    figure_dir = os.path.join(output_dir, cfg.dirs.figures)
    os.makedirs(figure_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, cfg.dirs.results)
    os.makedirs(result_dir, exist_ok=True)

    plot_training_data(
        displacement_train_clean.cpu().numpy(),
        displacement_train.cpu().numpy(),
        "displacement",
        f"{figure_dir}/displacement_train.png",
    )

    batch_size = cfg.training.batch_size
    seq_len = excitation_train.shape[1]

    train_dataset = CustomDataset(excitation_train,displacement_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_dataset = CustomDataset(excitation_validation, displacement_validation)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    trainer = Trainer(cfg, logger, output_dir)

    batch_integration_operator = integration_operator(batch_size, seq_len, dt).to(device)
    batch_differentiation_operator = differentiation_operator(batch_size, seq_len, dt).to(device)

    max_values = {
        "excitation_max": excitation_max,
        "displacement_max": displacement_noisy_max,
        "velocity_max": velocity_noisy_max,
    }
    clean_data = {
        "displacement_clean": displacement_train_clean,
        "velocity_clean": velocity_test_clean,
    }

    # training已被修改，输入参数与返回参数均不一致
    trained_results = trainer.training(
        train_loader,
        val_loader,
        batch_integration_operator,
        batch_differentiation_operator,
        clean_data,
        max_values
    )


    displacement_all = trained_results["displacement_all"].cpu().numpy()
    velocity_hat_all = trained_results["velocity_hat_all"].cpu().numpy()
    result_path = f"{result_dir}/{cfg.data.system}_{cfg.data.type}_{cfg.data.noise_ratio}.mat"

    scipy.io.savemat(
        result_path,
        {
            "fs": fs,
            "dt": dt.cpu().numpy() if isinstance(dt, torch.Tensor) else dt,
            "excitation_max": excitation_max.cpu().numpy(),
            "excitation": excitation_validation.cpu().numpy(),
            "displacement_noisy_max": displacement_noisy_max.cpu().numpy(),
            "displacement_noisy": displacement_validation.cpu().numpy(),
            "displacement_clean": displacement_validation_clean.cpu().numpy(),
            "displacement_pred": displacement_all,
            "velocity_noisy_max": velocity_noisy_max.cpu().numpy(),
            "velocity_clean": velocity_test_clean.cpu().numpy(),
            "velocity_pred": velocity_hat_all,
        }
    )

































