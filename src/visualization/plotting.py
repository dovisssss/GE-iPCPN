import matplotlib.pyplot as plt
import torch

def plot_training_data(clean_data, noisy_data, title, training_figure_path):
    clean_instance = clean_data[0].cpu().numpy() if torch.is_tensor(clean_data[0]) else clean_data[0]
    noisy_instance = noisy_data[0].cpu().numpy() if torch.is_tensor(noisy_data[0]) else noisy_data[0]

    rms_1 = torch.sqrt(torch.mean((torch.tensor(clean_instance) - torch.tensor(noisy_instance)) ** 2)).item()
    rms_2 = torch.sqrt(torch.mean(torch.tensor(clean_instance) ** 2)).item()
    rms_ratio = rms_1 / rms_2 if rms_2 != 0 else 0

    plt.figure()
    plt.plot(clean_instance, label="Clean x")
    plt.plot(noisy_instance, label="Noisy x")
    plt.plot(clean_instance - noisy_instance, label="Error x")
    plt.title(f"Training {title}, {rms_ratio:.4f}")
    plt.legend()
    plt.savefig(training_figure_path)