import matplotlib.pyplot as plt
import torch

def plot_training_data(clean_data, noisy_data, title, training_figure_path):
    plt.rcParams["font.family"] = "Times New Roman"
    clean_instance = clean_data[0].cpu().numpy() if torch.is_tensor(clean_data[0]) else clean_data[0]
    noisy_instance = noisy_data[0].cpu().numpy() if torch.is_tensor(noisy_data[0]) else noisy_data[0]

    rms_1 = torch.sqrt(torch.mean((torch.tensor(clean_instance) - torch.tensor(noisy_instance)) ** 2)).item()
    rms_2 = torch.sqrt(torch.mean(torch.tensor(clean_instance) ** 2)).item()
    rms_ratio = rms_1 / rms_2 if rms_2 != 0 else 0

    plt.figure()
    plt.plot(clean_instance, color="#205EA7",label="Clean x")
    plt.plot(noisy_instance, linestyle="--", color="#F8A0C9",label="Noisy x")
    plt.plot(clean_instance - noisy_instance,color="#650233", label="Error x")
    plt.grid(
        True,
        linestyle=":",
        color="gray",
        alpha=0.6,
        linewidth=0.5
    )
    plt.title(f"Training {title}, {rms_ratio:.4f}")
    plt.legend()
    plt.savefig(training_figure_path)

def plot_loss_curve(loss_training_log, loss_validation_log, title,  training_figure_path):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()
    plt.semilogy(loss_training_log, color="#205EA7",label="Training Loss")
    plt.semilogy(loss_validation_log, color="#650233",label="Validation Loss")
    plt.grid(
        True,
        linestyle=":",
        color="gray",
        alpha=0.6,
        linewidth=0.5
    )
    plt.title(f"Loss curve of {title}")
    plt.legend()
    plt.savefig(training_figure_path)

def plot_prediction(prediction, clean_data, title, prediction_figure_path):
    plt.rcParams["font.family"] = "Times New Roman"

    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    if torch.is_tensor(clean_data):
        clean_data = clean_data.detach().cpu().numpy()

    prediction_instance = prediction[0]
    clean_instance = clean_data[0]

    rms_1 = torch.sqrt(torch.mean((torch.tensor(clean_instance) - torch.tensor(prediction)) ** 2)).item()
    rms_2 = torch.sqrt(torch.mean(torch.tensor(clean_instance) ** 2)).item()
    rms_ratio = rms_1 / rms_2 if rms_2 != 0 else 0

    plt.figure()
    plt.plot(clean_instance, color="#205EA7", label="True")
    plt.plot(prediction_instance, linestyle="--", color="#F8A0C9", label="Predicted")
    plt.plot(prediction_instance - clean_instance, color="#650233",label="Error")
    plt.grid(
        True,
        linestyle=":",
        color="gray",
        alpha=0.6,
        linewidth=0.5
    )
    plt.title(f" {title} Prediction, RMS {rms_ratio:.4f}")
    plt.legend()
    plt.savefig(prediction_figure_path)
    plt.close()