import scipy.io as sio
from einops import repeat
import numpy as np


def return_data(data_path, split_ratio):
    dataset = sio.loadmat(data_path)

    fs = dataset["fs"]
    fs = fs[0, 0]
    excitation_max = dataset["excitation_max"]
    excitation_max = excitation_max[0, 0]
    displacement_noisy_max = dataset["displacement_noisy_max"]
    displacement_noisy_max = displacement_noisy_max[0, 0]
    velocity_noisy_max = dataset["velocity_noisy_max"]
    velocity_noisy_max = velocity_noisy_max[0, 0]

    excitation = dataset["excitation_matrix"]
    displacement = dataset["displacement_matrix"]
    displacement_noisy = dataset["displacement_noisy_matrix"]
    velocity = dataset["velocity_matrix"]
    velocity_noisy = dataset["velocity_noisy_matrix"]

    excitation = repeat(excitation, "batch seq -> batch seq c", c=1)
    displacement = repeat(displacement, "batch seq -> batch seq c", c=1)
    displacement_noisy = repeat(displacement_noisy, "batch seq -> batch seq c", c=1)
    velocity = repeat(velocity, "batch seq -> batch seq c", c=1)
    velocity_noisy = repeat(velocity_noisy, "batch seq -> batch seq c", c=1)

    sample_number = displacement.shape[0]
    train_number = int(sample_number * split_ratio)

    excitation_train = excitation[0:train_number, :, :]
    excitation_test = excitation[train_number:, :, :]
    displacement_train = displacement_noisy[0:train_number, :, :]
    displacement_train_label = displacement[0:train_number, :, :]
    displacement_test = displacement_noisy[train_number:, :, :]
    displacement_test_label = displacement[train_number:, :, :]
    velocity_train = velocity_noisy[0:train_number, :, :]
    velocity_test = velocity_noisy[train_number:, :, :]
    velocity_train_label = velocity[0:train_number, :, :]
    velocity_test_label = velocity[train_number:, :, :]

    return {
        "fs": fs,
        "excitation_max": excitation_max,
        "displacement_noisy_max": displacement_noisy_max,
        "velocity_noisy_max": velocity_noisy_max,
        "excitation_train": excitation_train,
        "excitation_test": excitation_test,
        "displacement_train": displacement_train,
        "displacement_train_label": displacement_train_label,
        "displacement_test": displacement_test,
        "displacement_test_label": displacement_test_label,
        "velocity_train": velocity_train,
        "velocity_train_label": velocity_train_label,
        "velocity_test": velocity_test,
        "velocity_test_label": velocity_test_label,
    }


