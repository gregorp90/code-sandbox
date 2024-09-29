import matplotlib.pyplot as plt
import numpy as np


def plot_loss_vs_val_loss(data: dict, log_scale: bool = False):
    """
    This function takes a dictionary with 'epoch_num', 'loss', and 'loss_val' keys and
    plots loss as a solid blue line and validation loss as a dashed orange line.

    Args:
    - data (dict): A dictionary containing 'epoch_num', 'loss', and 'loss_val' as keys.
    """
    # Copy data to avoid modifying original values
    loss = np.array(data['loss'])
    loss_val = np.array(data['loss_val'])

    # Plotting
    plt.plot(data['epoch_num'], loss, label='Loss', color='blue', linestyle='-')
    plt.plot(data['epoch_num'], loss_val, label='Validation Loss', color='orange', linestyle='--')

    # Adding labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Validation Loss per Epoch')
    plt.legend()

    # Apply log scale to the y-axis if requested
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Loss (Log Scale)')

    # Show plot
    plt.show()