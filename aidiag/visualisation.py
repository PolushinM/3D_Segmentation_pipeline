import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def show_image(image: torch.tensor, n_max: int = 200):
    """Plot sliced 3D image."""
    fig, ax = plt.subplots(figsize=(60, 60))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(image.detach()[:n_max], nrow=8).permute(1, 2, 0))


def save_and_show_history_graph(history: list[tuple[float, float]], path: str = '') -> None:
    """Show and save history graph, showing how the metrics changed during the learning process."""
    train_loss, val_loss = zip(*history)
    plt.figure(figsize=(18, 6))
    plt.plot(train_loss, label="Train_Dice")
    plt.plot(val_loss, label="Val_Dice")
    plt.legend(loc='best')
    plt.xlabel("Epochs")
    plt.ylabel("Dice")
    if path != '':
        plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.show()
