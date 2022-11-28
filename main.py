import os
from pathlib import Path
from time import time

import torch
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, ExponentialLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from aidiag.model import DicomUNet
from aidiag.dataset import ImageFolder3D
from aidiag.metrics import dice_coef, dice_loss
from aidiag.utils import set_seeds
from aidiag.visualisation import save_and_show_history_graph
from aidiag.training import get_dataloaders, fit_epoch, eval_epoch
from aidiag.preprocessing import load_sample


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
MODEL_SCALE = 2
SOURCE_FOLDER = '/home/maksim/AIDiagnostic/subset/'


def main():
    set_seeds(0)

    names = []
    for folder in Path(os.path.join(SOURCE_FOLDER, 'subset')).glob('*'):
        names.append(folder.name)

    train_transforms = A.Compose([A.HorizontalFlip(p=0.5), ToTensorV2()])
    val_transforms = A.Compose([ToTensorV2()])

    train_names, val_names = train_test_split(names, test_size=0.2)

    train_dataset = ImageFolder3D(SOURCE_FOLDER, train_names, transforms=train_transforms, load_sample_fn=load_sample)

    val_dataset = ImageFolder3D(SOURCE_FOLDER, val_names, transforms=val_transforms, load_sample_fn=load_sample)

    train_loader, val_loader = get_dataloaders(BATCH_SIZE,
                                               train_set=train_dataset,
                                               valid_set=val_dataset,
                                               device=DEVICE,
                                               sampler=None)

    torch.cuda.empty_cache()
    history = train(model=DicomUNet(scale=MODEL_SCALE),
                    loss_fn=dice_loss, device=DEVICE,
                    data_tr=train_loader, data_val=val_loader,
                    path='output/best_weights.pl',
                    epochs=80, max_lr=2e-2, warmup_epochs=6, gamma=0.93)

    save_and_show_history_graph(history, path='output/graph.png')


def train(model, loss_fn, data_tr, data_val, path, epochs, device, max_lr=3e-2, warmup_epochs=6, gamma=0.9):
    history = []
    opt = torch.optim.Adam(model.parameters(), lr=max_lr * gamma ** -warmup_epochs)
    scheduler = ChainedScheduler([LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs),
                                  ExponentialLR(opt, gamma=gamma)])
    best_dice = 1.0
    model.to(device)
    for epoch in range(epochs):

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Lr: {scheduler.get_last_lr()[0]:.4f}')

        tic = time()
        metrics_train = fit_epoch(model, opt, loss_fn, data_tr, [dice_coef, ])
        scheduler.step()

        print(f'Train_loss: {metrics_train["loss"]:.4f}')
        print(f'Train_Dice: {metrics_train["dice_coef"]:.4f}')

        metrics_val = eval_epoch(model, loss_fn, data_val, [dice_coef, ])
        toc = time()

        print(f'Val_loss: {metrics_val["loss"]:.4f}')
        print(f'Val_Dice: {metrics_val["dice_coef"]:.4f}')
        print(f'Time: {(toc - tic):.1f} s.')

        history.append((metrics_train["dice_coef"], metrics_val["dice_coef"]))

        if metrics_val["dice_coef"] < best_dice:
            best_dice = metrics_val["dice_coef"]
            torch.save(model.state_dict(), path)
            print('Weights saved.')

        print()

    return history


if __name__ == '__main__':
    main()
