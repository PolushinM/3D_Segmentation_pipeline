from typing import List, Union

import torch
from torch.utils.data import DataLoader


def to_device(data: Union[list[torch.tensor], tuple[torch.tensor], torch.tensor], device: torch.device) -> Union[
    list[torch.tensor], tuple[torch.tensor], torch.tensor]:
    """Move tensor(s) to chosen device.
            Parameters
            ----------
            data : {Union[list[torch.tensor]} collection of train/test data.
            device: {torch.device} the device (cuda or cpu) to which the data will be moved.

            Returns
            -------
            data : {Union[list[torch.tensor], tuple[torch.tensor], torch.tensor]}  Returns the collection of
                train/test data, moved to chosen device.
            """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader(object):
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dataloader: torch.utils.data.dataloader, device: torch.device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self) -> torch.Tensor:
        """Yield a batch of data after moving it to device"""
        for batch in self.dataloader:
            yield to_device(batch, self.device)

    def __len__(self) -> int:
        """Number of batches"""
        return len(self.dataloader)


def get_dataloaders(batch_size: int, train_set: torch.utils.data.dataset, valid_set: torch.utils.data.dataset,
                    device: torch.device, sampler: torch.utils.data.Sampler = None, num_workers: int = 5) -> tuple[
    torch.utils.data.dataloader, torch.utils.data.dataloader]:
    """Get wrapped train and validation device dataloaders.
            Parameters
            ----------
            batch_size : {int} Batch size for train and validation dataloaders.
            train_set : {torch.utils.data.dataset}
            valid_set : {torch.utils.data.dataset}
            device : {torch.device} the device (cuda or cpu) to which the data will be moved.
            sampler : {torch.utils.data.Sampler} batch sampler, default "None".
            num_workers : {int} number of threads fo loading data, default 5.

            Returns
            -------
            dataloaders : {tuple[torch.utils.data.dataloader, torch.utils.data.dataloader]}
                Tuple of train dataloader and validation dataloader.
            """
    shuffle = sampler is None
    train_dataloader = DataLoader(train_set, batch_size, shuffle=shuffle, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return DeviceDataLoader(train_dataloader, device), DeviceDataLoader(valid_dataloader, device)


def fit_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: callable,
              data: torch.utils.data.dataloader, metrics: List[callable]) -> dict[str, float]:
    """Fit model during one epoch.
            Parameters
            ----------
            model : {torch.nn.Module} Torch NN model to fit.
            optimizer : {torch.optim.Optimizer} Torch optimizer, associated with the model.
            loss_fn : {callable} Loss function.
            data : {torch.utils.data.dataloader} Train dataloader.
            metrics : {List[callable]} List of metric functions of
                the form mertic_func(targets: torch.tensor, inputs: torch.tensor).

            Returns
            -------
            metrics_dict : {dict[str, float]} Dict of calculated metrics: loss function value and all the metrics
                from "metrics" list, where keys in dict corresponds to function names.
            """
    metrics_dict = dict()
    metrics_dict['loss'] = 0.
    for metric in metrics:
        metrics_dict[metric.__name__] = 0.

    model.train()
    for X_batch, Y_batch in data:

        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = loss_fn(Y_batch, Y_pred)
        loss.backward()
        optimizer.step()

        metrics_dict['loss'] += loss.cpu().detach().numpy() / len(data)
        for metric in metrics:
            metrics_dict[metric.__name__] += metric(Y_batch.cpu().detach(), Y_pred.cpu().detach()) / len(data)

    return metrics_dict


def eval_epoch(model, loss_fn, data, metrics: List[callable]) -> dict[str, float]:
    """Eval (calculate loss and metrics values) model during one epoch.
            Parameters
            ----------
            model : {torch.nn.Module} Torch NN model to fit.
            loss_fn : {callable} Loss function.
            data : {torch.utils.data.dataloader} Validation dataloader.
            metrics : {List[callable]} List of metric functions of
                the form mertic_func(targets: torch.tensor, inputs: torch.tensor).

            Returns
            -------
            metrics_dict : {dict[str, float]} Dict of calculated metrics: loss function value and all the metrics
                from "metrics" list, where keys in dict corresponds to function names.
            """
    metrics_dict = dict()
    metrics_dict['loss'] = 0.
    for metric in metrics:
        metrics_dict[metric.__name__] = 0.

    model.eval()
    for X_batch, Y_batch in data:

        Y_pred = model(X_batch).detach()
        loss = loss_fn(Y_batch, Y_pred).cpu().detach().numpy()

        metrics_dict['loss'] += loss / len(data)
        for metric in metrics:
            metrics_dict[metric.__name__] += metric(Y_batch.cpu().detach(), Y_pred.cpu().detach()) / len(data)

    return metrics_dict
