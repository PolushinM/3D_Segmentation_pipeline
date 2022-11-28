from torch.utils.data import Dataset


class ImageFolder3D(Dataset):
    """Segmentation dataset for 3D tomography images with masks.
            Parameters
            ----------
            directory : {str} directory with data (images and masks).
            names : {list[str]} list of names, corresponding to subfolders names.
            transforms : "albumentations" transformation, which will be applied to both image and mask for every sample.
            load_sample_fn(directory, name) : external method for sample extraction from
                data directory with necessary preprocession.
            """

    def __init__(self, directory: str, names: list, transforms, load_sample_fn: callable):
        super().__init__()
        self.transforms = transforms
        self.names = names
        self.directory = directory
        self.load_sample = load_sample_fn

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index: int) -> tuple:
        """Load augmented and transformed sample.
                Parameters
                ----------
                index : {int} index of sample in "names" list.

                Returns
                -------
                sample : {tuple[torch.tensor, torch.tensor]} (image, mask)
                    Returns chosen sample after transformation.
                    Shape of both image and mask: 1*D*W*H (dept, width, height).
                """
        x, y = self.load_sample(self.directory, self.names[index])
        transform = self.transforms(image=x, mask=y)
        image = transform['image'].moveaxis(0, 2)[None, ...].float()
        mask = transform['mask'][None, ...].float()
        return image, mask
