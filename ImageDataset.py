import numpy as np
import torch

class ImageDataset():
    """
    Creates a DataSet from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torchtensors when needed (torch tensors are the objects used
    to pass the data through the neural network and apply weights).
    """

    def __init__(self, x, y, transform=None, target_transform=None):
        self.targets = self.load_numpy_arr_from_npy(y)
        self.imgs = self.load_numpy_arr_from_npy(x)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        label = self.targets[idx]
        return image, label


    def load_numpy_arr_from_url(self, url):
        """
        Loads a numpy array from surfdrive.

        Input:
        url: Download link of dataset

        Outputs:
        dataset: numpy array with input features or labels
        """

        response = requests.get(url)
        response.raise_for_status()

        return np.load(io.BytesIO(response.content))

    def load_numpy_arr_from_npy(self, path):
        """
        Loads a numpy array from surfdrive.

        Input:
        path: local path of file

        Outputs:
        dataset: numpy array with input features or labels
        """

        return np.load(path)