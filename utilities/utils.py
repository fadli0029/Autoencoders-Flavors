import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

def flatten(X, dtype="np.ndarray"):
    '''
    Given `X` of shape (N, C, H, W), this function 'flattens'
    the `C`, `H`, and `W` dimension of `X` into a single
    vector C*H*W per image.

    Args:
        X: `X` is an image dataset shape (N, C, H, W), where:
                N := Number of datapoints.
                C := Number of channels (ex: C=3 for RGB channel).
                H := Height of image.
                W := Width of image.
            For example, sampling batches of 64 of MNIST dataset will
            give: (64, 1, 28, 28) since the MNIST dataset are black
            and white images (1 channel) of size 28x28 pixels.
        dtype: A string, "np.ndarray" or "torch.tensor"
    Returns:
        X: Flatten version of `X`
    Raises:
        AssertionError: If `X` is not of shape (N, C, H, W)
    '''
    assert torch.is_tensor(X) or type(X) == np.ndarray
    assert len(X.shape) == 4, \
            'Input X must be of shape (N, C, H, W) to use flatten(X)'
    assert dtype == "torch.tensor" or dtype == "np.ndarray"

    N = X.shape[0]
    if dtype == "torch.tensor":
        return X.view(N, -1)
    elif dtype == "np.ndarray":
        return X.reshape(N, -1)

def visualize_dataset(data, dtype, n_samples=25, figsize=(10, 10)):
    '''
    Randomly pick datapoints in `data` and visualize (plot)
    them as a MxN matplotlib grid where MxN evaluates to n_samples.

    Args:
        data: A datasets.XXX object or a numpy
              ndarray, where XXX is MNIST, CIFAR10, etc.
        dtype: datasets.XXX (where XXX is MNIST for example) or
               np.ndarray
        n_samples: An integer that must be a perfect square.
    Returns:
        None.
    Raises:
        TODO
    '''
    assert isinstance(n_samples, int)
    # Check if n_samples is a perfect square:
    psq = np.sqrt(n_samples)
    assert int(str(psq).split('.')[1]) == 0

    samples = [random.randint(0, len(data)) for _ in range(n_samples)] # Generate random samples
    M = N = int(psq)
    if not isinstance(data, np.ndarray):
        fig = plt.figure(figsize=figsize)
        for i in range(n_samples):
            (x, y) = data[samples[i]]
            img = x.squeeze()
            label = y
            
            fig.add_subplot(M, N, i+1)
            plt.imshow(img, cmap="gray")
            plt.axis('off')
            plt.title(y)
    else:
        return None



