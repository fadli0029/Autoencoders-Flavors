import torch
import random
import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader

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

def get_optimal_num_workers(data, num_epochs=3, batch_size=64, VERBOSE=False):
    '''
    Find the optimal number of workers (i.e: processors) to use
    during data loading, where optimality is measured based on 
    time taken to load the data using torch DataLoader.

    Note: For small dataset, it might be unnecessary to use
    multiple processors.

    Args:
        data: A torch datasets.XXX object.
        num_epochs: For how many epochs our loading `data`.
        batch_size: How many datapoints to load per batch.
    Returns:
        num_workers: Optimal number of workers.
    Raises:
        TODO
    '''
    print('Finding the best number of workers for dataloader...')
    print('\nP/S: Do not panick with the warning that says: \
           \n[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool). \
           \nThis is a known issue and it is not something you should be worried about. \
           \nSee: https://github.com/pytorch/pytorch/issues/57273')
    record = {}
    for num_workers in range(2, mp.cpu_count() + 1, 2):
        loader_TRAIN = DataLoader(data, batch_size=batch_size,
                shuffle=True, num_workers=num_workers, pin_memory=True)
        start = time.perf_counter()
        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(loader_TRAIN):
                pass
        end = time.perf_counter()
        record[num_workers] = round(end-start, 3)

    if VERBOSE:
        for num_workers, t in record.items():
            print('Number of workers: {} | Time taken: {}'.format(num_workers, t))

    best_num_workers = min(record, key=record.get)
    worst_num_workers = max(record, key=record.get)
    best_time_taken = record[best_num_workers]
    worst_time_taken = record[worst_num_workers]
    diff = round(worst_time_taken - best_time_taken, 3)

    print('\nBest mumber of workers for DataLoader is {} \
            \nwith time taken {}s, about {}s faster \
            \nthan the slowest number of workers ({}).'\
            .format(best_num_workers, best_time_taken, diff, worst_num_workers))

    return best_num_workers



