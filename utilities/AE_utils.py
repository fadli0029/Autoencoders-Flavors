import torch
import random
import numpy as np
from utils import flatten
import matplotlib.pyplot as plt
import torch.nn.functional as F

def visualize(
        data_loader, 
        model,
        flatten_data=True,
        dtype=torch.float32,
        device=torch.device('cpu'),
        data_per_batch=64,
        n_samples=25,
        figsize=(10, 10)
    ):
    # TODO: docstrings for args, returns, raises.
    ''' 
    Get datapoints from `data_loader`, pass it to the model,
    then visualize (plot) them.
    '''
    assert isinstance(n_samples, int)
    
    # Check if n_samples is a perfect square:
    psq = np.sqrt(n_samples)
    assert int(str(psq).split('.')[1]) == 0
    M = N = int(psq)
    
    # Generate random samples
    samples = [random.randint(0, data_per_batch-1) \
               for _ in range(n_samples)]

    fig = plt.figure(figsize=figsize)
    for _, (x, y) in enumerate(data_loader):
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        
        if flatten_data:
            x = flatten(x)
        
        # Pass x to the model (encoder)
        x = model(x)
        
        # Don't forget to reshape x after
        # flattening it in order to display it
        # as an image.
        x = x.detach().cpu().numpy().reshape(x.size(0), 28, 28)
        
        for i in range(n_samples):
            # pick a random (x, y) pair from 
            # the 64 datapoints in `batch`
            x_ = x[samples[i]]
            y_ = y[samples[i]]
            img = x_.squeeze()
            label = y_.item()
            
            fig.add_subplot(M, N, i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(label)
            
        break
    plt.show()
        
def evaluate_performance(
        data_loader, 
        model,
        loss_fn=F.mse_loss, 
        flatten_data=True,
        dtype=torch.float32, 
        device=torch.device('cpu')
    ):
    # TODO: docstrings for desc, args, returns, raises.
    '''
    Desc.
    '''
    if data_loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device=device, dtype=dtype)

            if flatten_data:
                x = flatten(x)

            scores = model(x)
            loss = loss_fn(x, scores)

        print("loss: %.3f" % loss)        
    visualize(data_loader, model, flatten_data=flatten_data, 
              device=device, dtype=dtype)

def train(
        model,
        optimizer,
        loader_train,
        loader_val,
        loss_fn=F.mse_loss,
        flatten_data=False,
        print_every=100,
        dtype=torch.float32,
        device=torch.device('cpu'),
        epochs=1
    ):
    # TODO: docstrings for desc, args, returns, raises.
    '''
    Training pipeline/loop for PyTorch
    implementations.

    Args:
        model: The model/architecture, an `torch.nn` object.
        optimizer: The optimzer used, an `torch.optim` object.
        loader_train: DataLoader object for training set.
        loader_val: DataLoader object for validation set.
        loss_fn: Loss function of type `torch.nn.functional`
        flatten_data: Boolean whether to flatten input or not.
        print_every: How often to print accuracy and loss.
        dtype: The datatype of the tensors
        device: Where to place the tensors, cpu or cuda.
        epochs: Number of epochs.

    Returns:
        None

    Raises
    '''
    model = model.to(device=device)
    print("Training for {} epochs\n".format(epochs))
    for epoch in range(epochs):
        for batch, (x, _) in enumerate(loader_train):
            model.train()

            x = x.to(device=device, dtype=dtype)

            if flatten_data:
                # Flatten x.
                x = flatten(x)

            scores = model(x)
            loss = loss_fn(x, scores)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if batch % print_every == 0:
                evaluate_performance(loader_val, model, dtype=dtype, device=device)
                print()

        print('{} epochs completed...\n'.format(epoch+1))
