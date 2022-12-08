from BatchSampler import BatchSampler
from ImageDataset import ImageDataset
from Net import Net
from Train_Test import train_model, test_model

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import matplotlib.pyplot as plt
import os


def main():

    train_dataset = ImageDataset("data/X_train.npy", "data/Y_train.npy")
    test_dataset = ImageDataset("data/X_test.npy", "data/Y_test.npy")

    model = Net(n_classes=6)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    n_epochs = 3
    batch_size = 25

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif (
        torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=False
    )
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=False)

    mean_losses_train = []
    mean_losses_test = []
    accuracies = []

    for e in range(n_epochs):
        # Training:
        losses = train_model(model, train_sampler, optimizer, loss_function, device)
        # Calculating and printing statistics:
        mean_loss = sum(losses) / len(losses)
        mean_losses_train.append(mean_loss)
        print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

        # Testing:
        losses = test_model(model, test_sampler, loss_function, device)
        # Calculating and printing statistics:
        mean_loss = sum(losses) / len(losses)
        mean_losses_test.append(mean_loss)
        print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")
        # Plotting the historic loss: #TODO: Add plotting functionality
        # fig, ax = plt.subplots()
        # ax.plot(mean_losses_train, label="Train loss")
        # ax.plot(mean_losses_test, label="Test loss")
        # ax.legend()
        # plt.show()

    # Saving the model
    if os.path.exists(os.path.join(os.getcwd() + "model_weights/")):

        torch.save(model.state_dict(), "model_weights/weights_model.txt")
    else:
        os.mkdir(os.path.join(os.getcwd() + "model_weights/"))
        torch.save(model.state_dict(), "model_weights/weights_model.txt")


if __name__ == "__main__":
    main()
