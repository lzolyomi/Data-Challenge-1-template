# Custom imports
from BatchSampler import BatchSampler
from ImageDataset import ImageDataset
from Net import Net
from Train_Test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

# Other imports
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import argparse
import plotext
from datetime import datetime


def main(args:argparse.Namespace) -> None:

    # Load the train and test data set
    train_dataset = ImageDataset("data/X_train.npy", "data/Y_train.npy")
    test_dataset = ImageDataset("data/X_test.npy", "data/Y_test.npy")

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

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
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

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

        # # Calculating and printing statistics:
        mean_loss = sum(losses) / len(losses)
        mean_losses_test.append(mean_loss)
        print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

        ### Plotting during training
        plotext.clf()
        plotext.scatter(mean_losses_train, label="train")
        plotext.scatter(mean_losses_test, label="test")
        plotext.title("Train and test loss")

        plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

        plotext.show()

    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not os.path.exists(os.path.join(os.getcwd() + "/model_weights/")):
        os.mkdir(os.path.join(os.getcwd() + "/model_weights/"))
    
    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")
    
    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()
    
    # Check if /artifacts/ subdir exists
    if not os.path.exists(os.path.join(os.getcwd() + "/artifacts/")):
        os.mkdir(os.path.join(os.getcwd() + "/artifacts/"))

    # save plot of losses
    fig.savefig(f"artifacts/session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")
    
    
    
    # Check if /summaries/ subdir exists
    if not os.path.exists(os.path.join(os.getcwd() + "/summaries/")):
        os.mkdir(os.path.join(os.getcwd() + "/summaries/"))
    
    #Saving model summary and parser arguments
    summary_str = str(summary(model, (1, 128, 128), device=device, verbose=0))
    with open(f"summaries/summary_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt", "w", encoding="utf-8") as text_file:
        text_file.write(summary_str)
        text_file.write('\n\n')
        text_file.write(str(args))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=10, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        action=argparse.BooleanOptionalAction,
        type=bool
    )

    args = parser.parse_args()

    main(args)
