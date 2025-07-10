import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary
from tqdm.auto import tqdm
import os
from models.config import *


class Classifier(nn.Module):
    def __init__(self, img_ch=1, hidden_ch=32, num_class=NUM_CLASS):
        super().__init__()

        # Define convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(img_ch, hidden_ch, 3)
        self.bn1 = nn.BatchNorm2d(hidden_ch)

        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch*2, 3)
        self.bn2 = nn.BatchNorm2d(hidden_ch*2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(hidden_ch*2, hidden_ch*4, 3)
        self.bn3 = nn.BatchNorm2d(hidden_ch*4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(hidden_ch*4, hidden_ch*8, 3)
        self.bn4 = nn.BatchNorm2d(hidden_ch*8)

        # Define fully connected layers
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(hidden_ch * 8 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        # Forward pass through the layers
        x = F.relu(self.bn1(self.conv1(x))) # -->(batch_size, channel, 30, 30)
        x = self.pool2(F.relu(self.bn2(self.conv2(x)))) # --->(batch_size, channel, 14, 14)
        x = self.pool3(F.relu(self.bn3(self.conv3(x)))) # --->(batch_size, 128, 6, 6)
        # The out put of pool3 layer have the desire dimention for feature extraction

        x = F.relu(self.bn4(self.conv4(x))) # # --->(batch_size, channel, 4, 4)
        
        # Flatten the output and pass through fully connected layers
        return self.fc2(self.fc1(self.flat(x)))


# Define accuracy function
def accuracy_fn(true, pred):
  correct = torch.eq(true, pred)
  return (torch.sum(correct)/len(true))*100


def classifier_dataset():

    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(H),
            torchvision.transforms.ToTensor()
        ])

    train_set = torchvision.datasets.MNIST('data/MNIST',
                                    train=True,
                                    download=True,
                                    transform=transform,
                                    target_transform=None)

    dev_set = torchvision.datasets.MNIST('data/MNIST',
                                    train=False,
                                    download=True,
                                    transform=transform,
                                    target_transform=None)


    train_loader = torch.utils.data.DataLoader(train_set,
                                        batch_size=64,
                                        shuffle=True)

    dev_loader = torch.utils.data.DataLoader(dev_set,
                                        batch_size=64,
                                        shuffle=False)
    
    return train_loader, dev_loader


def train_model(classifier, criterion, accuracy_fn, classifier_opt, opt_scheduler,
                train_loader, dev_loader, validation=False):
    from time import time

    start = time()
    epochs = 5
    epoch_log = []
    loss_log = []
    accuracy_log = []

    if validation:
        classifier.eval()
        with torch.no_grad():
            dev_loss, dev_acc = 0, 0
            for X_dev, y_dev in dev_loader:
                X_dev, y_dev = X_dev.to(device), y_dev.to(device)
                y_pred_dev = classifier(X_dev)
                loss_d = criterion(y_pred_dev, y_dev)
                accuracy_d = accuracy_fn(y_dev, y_pred_dev.argmax(dim=1))

                dev_loss += loss_d
                dev_acc += accuracy_d

            dev_loss /= len(dev_loader)
            dev_acc /= len(dev_loader)
            print(f'Dev loss: {dev_loss:.4f} -- Dev accuracy: {dev_acc:.4f}\n')

    else:
        for epoch in range(epochs):
            print(f"\nEpoch: {epoch + 1}\n--------")
            train_loss = dev_loss = train_acc = dev_acc = 0

            for batch, (X, y) in enumerate(tqdm(train_loader)):
                X, y = X.to(device), y.to(device)

                y_pred = classifier(X)
                loss = criterion(y_pred, y)
                accuracy = accuracy_fn(y, y_pred.argmax(dim=1))

                train_loss += loss
                train_acc += accuracy

                classifier_opt.zero_grad()
                loss.backward()
                classifier_opt.step()

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            epoch_log.append(epoch)
            accuracy_log.append(train_acc)
            loss_log.append(train_loss)
            print(f'Train loss: {train_loss:.4f} -- Train accuracy: {train_acc:.4f}')

            classifier.eval()
            # with torch.inference_mode():
            with torch.no_grad():
                for X_dev, y_dev in dev_loader:
                    X_dev, y_dev = X_dev.to(device), y_dev.to(device)
                    y_pred_dev = classifier(X_dev)
                    loss_d = criterion(y_pred_dev, y_dev)
                    accuracy_d = accuracy_fn(y_dev, y_pred_dev.argmax(dim=1))

                    dev_loss += loss_d
                    dev_acc += accuracy_d

            dev_loss /= len(dev_loader)
            dev_acc /= len(dev_loader)
            opt_scheduler.step()
            print(f'Dev loss: {dev_loss:.4f} -- Dev accuracy: {dev_acc:.4f}\n')
            print(f'Time: {time() - start:.3f} seconds')

        PATH = "/content/"
        torch.save(classifier.state_dict(), f"{PATH}MNIST_Classifier")
        print(f"model has been saved at {PATH}MNIST_Classifier")


def run(classifier, criterion, accuracy_fn, classifier_opt, opt_scheduler,
        train_loader, dev_loader, modelpath=''):

    # Check if a pre-trained classifier model exists
    if os.path.isfile(modelpath):
        classifier.load_state_dict(torch.load(
            modelpath, map_location=device, weights_only=True
        ))
        print("\n>>> Check Classifier Model accuracy:")

        # Evaluate the accuracy of the pre-trained model on the validation set
        train_model(classifier=classifier, criterion=criterion, accuracy_fn=accuracy_fn,
                    classifier_opt=classifier_opt, opt_scheduler=opt_scheduler,
                    train_loader=train_loader, dev_loader=dev_loader, validation=True)
    else:
        # Train the classifier if no pre-trained model is found
        train_model(classifier=classifier, criterion=criterion, accuracy_fn=accuracy_fn,
                    classifier_opt=classifier_opt, opt_scheduler=opt_scheduler,
                    train_loader=train_loader, dev_loader=dev_loader)
