import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from frechet_distance import *
from Generator_Critic import *
from Classifier import *
from feature_extraction import *

import scipy

import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision import transforms

# Hyperparameters
EPOCH = 20
Z_DIM = 100
LR = 2e-4
BS = 64
C, H, W = 1, 32, 32
NUM_CLASS = 10 # 0, 1, 2, ..., 9
EMBED_SIZE = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Availabe device is: ", device)

# Prepare Classifier
train_loader, dev_loader = classifier_dataset()

classifier = Classifier().to(device)
classifier.apply(weights_init)
print(classifier)
summary(classifier, (1, 32, 32), batch_size=-1, device='cuda')

# Define optimizer for Classifier
classifier_opt = torch.optim.Adam(classifier.parameters(), lr=0.001)
classifier_opt_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(classifier_opt, step_size=1, gamma=0.5)

# Define loss function for Classifier
criterion = nn.CrossEntropyLoss()

classifier_path = './model/MNIST_Classifier_GAN_FID'
run(classifier=classifier, criterion=criterion, accuracy_fn=accuracy_fn,
    classifier_opt=classifier_opt, opt_scheduler=classifier_opt_exp_lr_scheduler,
    train_loader=train_loader, dev_loader=dev_loader, modelpath=classifier_path, validation=False)

#Prepare Generator
data, data_classes = get_data(BS)
C, H, W = next(iter(data))[0][0].shape

generator_path = 'model/cGANs_Gen_mnist'
# Define generator
gen = Generator().to(device)  # Instantiate the Generator and move it to the specified device (GPU or CPU)
gen.apply(weights_init)  # Initialize the weights of the generator using the weights_init function
gen_opt = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))  # Adam optimizer for the generator
gen_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt, step_size=5, gamma=0.8)  # Learning rate scheduler

# Define critic
critic = Critic().to(device)  # Instantiate the Critic and move it to the specified device
critic.apply(weights_init)  # Initialize the weights of the critic using the weights_init function
critic_opt = torch.optim.Adam(critic.parameters(), lr=2.5e-4, betas=(0.5, 0.999))  # Adam optimizer for the critic
critic_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(critic_opt, step_size=5, gamma=1)  # Learning rate scheduler

run_gen_critic(generator=gen, critic=critic, critic_opt=critic_opt, critic_loss_func=critic_MSEloss_func,
               gen_opt=gen_opt, gen_loss_func=gen_MSEloss_func,
               gen_lr_scheduler=gen_exp_lr_scheduler, critic_lr_scheduler=critic_exp_lr_scheduler,
               generator_path=generator_path)

main(gen, classifier_path, generator_path, train_loader)




