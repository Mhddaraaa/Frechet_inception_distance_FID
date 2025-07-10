from models.frechet_distance import *
from models.Generator_Critic import *
from models.Classifier import *
from models.feature_extraction import *
from models.config import *

import torch
from torch import nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--classifier-path", help="Pretrained classifier model")
parser.add_argument("-g", "--generator-path", help="Pretrained generator-critic model")
args = parser.parse_args()
print(args.generator_path)


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

# classifier_path = 'pretrain_models/MNIST_Classifier_GAN_FID'
if os.path.isfile(args.classifier_path):
    print("Loading pretrianed classifier model...")
    classifier_path = args.classifier_path
else:
    print("Starting to train the classifier on MNIST dataset...")
    classifier_path = ''

run(classifier=classifier, criterion=criterion, accuracy_fn=accuracy_fn,
    classifier_opt=classifier_opt, opt_scheduler=classifier_opt_exp_lr_scheduler,
    train_loader=train_loader, dev_loader=dev_loader, modelpath=classifier_path)

#Prepare Generator
data, data_classes = get_data(BS)
C, H, W = next(iter(data))[0][0].shape

# generator_path = 'pretrain_models/cGANs_Gen_mnist'
if os.path.isfile(args.generator_path):
    print("Loading pretrained generator-critic model...")
    generator_path = args.generator_path
else:
    print("Training the generator-critic model from scratch...")
    generator_path = ''

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

main(gen, classifier_path, train_loader)