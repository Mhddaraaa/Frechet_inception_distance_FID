import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models.config import *


#Generator
class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, hidden_ch=8, out_ch=C,
                 num_class=NUM_CLASS, img_size=H, embed_size=EMBED_SIZE):
        super().__init__()
        
        # Initialize generator parameters
        self.img_size = img_size
        self.z_dim = z_dim
        
        # Define the generator network architecture using nn.Sequential
        self.gen = nn.Sequential(
            self._gen_block(z_dim + num_class, hidden_ch*8, 4, 1, 0),
            self._gen_block(hidden_ch*8, hidden_ch*4, 4 , 2 , 1),
            self._gen_block(hidden_ch*4, hidden_ch*2, 4, 2, 1),
            nn.ConvTranspose2d(hidden_ch*2, out_ch, 4, 2, 1),
            nn.Tanh()
        )
        
        # Use embedding instead of one-hot encoding for class labels
        self.embed = nn.Embedding(num_class, num_class)

    def forward(self, noise, labels):
        # Reshape input noise tensor
        noise = noise.view(len(noise), self.z_dim, 1, 1)
        
        # Generate class label embeddings and concatenate with noise
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        noise = torch.cat([noise, embedding], dim=1)
        
        # Pass the concatenated input through the generator network
        return self.gen(noise)

    def _gen_block(self, in_ch, out_ch, kernel, stride, padding):
        # Define a generator block with transposed convolution, instance normalization, and ReLU activation
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel, stride, padding, bias=False
            ),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


#Discriminator
class Critic(nn.Module):
    def __init__(self, img_ch=1, hidden_ch=8, out_dim=1,
                 num_class=NUM_CLASS, img_size=H):
        super().__init__()
        
        # Initialize critic parameters
        self.img_size = img_size
        self.num_class = num_class
        
        # Define the critic network architecture using nn.Sequential
        self.disc = nn.Sequential(
            nn.Conv2d(img_ch + self.num_class, hidden_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self._disc_block(hidden_ch, hidden_ch*2, 4, 2, 1),
            self._disc_block(hidden_ch*2, hidden_ch*4, 4, 2, 1),
            self._disc_block(hidden_ch*4, hidden_ch*8, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(hidden_ch * 8 * 2 * 2, out_dim),
            nn.Sigmoid()
        )
        
        # Use embedding for class labels
        self.embed = nn.Embedding(self.num_class, img_size*img_size*self.num_class)

    def forward(self, x, labels):
        # Generate class label embeddings and concatenate with input images
        embedding = self.embed(labels).view(len(labels), self.num_class, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        
        # Pass the concatenated input through the critic network
        return self.disc(x).view(-1, 1)

    def _disc_block(self,in_ch, out_ch, kernel, stride, padding):
        # Define a critic block with convolution, batch normalization, and LeakyReLU activation
        return nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )


# Function to initialize weights of neural network modules
def weights_init(m):
    # Get the class name of the module
    classname = m.__class__.__name__
    
    # Initialize weights for Convolutional layers
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    # Initialize weights and biases for Batch Normalization layers
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Lambda function to generate random noise for the generator
gen_noise = lambda number, z_dim: torch.randn(number, z_dim).to(device)


def get_data(bs=128):
    """
    This function retrieves training data for Generative Adversarial Networks (GANs) from the MNIST dataset.
    
    Parameters:
    - bs (int): Batch size for grouping the data (default is 128).

    Returns:
    - data_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
    - data_classes (list): List of classes present in the training dataset.

    The function applies a series of transformations to the MNIST dataset, including resizing to a specified height (H),
    conversion to a PyTorch tensor, and normalization. It then creates a DataLoader for the training set with the specified batch size.
    The classes present in the dataset are also returned for reference.
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(H),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.5 for _ in range(C)], [0.5 for _ in range(C)]
        )
    ])
    
    # Download and transform the MNIST training dataset
    train_set = MNIST('.', train=True, transform=transform, download=True)

    # Group the data into batches using DataLoader
    data_classes = train_set.classes
    data_loader = DataLoader(train_set, bs, shuffle=True)

    return data_loader, data_classes

data, data_classes = get_data(BS)
C, H, W = next(iter(data))[0][0].shape



#This the the most important part of the training
def gen_MSEloss_func(gen_net, critic_net, batch_size, z_dim, labels, mode=1):
    """
    This function calculates the Mean Squared Error (MSE) loss for the generator in the context of a Generative Adversarial Network (GAN).

    Parameters:
    - gen_net (torch.nn.Module): Generator neural network.
    - critic_net (torch.nn.Module): Critic neural network.
    - batch_size (int): Number of samples in a batch.
    - z_dim (int): Dimensionality of the random noise vector.
    - labels (torch.Tensor): Class labels for conditioning the generator.
    - mode (int): Mode of operation:
        - mode=1: Generating samples as real as possible.
        - mode=2: Minimizing Pearson χ² divergence.

    Returns:
    - loss (torch.Tensor): Calculated generator loss.

    The function generates fake samples using the generator with random noise and specified class labels.
    It then evaluates the critic's predictions on the generated samples and calculates the MSE loss with a target label (c).
    The mode parameter determines the objective of the generator loss, either generating samples as real as possible (mode=1)
    or minimizing Pearson χ² divergence (mode=2).
    """
    # Generate random noise for the generator
    noise = gen_noise(batch_size, z_dim)

    # Generate fake samples using the generator with specified class labels
    fake = gen_net(noise, labels)

    # Get critic's predictions on the generated samples
    pred = critic_net(fake, labels)

    # Set target label (c) based on the mode
    c = 1 if mode == 1 else 0

    # Calculate Mean Squared Error (MSE) loss
    loss = 0.5 * torch.sum((pred - c) ** 2)

    return loss

def critic_MSEloss_func(gen_net, critic_net, image, batch_size, z_dim, labels, mode=1):
    """
    This function calculates the Mean Squared Error (MSE) loss for the critic in the context of a Generative Adversarial Network (GAN).

    Parameters:
    - gen_net (torch.nn.Module): Generator neural network.
    - critic_net (torch.nn.Module): Critic neural network.
    - image (torch.Tensor): Real images from the dataset.
    - batch_size (int): Number of samples in a batch.
    - z_dim (int): Dimensionality of the random noise vector.
    - labels (torch.Tensor): Class labels for conditioning the generator.
    - mode (int): Mode of operation:
        - mode=1: Critic evaluates real samples as real and fake samples as fake.
        - mode=2: Critic evaluates real samples as real and fake samples as not real.

    Returns:
    - loss (torch.Tensor): Calculated critic loss.

    The function generates fake samples using the generator with random noise and specified class labels.
    It evaluates the critic's predictions on both real and generated samples and calculates the MSE loss with target labels.
    The mode parameter determines the objective of the critic loss, either evaluating real samples as real and fake as fake (mode=1)
    or evaluating real samples as real and fake samples as not real (mode=2).
    """
    # Generate random noise for the generator
    noise = gen_noise(batch_size, z_dim)

    # Generate fake samples using the generator with specified class labels
    fake = gen_net(noise, labels)

    # Evaluate the critic's predictions on real and generated samples
    fake_pred = critic_net(fake.detach(), labels)  # detach() the generator output so it won't participate in gen_net learning
    real_pred = critic_net(image, labels)

    # Set target labels (b, a) based on the mode
    b, a = (1, 0) if mode == 1 else (1, -1)

    # Calculate Mean Squared Error (MSE) loss for the critic
    loss = 0.5 * torch.sum(((real_pred - b) ** 2 )+ ((fake_pred - a) ** 2))

    return loss

if os.path.isdir('runs'):
    pass
else:
    os.mkdir('runs')
writer = SummaryWriter("runs")
writer_fake = SummaryWriter("runs/fake")
writer_real = SummaryWriter("runs/real")

# Training loop for a Generative Adversarial Network (GAN)
def generator_critic_train(generator, critic, critic_opt, critic_loss_func, gen_opt, gen_loss_func,
                           gen_lr_scheduler, critic_lr_scheduler, epoch=20):

    # Initialize step for Tensorboard visualization
    step = 0

    # Iterate through epochs
    for epoch in range(epoch):
        criticLoss, genLoss = 0, 0
        print(f"\nEpoch: {epoch + 1}")

        # Iterate through batches in the dataset
        for batch, (real, labels) in enumerate(tqdm(data)):
            real = real.to(device)
            labels = labels.to(device)

            # Update the critic network
            critic_opt.zero_grad()
            critic_loss = critic_loss_func(generator, critic, real, len(real), Z_DIM, labels)
            critic_loss.backward(retain_graph=True)
            critic_opt.step()

            # Update the generator network
            gen_opt.zero_grad()
            gen_loss = gen_loss_func(generator, critic, len(real), Z_DIM, labels)
            gen_loss.backward()
            gen_opt.step()

            # Accumulate losses for logging
            criticLoss += critic_loss / len(data)
            genLoss += gen_loss / len(data)

            # Tensorboard visualization
            if batch % 150 == 0 and batch != 0:
                with torch.no_grad():
                    step += 1
                    fake = gen(gen_noise(BS, Z_DIM), labels).view(-1, C, H, W)
                    image = real.view(-1, C, H, W)
                    real_grid = make_grid(image[:32], normalize=True)
                    fake_grid = make_grid(fake[:32], normalize=True)

                    # Add generated and real images to Tensorboard
                    writer_fake.add_image(
                        "MNIST fake image", fake_grid, global_step=step
                    )
                    writer_real.add_image(
                        "MNIST real image", real_grid, global_step=step
                    )

            # Tensorboard logging of loss values
            writer.add_scalars("Loss", {
                        "Critic": critic_loss.item(),
                        "Generator": gen_loss.item()
                    }, (epoch + 1) * batch)

        # Print accumulated losses for the epoch
        print(f'  Critic Loss: {criticLoss:.4f} -- Generator Loss: {genLoss:.4f}')

        # Step the learning rate schedulers
        gen_lr_scheduler.step()
        critic_lr_scheduler.step()

        # Save model parameters
        PATH = "/content/model/"
        if os.path.isdir(PATH):
            pass
        else:
            os.mkdir(PATH)

        torch.save(gen.state_dict(), f"{PATH}Gen_{epoch+1}")
        torch.save(critic.state_dict(), f"{PATH}Critic_{epoch+1}")

        # Print learning rates every 2 epochs
        if (epoch + 1) % 2 == 0 and epoch > 0:
            print(f"  >>> Critic Learning Rate: {critic_opt.param_groups[0]['lr']}")
            print(f"  >>> Generator Learning Rate: {gen_opt.param_groups[0]['lr']}")


# check if generator already exist or need to train model
def run_gen_critic(generator, critic, critic_opt, critic_loss_func, gen_opt, gen_loss_func,
                   gen_lr_scheduler, critic_lr_scheduler, generator_path=''):

    # Check if a pre-trained generator model exists
    if os.path.isfile(generator_path):
        print('Check generator generated images')

        # Load the pre-trained generator model
        generator.load_state_dict(torch.load(generator_path, map_location=device))

        # Create a figure for displaying generated images
        fig = plt.figure(figsize=(8, 4))

        # Generate and display images for each class
        for i in range(10):
            num = torch.tensor([i], device=device)
            generated = generator(gen_noise(1, Z_DIM), num).view(-1, C, H, W)
            fig.add_subplot(2, 5, i+1)
            plt.imshow(generated.detach().cpu().squeeze(0).permute(1, 2, 0), cmap='gray_r')
            plt.title(data_classes[i])
            plt.axis(False)
    else:
        # Train the generator and critic if no pre-trained model is found
        generator_critic_train(generator, critic, critic_opt, critic_loss_func, gen_opt, gen_loss_func,
                               gen_lr_scheduler, critic_lr_scheduler)
