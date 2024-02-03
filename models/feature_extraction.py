from models.Classifier import *
from torch import nn
import matplotlib.pyplot as plt
from models.frechet_distance import *
from models.Generator_Critic import gen_noise
import numpy as np

Z_DIM = 100
LR = 2e-4
BS = 64
C, H, W = 1, 32, 32
NUM_CLASS = 10 # 0, 1, 2, ..., 9

class featureExtraction(nn.Module):

    def __init__(self, modelpath):
        super().__init__()
        self.net = Classifier().to(device)
        self.net.load_state_dict(torch.load(modelpath, map_location=device))
        self.net.pool3.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # N x 128 x 6 x 6
        self.output = output

    def forward(self, x): # Remember to normalize to [-1, 1]
        # Trigger output hook
        self.net(x)

        activations = self.output
        activations = F.adaptive_avg_pool2d(activations, (1,1)) # Output: N x 128 x 1 x 1
        return activations.view(x.shape[0], 128)
    

def main(generator, classifier_path, generator_path, train_loader):
    # Instantiate a feature extractor using a pre-trained classifier model
    feature_extractor = featureExtraction(classifier_path).to(device)
    
    # Load the pre-trained generator model
    # generator.load_state_dict(torch.load(generator_path))

    # Get a batch of real images from the training loader
    X, y = next(iter(train_loader))
    X, y = X.to(device), y.to(device)
    label = y.cpu().numpy()

    # Generate fake images using the GAN generator
    fake = generator(gen_noise(BS, Z_DIM), y).view(-1, C, H, W)

    # Visualize generated and real images along with Frechet Distance for each digit
    fig = plt.figure(figsize=(15, 5))
    fig.text(0.4, 0.9, 'Generated images', fontstyle='oblique', fontsize=12, color='blue')
    fig.text(0.4, 0.4, 'Real images', fontstyle='oblique', fontsize=12, color='red')

    fig.text(0.12, 0.9, 'Frechet Distance for each number: ', fontsize=12, color='darkorchid')

    for i in range(10):
        n = np.where(label == i)[0]
        num = np.random.choice(n)

        # Extract features from real and fake images
        real_img = feature_extractor(X[num, :, : ,:].unsqueeze(0)).view(-1, 1).cpu().detach()
        fake_img = feature_extractor(fake[num, :, : ,:].unsqueeze(0)).view(-1, 1).cpu().detach()

        # Calculate Frechet Distance between real and fake image features
        mu1, sigma1 = torch.mean(real_img, dim=0).view(-1, 1), torch_cov(real_img, rowvar=False)
        mu2, sigma2 = torch.mean(fake_img, dim=0).view(-1, 1), torch_cov(fake_img, rowvar=False)
        frechet_distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        # Visualize the generated and real images along with Frechet Distance
        fig.add_subplot(2, 10, i+1)
        plt.imshow(fake[num, :, : ,:].detach().cpu().permute(1, 2, 0), cmap='gray_r')
        plt.title(f"{frechet_distance:.6f}", color='darkorchid')
        plt.axis(False)

        fig.add_subplot(2, 10, i+11)
        plt.imshow(X[num, :, : ,:].detach().cpu().permute(1, 2, 0), cmap='gray_r')
        plt.axis(False)
    plt.show()
