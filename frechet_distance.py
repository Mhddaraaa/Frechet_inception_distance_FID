import torch
from scipy.linalg import sqrtm

def torch_cov(m, rowvar=False):
    """
    Estimate the covariance matrix of a matrix.

    Args:
        m (torch.Tensor): Input matrix which is output of the neural network
        rowvar (bool, optional): If True, then each row represents a variable, and each column a single observation.
                                 Default is False.

    Returns:
        torch.Tensor: Covariance matrix.
    """
    if not rowvar and m.shape[0] != 1:
        m = m.t()

    # Calculate mean along columns
    mean = torch.mean(m, dim=1, keepdim=True)

    # Center the data
    m_centered = m - mean

    # Calculate covariance matrix
    cov_matrix = torch.mm(m_centered, m_centered.t()) / (m.shape[1] - 1)

    return cov_matrix


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, logterm=False):
    """
    Calculate Fréchet Distance between two multivariate Gaussian distributions.

    Args:
        mu1 (torch.Tensor): Mean of the first distribution.
        sigma1 (torch.Tensor): Covariance matrix of the first distribution.
        mu2 (torch.Tensor): Mean of the second distribution.
        sigma2 (torch.Tensor): Covariance matrix of the second distribution.

    Returns:
        float: Fréchet Distance between the distributions.
    """

    # Compute the squared Mahalanobis distance between the means
    diff = mu1 - mu2
    diff_squared = torch.matmul(diff, diff)

    # Compute the product of square roots of covariance matrices
    cov_prod_sqrtm = sqrtm(sigma1 @ sigma2).real

    if not logterm:
        return diff_squared.item() + torch.trace(sigma1 + sigma2 - 2 * cov_prod_sqrtm.item())
    else:
        # Compute the logarithm of the product
        epsilon = 1e-6
        cov_prod = sigma1 @ sigma2 + epsilon * torch.eye(sigma1.shape[0])

        simga1_det = torch.det(sigma1 + epsilon * torch.eye(sigma1.shape[0]))
        simga2_det = torch.det(sigma2 + epsilon * torch.eye(sigma2.shape[0]))
        log_sqrtm = torch.sqrt(simga1_det * simga2_det)

        log_cov_prod = torch.log(torch.det(cov_prod) / log_sqrtm)

        # Return the Fréchet Distance
        return diff_squared.item() + torch.trace(sigma1 + sigma2 - 2 * cov_prod_sqrtm.item()) + log_cov_prod.item()
    