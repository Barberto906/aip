import torch
from utils import euclidean_dist, retrieve_centroids, get_device


def center_loss(labels, batch):
    device = get_device(batch)
    centroids, centroids_class = retrieve_centroids(labels, batch)
    # centroids shape [num_classes_in_the_batch, emb_dim]
    # centroids_class shape [1, num_classes_in_the_batch]
    distances = euclidean_dist(batch, centroids)  # shape [batch_size, num_classes_in_the_batch]
    labels = labels.unsqueeze(dim=1)
    mask_correct_dist = torch.eq(labels, centroids_class)
    distances = torch.masked_select(distances, mask_correct_dist)

    zeros = torch.zeros(distances.shape).to(device)
    loss = torch.maximum(distances, zeros).sum()
    return loss


