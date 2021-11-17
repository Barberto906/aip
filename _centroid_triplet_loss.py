import torch
from utils import euclidean_dist, retrieve_centroids, get_device


def retrieve_positive_centroids(labels, batch):
    """Compute the positive centroid of each embeddings.
        For each embedding take the others embeddings belonging to the same class and compute
        the centroid. If a class has only 1 embedding in the batch a centroid made of 0 is returned at
        the relative position

        Args:
            labels: tensor of shape (batch_size,1)
            batch: tensor of shape (batch_size, embed_dim) tensor.float32


        Returns:
            pos_centroids : tensor of shape (batch_size, embed_dim)

        """
    device = get_device(batch)
    mask_anch_pos = torch.eq(labels, labels.transpose(1, 0))
    id_matrix = torch.eye(n=labels.shape[0]).to(device)
    mask_anch_pos = torch.logical_and(mask_anch_pos, torch.logical_not(id_matrix))  # setting the diagonal to false
    mask_anch_pos = mask_anch_pos.to(torch.float32)  # now they are 1 and 0
    numel_per_class = torch.sum(mask_anch_pos, dim=0)

    mask_anch_pos.unsqueeze_(dim=2)  # shape [batch_size, batch_size, 1]
    # batch has shape [batch_size, embedding_dim  -> broadcastable with mask_anch_pos
    aus = torch.sum(mask_anch_pos*batch, dim=1)  # [batch_size, embedding_dim]
    numel_per_class.unsqueeze_(dim=1)  # [batch_size,1]
    pos_centroids = aus / (numel_per_class + 1e-12)  # avoid division by zero
    return pos_centroids


def centroid_triplet_loss(labels, batch, margin=0.4):  # label shape [batch_size], input shape [batch_size, embed_dim]
    device = get_device(batch)
    centroids, centroids_class = retrieve_centroids(labels, batch)
    labels = labels.unsqueeze(dim=1)
    mask_neg_centroid = torch.logical_not(torch.eq(labels, centroids_class))  # shape [batch_size, num_classes]
    mask_valid_neg_centroid = torch.any(mask_neg_centroid, dim=1)   # [batch_size] if an element is False means that a class has no negatives (the batch contains only a class)
    anchor_neg_distance = euclidean_dist(batch, centroids)  # shape [batch_size, num_classes]

    pos_centorids = retrieve_positive_centroids(labels, batch)  # [batch_size, embed_dim]
    # since if an embeddings has no other embedding of the same class it has no positive centroid
    # and at his position there is a zero vector
    zeros = torch.zeros(pos_centorids.shape).to(device)
    mask_valid_pos_centroid = torch.eq(pos_centorids, zeros)
    mask_valid_pos_centroid = torch.all(mask_valid_pos_centroid, dim=1)
    mask_valid_pos_centroid = torch.logical_not(mask_valid_pos_centroid)
    mask_valid_triplet = torch.logical_and(mask_valid_pos_centroid, mask_valid_neg_centroid)
    anch_poscentroid_distance = torch.diagonal(euclidean_dist(batch, pos_centorids))

    max = torch.max(anchor_neg_distance)+1
    anchor_neg_distance = torch.masked_fill(anchor_neg_distance, torch.logical_not(mask_neg_centroid), max)
    hardest_neg_centroid, _ = torch.min(anchor_neg_distance, dim=1)

    anch_poscentroid_distance = anch_poscentroid_distance[mask_valid_triplet]
    hardest_neg_centroid = hardest_neg_centroid[mask_valid_triplet]

    aus = (anch_poscentroid_distance + margin) - hardest_neg_centroid
    zeros = torch.zeros(aus.shape[0]).to(device)
    loss = torch.maximum(zeros, aus).mean()
    return loss