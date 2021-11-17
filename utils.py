import torch
from torch import nn


def squared_euclidian_distance(input1, input2):
    """
    Input1: (N, D)(N,D) where D = vector dimension
    Input2: (N, D)(N,D), same shape as the Input1
    Output: (N)(N)
    """
    pdist = nn.PairwiseDistance(p=2)
    distance = torch.square(pdist(input1, input2))
    # distance = pdist(input1, input2)
    distance = distance.clamp(min=1e-12).sqrt()  # for numerical stability
    return distance


def pairwise_distances(input1, input2, squared=True):
    """
    Compute the 2D matrix of squared distances between all the embeddings.

    Args:
        input1: tensor of shape (batch_size, embed_dim) tensor.float32
        input2: tensor of shape (batch_size, embed_dim) tensor.float32
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
        at index [i,j] there is the distance between vector i of input1 and vector j of input2
    """
    batch_size = input1.shape[0]
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(input1, torch.transpose(input2, 0, 1))
    # Get squared L2 norm for each embedding.
    norm_input1 = torch.linalg.norm(input1, dim=1).unsqueeze(dim=1)  # shape (batch_size,1)
    norm_input2 = torch.linalg.norm(input2, dim=1).unsqueeze(dim=0)  # shape (1,batch_size)
    square_norm_input1 = torch.square(norm_input1)
    square_norm_input2 = torch.square(norm_input2)
    # the unsqueeze is necessary because expand need the dimension to expand to be 1
    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    square_pairwise_dist = square_norm_input1.expand(-1, batch_size) - 2.0*dot_product + square_norm_input2.expand(batch_size, -1)
    # zeros = torch.zeros(size=square_pairwise_dist.shape)
    # square_pairwise_dist = torch.maximum(square_pairwise_dist, zeros)
    square_pairwise_dist.clamp_(min=1e-12)
    # Because of computation errors, some distances might be negative so we put everything > 0.0
    return square_pairwise_dist


def euclidean_dist(x, y, squared=True):
    """
    Args:
      x: pytorch tensor, with shape [m, d]
      y: pytorch tensor, with shape [n, d]
      squared: if True returns the squared distance
    Returns:
      dist: pytorch tensor, with shape [m, n] at index ij there is x[i] - y[j]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    if squared:
        dist = dist.clamp(min=1e-12)
    else:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def retrieve_centroids(labels, batch):  # labels [batch_size] batch [batch_size, embedding_dim]
    """Compute the n centroids in batch, where n=num_classes

    Args:
        labels: torch.tensor (batch_size,)
        batch: torch.tensor (batch_size, embedding_dim) -> batch[i] is an embedding belonging to labels[i] class
    Returns:
        centroids_class torch.tensor (num_classes, )
        centroids torch.tensor (num_classes, embedding_dim) centroid[i] is the centroid of the class centroid_class[i]
    """
    batch_size = batch.shape[0]
    centroids_class = torch.unique(labels, sorted=True).unsqueeze_(dim=0)  # shape [1, num_classes]
    exp_centroid_class = centroids_class.expand(batch_size, -1)   # shape [batch_size, num_classes]
    lab = labels.unsqueeze(dim=1) # shape [batch_size,1] -> labels and exp_centroid_class are broadcastable to eachother

    mask = torch.eq(exp_centroid_class, lab).to(torch.float32)  # shape [batch_size, num_classes]
    mask = mask.permute(1, 0)  # shape [num_classes, batch_size]

    num_elem_class = torch.sum(mask, dim=1, keepdim=True)  # shape [num_classes, 1]
    centroids = torch.mm(mask, batch) / num_elem_class  # shape [num_classes, embedding_dim]

    return centroids, centroids_class


def get_device(input):
    """Returns the device in witch the input tensor is stored
    Args:
        input: torch.tensor
    Returns:
         device: torch.device
    """
    gpu_check = input.is_cuda
    if gpu_check:
        dev = 'cuda:{}'.format(input.get_device())
    else:
        dev = 'cpu'
    return torch.device(dev)
