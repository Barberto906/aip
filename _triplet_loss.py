import torch
from utils import pairwise_distances, get_device


def get_mask(labels, labels_equal=True):
    # labels_equals return a mask tensor for anchor positives pairs
    # else return a mask tensor for anchor negatives pairs
    device = get_device(labels)
    batch_size = labels.shape[0]
    right_labels = labels.unsqueeze(dim=0)
    left_labels = labels.unsqueeze(dim=1)
    mask = torch.eq(left_labels.expand(-1, batch_size), right_labels.expand(batch_size, -1))

    if labels_equal:
        id_matrix = torch.eye(n=mask.shape[0], dtype=torch.bool).to(device)
        mask.masked_fill_(id_matrix, False)
        # torch.eye Returns a 2-D tensor with ones on the diagonal and zeros elsewhere. ( n= num of rows, m (optional) num of cols)
        # masked_fill_ Fills elements of self tensor with value where mask is True.
        return mask
    else:
        return torch.logical_not(mask)


def hard_sample_mining(anchor_positive_dist, anchor_positive_mask, anchor_negative_dist, anchor_negative_mask):
    # distances and mask have shape [batch_size, batch_size]

    # for each anchor the hard positive is the one that maximize the distance
    min = torch.min(anchor_positive_dist).item() - 1
    anchor_positive_dist.masked_fill_(torch.logical_not(anchor_positive_mask), min)

    # since later only the max value of each row will be selected we removed the possibility of choosing ...
    hard_ap, _ = torch.max(anchor_positive_dist, dim=1)
    max = torch.max(anchor_negative_dist).item() + 1
    anchor_negative_dist.masked_fill_(torch.logical_not(anchor_negative_mask), max)

    hard_an, _ = torch.min(anchor_negative_dist, dim=1)
    # Could happen that an embedding belongs to a class with only one example in the batch. So it doesn't have a positive
    mask_hard_ap = hard_ap != min
    # same reason
    mask_hard_an = hard_an != max
    final_mask = torch.logical_and(mask_hard_ap, mask_hard_an)

    return hard_ap[final_mask], hard_an[final_mask]


def triplet_loss(labels, batch, hard_samples=True, margin=1.0):
    device = get_device(batch)
    anchor_positive_dist = pairwise_distances(batch, batch)  # [batch_size, batch_size]
    anchor_negative_dist = pairwise_distances(batch, batch)  # [batch_size, batch_size]

    anchor_positive_mask = get_mask(labels, labels_equal=True)  # [batch_size, batch_size]
    anchor_negative_mask = get_mask(labels, labels_equal=False)  # [batch_size, batch_size]

    if hard_samples:
        hard_anchor_positive, hard_anchor_negative = hard_sample_mining(anchor_positive_dist, anchor_positive_mask,
                                                                        anchor_negative_dist, anchor_negative_mask)
        # if in the worst case hard_anchor_positive and hard_anchor_negative are empty tensors the
        # resulting loss will be 0 at the end
        aus = (hard_anchor_positive + margin) - hard_anchor_negative

    else:
        list_ap = []
        list_an = []
        for i in range(anchor_positive_dist.shape[0]):

            num_positives = torch.sum(anchor_positive_mask[i])
            num_negatives = torch.sum(anchor_negative_mask[i])

            ap = torch.masked_select(anchor_positive_dist[i], anchor_positive_mask[i]).unsqueeze_(dim=1)
            an = torch.masked_select(anchor_negative_dist[i], anchor_negative_mask[i]).unsqueeze_(dim=1)

            if ap.nelement() != 0 and an.nelement() != 0:
                ap = ap.expand((-1, num_negatives))
                ap = ap.reshape(num_positives * num_negatives)

                an = an.expand((-1, num_positives))
                an = an.reshape(num_negatives * num_positives)
                list_ap.append(ap)
                list_an.append(an)
        if len(list_ap) > 0:
            anch_pos = torch.hstack(list_ap)
            anch_neg = torch.hstack(list_an)

            aus = (anch_pos + margin) - anch_neg

    zeros = torch.zeros(aus.shape).to(device)
    loss = torch.mean(torch.maximum(aus, zeros))

    # loss = torch.maximum(aus, zeros)

    return loss


def pairwise_dist_loss(labels, batch, margin=1.0):
    anchor_positive_dist = pairwise_distances(batch, batch)  # [batch_size, batch_size]
    anchor_negative_dist = pairwise_distances(batch, batch)  # [batch_size, batch_size]
    anchor_positive_mask = get_mask(labels, labels_equal=True)  # [batch_size, batch_size]
    anchor_negative_mask = get_mask(labels, labels_equal=False)  # [batch_size, batch_size]
    anch_pos = torch.masked_select(anchor_positive_dist, anchor_positive_mask)
    anch_neg = torch.masked_select(anchor_negative_dist, anchor_negative_mask)
    zeros = torch.zeros(anch_neg.shape)
    anch_neg = torch.maximum(anch_neg + margin, zeros)
    loss = torch.mean(torch.hstack((anch_pos, anch_neg)))

    return loss