from _center_loss import center_loss
from _centroid_triplet_loss import centroid_triplet_loss
from _triplet_loss import triplet_loss


# paper -> 3.2 Implementation details (for hyperparameters)
def loss_function(labels, out, alpha=1., beta=1., gamma=5**(-4), trip_loss_margin=0.8, centr_trip_loss_margin=0.8):
    trip_loss = triplet_loss(labels, out, hard_samples=True, margin=trip_loss_margin)
    centr_trip_loss = centroid_triplet_loss(labels, out, margin=centr_trip_loss_margin)
    centr_loss = center_loss(labels, out)
    return alpha*trip_loss + beta*centr_trip_loss + gamma*centr_loss
