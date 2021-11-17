"""
We implement our centroid-based solution on top of the current
fashion retrieval state-of-the-art model [13], which itself is based
on a top-scoring ReID model [7]. We train our model on various
Resnet-based backbones pretrained on ImageNet, and report results
for Fashion Retrieval and Person Re-Identification tasks. We evaluate
the model both in centroid-based and instance-based setting.
Instance-based setting means that pairs of images are evaluated,
identically as in the evaluation setting of [13]. We use the same
training protocol presented in the aforementioned papers (e.g. random erasing augmentation, label smoothing),
without introducing any additional steps.


Most existing instance retrieval solutions use Deep Metric Learn-
ing methodology [1, 3, 6, 7, 13, 16], in which a deep learning model
is trained to transform images to a vector representation, so that
samples from the same class are close to each other. At the retrieval
stage, the query embedding is scored against all gallery embed-
dings and the most similar ones are returned. Until recently, a
lot of works used classification loss for the training of retrieval
models [8, 14, 15, 17, 20]. Currently most works use compara-
tive/ranking losses and the Triplet Loss is one of the most widely
used approaches. However, state-of-the-art solutions often combine
a comparative loss with auxiliary losses such as classification or
center loss [5, 7, 12, 13, 16]

"""
import torch


def cosine_similarity(query_embeddings, test_embeddings):
    a = query_embeddings.unsqueeze(1).expand(-1, test_embeddings.shape[0], -1)
    b = test_embeddings.unsqueeze(0).expand(query_embeddings.shape[0], -1, -1)

    numerator = a * b
    numerator = torch.sum(numerator, dim=2)

    a_2 = a ** 2
    a_2 = torch.sum(a_2, dim=2)
    b_2 = b ** 2
    b_2 = torch.sum(b_2, dim=2)

    denominator = torch.sqrt(a_2) * torch.sqrt(b_2)
    result = numerator / denominator
    # print(result.shape)
    # print(result)
    return result


def accuracy(query_labels, test_labels):  # [n_query, label], [n_test, label], [n_test, index]
    acc = 0.

    for idx in range(0, len(query_labels)):
        for i in range(test_labels.shape[1]):
            aus = 0
            if test_labels[idx, i] == query_labels[idx]:
                aus += 1.
            acc += aus / test_labels.shape[1]

    acc = acc / len(query_labels)
    return acc


"""
query_embd = torch.tensor([[1, 0, 1], [-4, 5, -9]])

test_embd = torch.tensor([[1, 2, 3], [-4, 5, 6], [7, 8, -9]])

cosine_similarity(query_embd, test_embd)
"""

