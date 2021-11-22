import torch
import os
from model import Net
from loss import loss_function
from eval_utils import cosine_similarity, accuracy
# from time import perf_counter


class Classifier:
    def __init__(self, resnet_type, output_size, mlp_on_top, device):
        self.net = Net(resnet_type, output_size, mlp_on_top)
        self.device = device
        self.net.to(self.device)

    # save training checkpoint
    def save_checkpoint(self, model, optimizer, epoch, best_scores, loss):
        print("Saving model and training info!")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best scores': best_scores,
            'loss': loss,

        }, os.path.dirname(os.path.abspath(__file__)) + "/train_checkpoint.pth")
    ####################################################################################################################

    # load training checkpoint
    def load_checkpoint(self, path, learning_rate):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_scores = checkpoint['best scores']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Checkpoint loaded!")

        return optimizer, epoch, loss, best_scores
    ####################################################################################################################

    # train the model
    def train_net(self, num_epochs, learning_rate, train_set, query_set, test_set, checkpoint_path=None):
        self.net.train()
        loss = None

        # check if there is a checkpoint to continue training otherwise start from scratch
        if checkpoint_path is not None:
            optimizer, starting_epoch, loss, best_scores = self.load_checkpoint(checkpoint_path, learning_rate)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=learning_rate)
            best_scores = [0., 0., 0.]
            starting_epoch = 0

        for epoch in range(starting_epoch, num_epochs):
            print("Epoch:", epoch)
            epoch_loss = 0
            for index, (imgs, labels, _) in enumerate(train_set):
                # print("train_set batch index:", index)
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                output = self.net(imgs)

                # loss_start = perf_counter()
                loss = loss_function(labels, output)
                # loss_stop = perf_counter()
                # print("loss computation time:", loss_stop - loss_start)

                # if not torch.isnan(loss).item():
                epoch_loss += loss.item()

                # loss_backward_start = perf_counter()
                if loss.item() != 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # loss_backward_stop = perf_counter()
                # print("loss_backward, optimizer.step() and optimizer.zero_grad() time:",
                # loss_backward_stop - loss_backward_start)

            print("Epoch {} average loss value {}".format(epoch, epoch_loss / len(train_set)))

            scores = self.validation(query_set, test_set)

            if scores[0] >= best_scores[0] and scores[1] >= best_scores[1] and scores[2] >= best_scores[2]:
                print("Best model found, saving!")
                torch.save(self.net.state_dict(),
                           os.path.dirname(os.path.abspath(__file__)) + '/best_classifier.pth')
                best_scores[0] = scores[0]
                best_scores[1] = scores[1]
                best_scores[2] = scores[2]

            """"
            # validation step every 3 epochs
            if epoch % 3 == 2:
                scores = self.validation(query_set, test_set)

                if scores[0] >= best_scores[0] and scores[1] >= best_scores[1] and scores[2] >= best_scores[2]:
                    print("Best model found, saving!")
                    torch.save(self.net.state_dict(),
                               os.path.dirname(os.path.abspath(__file__)) + '/best_classifier.pth')
                    best_scores[0] = scores[0]
                    best_scores[1] = scores[1]
                    best_scores[2] = scores[2]
            """
            # saving after each epoch
            self.save_checkpoint(self.net, optimizer, epoch, best_scores, loss)
    ####################################################################################################################

    def evaluate(self, batch):
        self.net.eval()
        batch = batch.to(self.device)
        with torch.no_grad():
            output = self.net(batch)

        return output
    ####################################################################################################################

    def validation(self, query_set, test_set):
        test_embeddings = list()
        test_labels = list()

        # Pre-computing embeddings for test set
        for index, (imgs, labels, _) in enumerate(test_set):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            test_embeddings.append(self.evaluate(imgs))
            test_labels += labels

        test_embeddings = torch.vstack(test_embeddings)

        distances = list()
        query_labels = list()

        # print("Test set embeddings computed")

        # Computing query set embeddings
        for index, (imgs, labels, _) in enumerate(query_set):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            query_embeddings = self.evaluate(imgs)
            query_labels += labels
            batch_distances = cosine_similarity(query_embeddings, test_embeddings)
            distances += batch_distances

        # print("query set and distances computed")

        distances = torch.vstack(distances)
        indices = torch.argsort(distances, dim=1, descending=True)  # [num_query_imgs, num_test_imgs]

        test_labels = torch.tensor(test_labels, device=self.device)
        test_labels = test_labels.expand(indices.shape[0], -1)
        test_labels = torch.gather(test_labels, dim=1, index=indices)

        acc1 = accuracy(query_labels, test_labels[:, 0:1])
        acc3 = accuracy(query_labels, test_labels[:, 0:3])
        acc5 = accuracy(query_labels, test_labels[:, 0:5])

        print("Accuracy @ 1:", acc1)
        print("Accuracy @ 3:", acc3)
        print("Accuracy @ 5:", acc5)

        return [acc1, acc3, acc5]
