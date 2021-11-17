from data_augmentation import *
import cv2
import torch
import os
from PIL import Image
# from data_manager import DataManager
import scipy.io
from model import Net
import utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import CustomDataset
from classifier import Classifier
from sampler import CustomSampler


def print_img(title, img):
    # img = torch.permute(img, (1, 2, 0)).numpy()
    img = img.numpy()

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_img(img_path):
    img = Image.open(img_path)
    #  torch_img = torch.from_numpy(img).to(torch.float)
    #  img = torch.permute(torch_img, (2, 0, 1))
    print(img.size)
    print(type(img))
    return img


def plot_2d_points(points, labels):  # labels = list of labels. labels[i] = label of point[i]
    colours = ["#0000ff", "#ff0000", "#000000", "#FFFF00", "#00ff00", "#00ff00", "#00ff00", "#00ff00", "#00ff00",
               "#00ff00", "#00ff00"]
    labels = list(labels)
    for index in range(len(labels)):
        x = points[index][0].item()
        y = points[index][1].item()
        plt.scatter(x, y, c=colours[labels[index]], marker="o")
    plt.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    num_classes = 10  # must be at least 3
    num_instances = 5
    output_size = 1024

    train_set = CustomDataset(path=os.path.dirname(os.path.abspath(__file__)) + "/FinalDataset/bounding_box_train",
                              used_for_train=True)

    query_set = CustomDataset(path=os.path.dirname(os.path.abspath(__file__)) + "/FinalDataset/bounding_box_test",
                              used_for_train=False)
    test_set = CustomDataset(path=os.path.dirname(os.path.abspath(__file__)) + "/FinalDataset/query",
                             used_for_train=False)

    classifier = Classifier(resnet_type=50, output_size=output_size, mlp_on_top=True, device=device)

    train_dataloader = DataLoader(train_set,
                                  batch_sampler=CustomSampler(train_set.data, num_classes, num_instances), num_workers=2)

    query_dataloader = DataLoader(query_set, batch_size=num_classes*num_instances, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_set, batch_size=num_classes*num_instances, shuffle=False, num_workers=2)

    print("Starting training...")
    classifier.train_net(num_epochs=100, learning_rate=8*10**(-6), train_set=train_dataloader,
                         query_set=query_dataloader, test_set=test_dataloader)
