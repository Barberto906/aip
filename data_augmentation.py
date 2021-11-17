import math
import random
import numpy as np
import torchvision.transforms as T


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.shape[1] * img.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            # W/H
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

                #img = np.transpose(img, (1, 2, 0))
                return img

        return img


def build_transforms(config=False, is_train=True):
    image_size = [224, 448]
    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        transform = T.Compose([
            T.Resize(image_size),
            #T.RandomHorizontalFlip(p=config.prob),
            T.RandomHorizontalFlip(p=0.2),
            T.Pad(10),
            #T.RandomCrop(config.image_size),
            T.RandomCrop(image_size),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.3, mean=[0.485, 0.456, 0.406])
        ])
    else:
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            normalize_transform
        ])

    return transform