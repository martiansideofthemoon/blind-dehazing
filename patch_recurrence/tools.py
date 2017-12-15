import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_img(imgs):
    # setup the figure
    fig = plt.figure()
    # show first image
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs), i + 1)
        plt.imshow(cv2.cvtColor(np.array(np.abs(img) * 255, dtype=np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis("off")

    # show the images
    plt.show()


def show_patches(patches):
    img1 = np.copy(patches[0].img)
    patch_size = patches[0].patch_size
    for patch in patches:
        h1, h2 = patch.location[0], patch.location[0] + patch_size
        w1, w2 = patch.location[1], patch.location[1] + patch_size
        img1[h1:h2, w1:w2, :] = np.zeros((patch_size, patch_size, 3))
    show_img([img1])


def show_pair(pair):
    # setup the figure
    fig = plt.figure()
    # show first image
    l1 = (pair.first.location[1] - 1, pair.first.location[0] - 1)
    l2 = (pair.first.location[1] + 8, pair.first.location[0] + 8)
    img1 = cv2.rectangle(pair.first.img * 255, l1, l2, (255, 0, 0), 1)

    l1 = (pair.second.location[1] - 1, pair.second.location[0] - 1)
    l2 = (pair.second.location[1] + 8, pair.second.location[0] + 8)
    img2 = cv2.rectangle(pair.second.img * 255, l1, l2, (255, 0, 0), 1)

    imgs = [img1, img2]
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs), i + 1)
        plt.imshow(cv2.cvtColor(np.array(np.abs(img), dtype=np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis("off")

    # show the images
    plt.show()
