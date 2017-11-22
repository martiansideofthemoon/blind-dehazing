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


def draw_pairs(pair):
    # setup the figure
    fig = plt.figure()
    # show first image
    l1 = (pair.first.location[0], pair.first.location[1])
    l2 = (pair.first.location[0] + 8, pair.first.location[1] + 8)
    img1 = cv2.rectangle(pair.first.img * 255, l1, l2, (255, 0, 0), 2)

    l1 = (pair.second.location[0], pair.second.location[1])
    l2 = (pair.second.location[0] + 8, pair.second.location[1] + 8)
    img2 = cv2.rectangle(pair.second.img * 255, l1, l2, (255, 0, 0), 2)

    imgs = [img1, img2]
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs), i + 1)
        plt.imshow(cv2.cvtColor(np.array(np.abs(img), dtype=np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis("off")

    # show the images
    plt.show()
