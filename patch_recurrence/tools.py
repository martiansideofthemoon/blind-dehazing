import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import numpy as np
import pdb


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

#..............................................................................................#
# purva's functions

# Assigning colors to each patch based on bucket number - also, resizing img
def set_buckets(img, patch, constants):
    num_buckets = constants.NUM_BUCKETS
    patch_size = constants.PATCH_SIZE

    new_height = img.shape[0] - patch_size
    new_width = img.shape[1] - patch_size
    img = img[0:new_height, 0:new_width]

    img = np.reshape(img, [-1, 3])

    bucket_colors = [[0.8, 0.8, 1], [0.6, 0.6, 1], [0.4, 0.4, 1], [0.2, 0.2, 1], [0, 0, 1], [0, 0, 0.8], [0, 0, 0.6], [0, 0, 0.4], [0, 0, 0.2], [0, 0, 0]]  # dict of colors from 0-(num_buckets-1)
    for i in range(new_height*new_width):
        for j in range(num_buckets):
            if patch[i].bucket == j:
                img[i] = bucket_colors[j]
                break

    img = np.reshape(img, [new_height, new_width, 3])
    return img


# Display buckety image
def show_buckety_img(imgs):
    # setup the figure
    fig = plt.figure()
    # show first image
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs), i + 1)
        plt.imshow(cv2.cvtColor(np.array(np.abs(img) * 255, dtype=np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis("off")

    # purva code
    ax = fig.add_subplot(3, 4, 4)
    plt.axis("off")
    legend_data = [
        [
            1,
            [1, 0.8, 0.8],
            "1"
        ],
        [
            2,
            [1, 0.6, 0.6],
            "2"
        ],
        [
            3,
            [1, 0.4, 0.4],
            "3"
        ],
        [
            4,
            [1, 0.2, 0.2],
            "4"
        ],
        [
            5,
            [1, 0, 0],
            "5"
        ],
        [
            6,
            [0.8, 0, 0],
            "6"
        ],
        [
            7,
            [0.6, 0, 0],
            "7"
        ],
        [
            8,
            [0.4, 0, 0],
            "8"
        ],
        [
            9,
            [0.2, 0, 0],
            "9"
        ],
        [
            10,
            [0, 0, 0],
            "10 (highest std)"
        ]
    ]
    handles = [
        Rectangle((0,0),1,1, color = [v for v in c]) for k,c,n in legend_data
    ]
    labels = [n for k,c,n in legend_data]

    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(handles,labels, prop=fontP)
    # purva end

    # show the images
    plt.show()

#..............................................................................................#
