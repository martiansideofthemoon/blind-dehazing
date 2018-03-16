"""Steps required by main dehazing program."""
import cv2

import numpy as np

from patch import Pair, Patch

import math

from scipy import spatial

from sklearn.neighbors.kd_tree import KDTree

from decimal import Decimal
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

import sys

import tools

import logging

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def scale(img, scales):
    """Returns an array of images sized according to `scales`
    """
    outputs = []
    for sc in scales:
        outputs.append(
            cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        )
    return outputs


def generate_patches(scaled_imgs, constants, all_patches):
    patch_size = constants.PATCH_SIZE
    step = 1 if all_patches else 2
    patches = []
    for k, sc in enumerate(scaled_imgs):
        img_patches = []
        for i in range(0, sc.shape[0] - patch_size, step):
            for j in range(0, sc.shape[1] - patch_size, step):
                raw_patch = sc[i:i + patch_size, j:j + patch_size, :]
                patch = Patch(
                    raw_patch=raw_patch,
                    patch_size=patch_size,
                )
                patch.store(sc, [i, j])
                img_patches.append(patch)
        patches.append(img_patches)
    return patches


def smoothen(scaled_imgs, patches, constants):
    """Applying Gaussian filter
    to smoothen std deviations of all patches
    """
    patch_size = constants.PATCH_SIZE

    for k in range(len(patches)):
        img = scaled_imgs[k]
        patch = patches[k]

        # For half patches
        length_sd_array = width_sd_array = 0
        for i in range(0, img.shape[0] - patch_size, 2):
            length_sd_array += 1
        for i in range(0, img.shape[1] - patch_size, 2):
            width_sd_array += 1

        std_database = np.reshape(map(lambda x: x.std_dev, patch), [length_sd_array, width_sd_array])
        blur = np.reshape(cv2.GaussianBlur(std_database, (7, 7), sigmaX=6, sigmaY=6), [-1])
        map(lambda (i, x): setattr(x, 'std_dev', blur[i]), enumerate(patch))


def set_patch_buckets(patches, constants):
    """Assigning bucket numbers to each patch
    Histogram equalization across every scaled image
    """
    num_buckets = constants.NUM_BUCKETS
    scaled_imgs = len(patches)

    for k in range(scaled_imgs):
        std_database = []
        patch_database = []
        for patch in patches[k]:
            std_database.append(patch.std_dev)
            patch_database.append(0)

        index = np.argsort(std_database)
        interval = len(std_database) / num_buckets

        for i in range(num_buckets):
            for j in range(interval):
                patch_database[index[i * interval + j]] = i

        # Put it back in patches
        for i in range(len(patches[k])):
            patches[k][i].bucket = patch_database[i]


def generate_pairs(patches, constants):
    k_nearest = constants.K_NEAREST
    scaled_imgs = len(patches)

    pairs = []
    query_database = []
    candidate_database = []
    index_database = []
    length_database = []
    for k in range(scaled_imgs):
        a = [patch.norm_patch for patch in patches[k] if 7 <= patch.bucket <= 9]
        x = [index for index, patch in enumerate(patches[k]) if 7 <= patch.bucket <= 9]

        if len(x) > 20:
            np.random.seed(0)
            t = np.random.choice(np.arange(len(x)), 20, replace=False).tolist()
            t.sort()
            a_sample = [a[i] for i in t]
            x_sample = [x[i] for i in t]
        else:
            a_sample = a
            x_sample = x

        query_database.append(
            np.vstack([a_sample])
        )
        index_database.append(x_sample)
        length_database.append(len(x_sample))
        candidate_database.append(
            np.vstack([[patch.norm_patch for i, patch in enumerate(patches[k]) if 0 <= patch.bucket <= 5]])
        )

    p1 = np.concatenate(candidate_database)
    kdt = KDTree(p1, leaf_size=30, metric='euclidean')

    # Find list of nearest neighbours for each patch
    # `total` is used to correct indices of queried patches for every iteration
    total = 0
    for k in range(scaled_imgs):
        nn = kdt.query(query_database[k], k=k_nearest, return_distance=False, sort_results=False)
        q = [total + index_database[k][i] for i in range(length_database[k])]
        for i in range(len(nn)):
            for j in range(k_nearest):
                pairs.append([q[i], nn[i][j]])
        total += len(patches[k])

    return pairs


def filter_pairs(patches, pairs, constants):
    pair_threshold = constants.PAIR_THRESHOLD
    # Convert the list of patch norms into numpy arrays
    patch_database = []
    patches2 = []
    for k in range(len(patches)):
        patch_database.append(
            np.vstack([patch.norm_patch for patch in patches[k]])
        )
        patches2.extend(patches[k])
    patch_database = np.concatenate(patch_database)

    filtered_pairs = []
    for i, j in pairs:
        # Thresholding pairs based on last line in 3.1
        distance = spatial.distance.correlation(
            patch_database[i], patch_database[j]
        )
        correlation = 1 - distance

        if correlation >= pair_threshold:
            filtered_pairs.append(
                Pair(patches2[i], patches2[j])
            )
    return np.array(filtered_pairs)


def remove_outliers(pairs, constants):
    outlier_threshold = constants.OUTLIER_THRESHOLD
    new_pairs = []
    for pair in pairs:
        pair.calculate_outlier()
        if pair.outlier_indicator <= outlier_threshold:
            new_pairs.append(pair)
    return new_pairs


def estimate_airlight(pairs):
    numerator = np.zeros(3)
    denominator = 0.0
    for pair in pairs:
        numerator += pair.weight * pair.airlight
        denominator += pair.weight
    return (numerator / denominator)


# Tmap estimation functions

def sigmoid(y):
    """Returns sigmoid value of pixel x
    """
    l2_norm = math.sqrt(sum(map(lambda x: x * x, y)))
    res = 1 / 1 + Decimal(48 * (l2_norm - 0.1)).exp()
    return res


def get_norm(x):
    height, width = x.size(0), x.size(1)
    log = torch.log(x)

    log = log.view(1, 1, height, width)
    diff1 = torch.Tensor([-1, 0, 1])
    diff1 = Variable(diff1.view(1, 1, 1, 3), requires_grad=True)
    diff2 = torch.Tensor([1, 0, -1])
    diff2 = Variable(diff2.view(1, 1, 3, 1), requires_grad=True)

    # conv1 and conv2 should contain the x and y components resp, of grad(log)
    conv1 = F.conv2d(log, diff1, padding=(0, 1))
    conv2 = F.conv2d(log, diff2, padding=(1, 0))

    l2_norm = torch.mul(conv1, conv1) + torch.mul(conv2, conv2)
    l2_norm.data.resize_(height * width)
    return l2_norm


def loss_fun(x, sig):
    l2_norm = get_norm(x)
    sig2 = np.asarray(sig, dtype=np.float32)
    sig1 = torch.from_numpy(sig2)
    t = torch.mul(l2_norm.data, sig1)
    ret = Variable(torch.FloatTensor([torch.sum(t)]), requires_grad=True)
    return ret


def minimization(sig, tlb):
    """Returns new t-map
    """
    t_height, t_width = len(tlb), len(tlb[0])
    tlb = torch.FloatTensor(np.reshape(tlb, [t_height, t_width]))
    for i in range(t_height):
        for j in range(t_width):
            if tlb[i][j] <= 0:
                tlb[i][j] = 10 ** -7

    weight = Parameter(torch.Tensor(t_height, t_width), requires_grad=True)
    weight.data = tlb

    optimizer = torch.optim.SGD([weight], lr=0.1)
    loss = loss_fun(weight, sig)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    t = weight.data
    return t


def estimate_tmap(img, patches, airlight, constants):
    """Estimates t-map and returns dehazed output image after 20 iterations
    """
    patch_size = constants.PATCH_SIZE
    tlb = np.empty([len(patches)])
    for index, patch in enumerate(patches):
        raw = np.reshape(patch.raw_patch, [-1, 3])
        temp = 1 - raw / airlight
        tlb[index] = max(temp[patch_size ** 2 // 2])
    h, w = img.shape[0], img.shape[1]
    tlb = np.reshape(tlb, [h - patch_size, w - patch_size, 1])
    img = img[0:h - patch_size, 0:w - patch_size]
    img = np.reshape(img, [h - patch_size, w - patch_size, 3])
    l_img = (img - airlight) / tlb + airlight

    # Sigmoid calculation
    grad = np.empty([3, len(l_img) * len(l_img[0])])
    l_img = np.reshape(l_img, [3, -1])
    grad[0] = np.gradient(l_img[0])
    grad[1] = np.gradient(l_img[1])
    grad[2] = np.gradient(l_img[2])
    grad = np.reshape(grad, [-1, 3])
    l_img = np.reshape(l_img, [-1, 3])
    sig = torch.Tensor([float(sigmoid(grad[i])) for i in range(len(l_img))])

    # Run through 10 iterations
    t_prev = tlb
    for i in range(1):
        t_curr = minimization(sig, t_prev)
        t_curr = t_curr.numpy()
        t_curr = np.reshape(t_curr, [h - patch_size, w - patch_size, 1])
        l_img = (img - airlight) / t_curr + airlight
        t_prev = t_curr
        tools.show_img([img, l_img])

    return l_img
