import cv2
import time
import numpy as np

from scipy import spatial
from sklearn.neighbors import KDTree

from patch import Pair, Patch

import tools


def scale(img, scales):
    """Returns an array of images sized according to `scales`"""
    outputs = []
    for sc in scales:
        outputs.append(
            cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        )
    return outputs


def generate_patches(scaled_imgs, constants):
    patch_size = constants.PATCH_SIZE
    std_dev_threshold = constants.STD_DEV_THRESHOLD
    patches = []
    for k, sc in enumerate(scaled_imgs):
        img_patches = []
        for i in range(sc.shape[0] - patch_size):
            for j in range(sc.shape[1] - patch_size):
                raw_patch = sc[i:i + patch_size, j:j + patch_size, :]
                patch = Patch(
                    raw_patch=raw_patch,
                    patch_size=patch_size,
                )
                if patch.std_dev > std_dev_threshold:
                    patch.store(sc, [j, i])
                    img_patches.append(patch)
        patches.append(img_patches)
    return patches


def generate_pairs(patches, constants):
    k_nearest = constants.K_NEAREST
    scaled_imgs = len(patches)
    # Convert the list of patch norms into numpy arrays
    patch_database = []
    for k in range(scaled_imgs):
        patch_database.append(
            np.vstack([patch.norm_patch for patch in patches[k]])
        )
    # Find list of nearest neighbours for each patch
    # `total` is used to correct indices since we successively build smaller KD trees
    nearest = []
    total = 0
    for k in range(scaled_imgs):
        # This is done to avoid wasting time finding symmetric patches
        p1 = np.concatenate(patch_database[k:])
        kdt = KDTree(p1, leaf_size=30, metric='euclidean')

        # k+1 taken to account for self-matches
        k_value = min(k_nearest + 1, len(p1))
        nn = kdt.query(patch_database[k], k=k_value, return_distance=False, sort_results=False)

        # in the case p1 is very small, less than k
        if k_value < k_nearest + 1:
            extra = np.expand_dims(np.arange(nn.shape[0]), axis=1)
            for i in range(k_nearest + 1 - k_value):
                nn = np.concatenate((nn, extra), axis=1)

        nearest.append(total + nn)

        total += len(patch_database[k])

    return np.concatenate(nearest)


def remove_duplicates(pairs):
    unique_pairs = []
    pair_list = {}
    for i in range(pairs.shape[0]):
        for j in range(pairs.shape[1]):
            # This is to remove self-matches
            if i == pairs[i, j]:
                continue
            if ("%d,%d" % (i, pairs[i, j]) not in pair_list):
                # This is stored to remove symmetric duplicates
                pair_list["%d,%d" % (i, pairs[i, j])] = 1
                pair_list["%d,%d" % (pairs[i, j], i)] = 1
                unique_pairs.append([i, pairs[i, j]])
    return unique_pairs


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
