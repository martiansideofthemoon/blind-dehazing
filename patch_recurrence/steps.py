import cv2
import time
import numpy as np

from scipy import spatial
#from sklearn.neighbors import KDTree
from sklearn.neighbors.kd_tree import KDTree

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
                    patch.store(sc, [i, j])
                    img_patches.append(patch)
        patches.append(img_patches)
    return patches

#..............................................................................................#
# purva's functions

# exactly like generate_patches without the std deviation condition
def generate_all_patches(scaled_imgs, constants):
    patch_size = constants.PATCH_SIZE
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
                patch.store(sc, [i, j])
                img_patches.append(patch)
        patches.append(img_patches)
    return patches


# assigning bucket numbers to each patch
def set_patch_buckets(patches, constants):
    num_buckets = constants.NUM_BUCKETS
    scaled_imgs = len(patches)
    std_database = np.zeros((len(patches), len(patches[0])))
    for k in range(scaled_imgs):
        for index, patch in enumerate(patches[k]):
            std_database[k][index] = patch.std_dev

    min_std = np.amin(std_database)
    max_std = np.amax(std_database)
    bucket_size = (max_std - min_std) / num_buckets

    for k in range(scaled_imgs):
        for index, patch in enumerate(patches[k]):
            for b in range(num_buckets):
                if min_std + b * bucket_size <= std_database[k][index] <= min_std + (b+1) * bucket_size:
                    patches[k][index].bucket = b
                    break

#..............................................................................................#

def generate_pairs(patches, constants):
    k_nearest = constants.K_NEAREST
    scaled_imgs = len(patches)      # only those patches with high std deviation
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


def remove_overlaps(pairs, constants):
    patch_size = constants.PATCH_SIZE
    new_pairs = []
    for p in pairs:
        l1, s1 = p.first.location, p.first.img.shape
        l1_norm = (float(l1[0]) / s1[0], float(l1[1]) / s1[1])
        p1 = (float(patch_size) / s1[0], float(patch_size) / s1[1])
        l2, s2 = (p.second.location, p.second.img.shape)
        l2_norm = float(l2[0]) / s2[0], float(l2[1]) / s2[2]
        p2 = (float(patch_size) / s2[0], float(patch_size) / s2[1])
        # To ensure that images are away from each other ?
        if np.abs(l1_norm[0] - l2_norm[0]) > min(p1[0], p2[0]) and np.abs(l1_norm[1] - l2_norm[1]) > min(p1[1], p2[1]):
            new_pairs.append(p)
    return new_pairs


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
