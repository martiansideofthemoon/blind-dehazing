import cv2
import time
import numpy as np

from scipy import spatial
from sklearn.neighbors import KDTree


def scale(img, scales):
    """Returns an array of images sized according to `scales`"""
    outputs = []
    for sc in scales:
        outputs.append(cv2.resize(img, None, fx=sc, fy=sc))
    return outputs


def generate_patches(scaled_imgs, constants):
    patch_size = constants.PATCH_SIZE
    std_dev_threshold = constants.STD_DEV_THRESHOLD
    mean_patches = []
    patches = []
    locations = []
    for sc in scaled_imgs:
        for i in range(sc.shape[0] - patch_size):
            for j in range(sc.shape[1] - patch_size):
                patch = sc[i:i + patch_size, j:j + patch_size, :]
                # Mean computed separately for each of the three channels
                means = np.mean(np.reshape(patch, [patch_size * patch_size, -1, 3]), axis=0)
                # Mean subtraction and conversion to grayscale
                mean_patch = patch - np.reshape(means, [1, 1, 3])
                patch = np.mean(mean_patch, axis=2)
                std_dev = np.std(patch)
                if std_dev > std_dev_threshold:
                    mean_patches.append(mean_patch)
                    patches.append(np.reshape(patch, [-1]) / std_dev)
                    locations.append([i, j])
    return np.array(mean_patches), np.array(patches), np.array(locations)


def generate_pairs(patches, constants):
    k_nearest = constants.K_NEAREST
    kdt = KDTree(patches, leaf_size=30, metric='euclidean')
    # k+1 taken to account for self-matches
    start_time = time.time()
    nearest = kdt.query(patches, k=k_nearest + 1, return_distance=False, sort_results=False)
    print(time.time() - start_time)
    return nearest


def filter_pairs(patches, pairs, constants):
    pair_threshold = constants.PAIR_THRESHOLD
    filtered_pairs = set()
    for i in range(pairs.shape[0]):
        for j in range(pairs.shape[1]):
            if i == pairs[i, j]:
                continue
            correlation = 1 - spatial.distance.correlation(patches[i], patches[pairs[i, j]])
            if correlation < pair_threshold:
                continue
            # This if condition is to remove symmetric duplicates from the set
            if i < pairs[i, j]:
                filtered_pairs.add((i, pairs[i, j]))
            else:
                filtered_pairs.add((pairs[i, j], i))
    return filtered_pairs
