"""Steps required by main dehazing program."""
import cv2

import numpy as np

from patch import Pair, Patch

from scipy import spatial

from sklearn.neighbors.kd_tree import KDTree

import sys

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
        patch_database = [0] * len(patches[k])
        std_database = [patch.std_dev for patch in patches[k]]

        index = np.argsort(std_database)
        interval = len(std_database) / num_buckets

        for i in range(num_buckets):
            for j in range(interval):
                patch_database[index[i * interval + j]] = i

        # Put it back in patches
        for i in range(len(patches[k])):
            patches[k][i].bucket = patch_database[i]


def generate_pairs(patches, constants, raw):
    k_nearest = constants.K_NEAREST
    num_patches = constants.NUM_QUERY_PATCHES
    scaled_imgs = len(patches)

    pairs = []
    query_database = []
    candidate_database = []
    index_database = []
    length_database = []
    for k in range(scaled_imgs):
        if raw:
            qp = [np.reshape(patch.raw_patch, [-1]) for patch in patches[k] if 7 <= patch.bucket <= 9]
        else:
            qp = [patch.norm_patch for patch in patches[k] if 7 <= patch.bucket <= 9]
        qi = [index for index, patch in enumerate(patches[k]) if 7 <= patch.bucket <= 9]

        # Choose lesser query patches through random selection to improve speed
        if len(qi) > num_patches:
            np.random.seed(0)
            selection = np.random.choice(np.arange(len(qi)), num_patches, replace=False).tolist()
            selection.sort()
            query_patches = [qp[i] for i in selection]
            query_indices = [qi[i] for i in selection]
        else:
            query_patches = qp
            query_indices = qi

        query_database.append(
            np.vstack([query_patches])
        )
        index_database.append(query_indices)
        length_database.append(len(query_indices))
        if raw:
            candidate_database.append(
                np.vstack([[np.reshape(patch.raw_patch, [-1]) for i, patch in enumerate(patches[k]) if 0 <= patch.bucket <= 5]])
            )
        else:
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


def filter_pairs(patches, pairs, constants, all_pairs):
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

        if correlation >= pair_threshold or all_pairs:
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
