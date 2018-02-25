import cv2
import random
import numpy as np

from scipy import spatial
from sklearn.neighbors.kd_tree import KDTree

from patch import Pair, Patch


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
                if all_patches or patch.std_dev > std_dev_threshold:
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

        length_sd_array = img.shape[0] - patch_size
        width_sd_array = img.shape[1] - patch_size

        std_database = np.reshape(map(lambda x: x.std_dev, patch), [length_sd_array, width_sd_array])
        blur = np.reshape(cv2.GaussianBlur(std_database, (7, 7), sigmaX=6, sigmaY=6), [-1])
        map(lambda (i, x): setattr(x, 'std_dev', blur[i]), enumerate(patch))


def set_patch_buckets(patches, constants):
    """Assigning bucket numbers to each patch
    """
    num_buckets = constants.NUM_BUCKETS
    scaled_imgs = len(patches)
    width = max(len(patches[i]) for i in range(scaled_imgs))

    std_database = np.zeros((scaled_imgs, width))
    for k in range(scaled_imgs):
        for index, patch in enumerate(patches[k]):
            std_database[k][index] = patch.std_dev

    min_std = np.amin(std_database)
    max_std = np.amax(std_database)

    norm_std = np.floor((std_database - min_std) * (num_buckets - 1) / (max_std - min_std))
    for k in range(scaled_imgs):
        map(lambda (i,x): setattr(x, 'bucket', norm_std[k][i]), enumerate(patches[k]))


def generate_pairs(imgs, patches, constants):
    k_nearest = constants.K_NEAREST
    patch_size = constants.PATCH_SIZE
    scaled_imgs = len(patches)

    patch_database = []
    query_database = []
    candidate_database = []
    index_database = []
    length_database = []
    for k in range(scaled_imgs):
        patch_database.append(
            np.vstack([[patch.norm_patch for patch in patches[k] if 6 <= patch.bucket <= 9]])
        )

        x = [index for index,patch in enumerate(patches[k]) if 6 <= patch.bucket <= 9]
        index_database.append([x[i] for i in range(len(x))])

        length_database.append(len(x))

        query_database.append(
            np.vstack([patches[k][index_database[k][i]].norm_patch for i in range(len(x))])
        )

        candidate_database.append(
            np.vstack([[patch.norm_patch for i, patch in enumerate(patches[k]) if 0 <= patch.bucket <= 5]])
        )

    p1 = np.concatenate(candidate_database[0:])
    kdt = KDTree(p1, leaf_size=30, metric='euclidean')

    # Find list of nearest neighbours for each patch
    # `total` is used to correct indices since we successively build smaller KD trees
    nearest = []
    t = 0
    for k in range(scaled_imgs):
        nn = kdt.query(query_database[k], k=k_nearest, return_distance=False, sort_results=False)

        # Append query patch index at the end of nn
        b = np.array([[t + index_database[k][i] for i in range(length_database[k])]])
        nn = np.concatenate((nn, b.T), axis=1)

        nearest.append(nn)
        t += len(patches[k])

    return np.concatenate(nearest)


def remove_duplicates(pairs):
    unique_pairs = []
    pair_list = {}
    for i in range(len(pairs)):
        for j in range(len(pairs[0]) - 1):
            # This is to remove self-matches
            if i == pairs[i][j]:
                continue
            if ("%d,%d" % (pairs[i][9], pairs[i][j]) not in pair_list):
                # This is stored to remove symmetric duplicates
                pair_list["%d,%d" % (pairs[i][9], pairs[i][j])] = 1
                pair_list["%d,%d" % (pairs[i][j], pairs[i][9])] = 1
                unique_pairs.append([pairs[i][9], pairs[i][j]])
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
