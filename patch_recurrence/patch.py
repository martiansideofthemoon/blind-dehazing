import numpy as np

from scipy import spatial

import tools


class Patch(object):
    def __init__(self, raw_patch, patch_size):
        self.bucket = 0
        self.raw_patch = raw_patch
        self.patch_size = patch_size
        # Compute mean in each of the three channels individually
        means = np.mean(np.reshape(raw_patch, [patch_size * patch_size, 3]), axis=0)
        # Subtract mean color from all channels individually
        self.air_free_patch = raw_patch - np.reshape(means, [1, 1, 3])
        # All three channels concatenated before normalizing
        # mean_free_patch is 147 x 1, with channels B,G,R concatenated (7x7x3)
        self.mean_free_patch = np.reshape(np.transpose(self.air_free_patch, [2, 0, 1]), [-1])
        self.std_dev = np.std(self.mean_free_patch)

    def store(self, img, location):
        # If the patch passes std_dev test, the following vector used in KNN
        self.norm_patch = 0 if self.std_dev == 0 else self.mean_free_patch / self.std_dev
        # We want l2 norm of each vector to be 1
        self.norm_patch = self.norm_patch / np.sqrt(self.patch_size * self.patch_size * 3)
        # Used to refer to original location in future if needed
        self.location = location
        self.img = img


class Pair(object):
    def __init__(self, first, second):
        self.weight = 0.0
        self.first = first
        self.second = second
        distance = spatial.distance.correlation(
            first.norm_patch, second.norm_patch
        )
        self.correlation = 1 - distance
        # Estimate the airlights for this pair using least-squares
        air_free1 = np.reshape(first.air_free_patch, [-1, 3])
        air_free2 = np.reshape(second.air_free_patch, [-1, 3])
        raw1 = np.reshape(first.raw_patch, [-1, 3])
        raw2 = np.reshape(second.raw_patch, [-1, 3])
        # Equation (11) in Bahat et al.
        self.airlight = np.zeros(3)
        np.seterr(all='raise')
        for i in range(3):
            airlight = np.dot(
                np.transpose(air_free2[:, i] - air_free1[:, i]),
                np.multiply(raw1[:, i], air_free2[:, i]) - np.multiply(raw2[:, i], air_free1[:, i])
            )
            den = np.sum((air_free2[:, i] - air_free1[:, i]) ** 2)
            self.airlight[i] = 0 if den == 0 else airlight / den

    def calculate_outlier(self):
        raw1 = np.reshape(self.first.raw_patch, [-1, 3])
        raw2 = np.reshape(self.second.raw_patch, [-1, 3])
        airlight = np.reshape(self.airlight, [1, 3])    # split airlight along BGR
        std1 = self.first.std_dev
        std2 = self.second.std_dev
        # Checking maxima across BGR for both patches (equn 15)
        self.tlb1 = np.amax(1 - raw1 / airlight, axis=1)
        self.tlb2 = np.amax(1 - raw2 / airlight, axis=1)

        tlb1_max = np.max(self.tlb1)
        tlb2_max = np.max(self.tlb2)
        self.weight = ((tlb1_max - tlb2_max) * (tlb1_max / tlb2_max - 1)) ** 2

        self.outlier_indicator = np.mean(
            np.abs(self.tlb2 / self.tlb1 - std2 / std1)
        )

    def show(self):
        tools.draw_pairs(self)
