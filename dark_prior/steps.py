import cv2
import numpy as np

from guidedfilter import guided_filter

# Debugging only
import pdb
#pdb.set_trace()
import tools
#tools.show_img([dark_channel])


def generate_dark_channel(img, constants):
    patch_size = constants.PATCH_SIZE
    half = int(patch_size / 2)
    height, width, _ = img.shape
    dark_channel = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            h1, h2 = max(0, i - half), min(height - 1, i + half)
            w1, w2 = max(0, j - half), min(width - 1, j + half)
            raw_patch = img[h1:h2, w1:w2, :]
            dark_channel[i, j] = np.min(np.min(raw_patch, axis=2))
    return dark_channel


def estimate_airlight(img, dark_channel, constants):
    haze_bright_ratio = constants.HAZE_BRIGHT_RATIO
    height, width, num_channels = img.shape
    # Flattening the image and dark channel
    img_flat = np.reshape(img, [-1, num_channels])
    dark_channel = np.reshape(dark_channel, [-1])
    # Number of haze opaque pixels to look at
    num_bright = int(haze_bright_ratio * dark_channel.shape[0])
    haze_locations = np.argpartition(dark_channel, -num_bright)[-num_bright:]
    # Value of haze locations in the original image
    haze_pixels = img_flat[haze_locations]
    # Returning brightest pixel among haze_locations
    return haze_pixels[np.argmax(np.mean(haze_pixels, axis=1))]


def estimate_tmap(dark_channel, constants):
    keep_haze = constants.KEEP_HAZE
    tmap = np.ones(dark_channel.shape) - (1 - keep_haze) * dark_channel
    return tmap


def smooth_tmap(img, tmap, constants):
    epsilon = constants.EPSILON
    radius = constants.GUIDED_RADIUS
    normI = (img - img.min()) / (img.max() - img.min())
    smooth_tmap = guided_filter(normI.astype(np.float32), tmap.astype(np.float32), radius, epsilon)
    smooth_tmap = cv2.bilateralFilter(smooth_tmap.astype(np.float32), 0, 0.1, 5)
    return smooth_tmap


def dehaze(img, airlight, tmap, constants):
    tmap_lower = constants.TRANSMISSION_LOWER_BOUND
    height, width, num_channels = img.shape
    tmap = np.reshape(
        np.maximum(tmap, tmap_lower * np.ones(tmap.shape)),
        [height, width, 1]
    )
    airlight = np.reshape(airlight, [1, 1, num_channels])
    dehazed = (img - airlight) / tmap
    return dehazed + airlight
