import cv2
import logging
import sys
import yaml
import numpy as np
from bunch import bunchify

# Our module imports
import utils

from config.arguments import parser


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PATCH_SIZE = 7


def show_img(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def main():
    args = parser.parse_args()
    with open(args.constants, 'r') as f:
        constants = bunchify(yaml.load(f))

    logger.info("Loading image %s ..." % args.input)
    img = cv2.imread(args.input, flags=cv2.IMREAD_COLOR)
    scaled_imgs = utils.scale(img, [0.25, 0.5, 0.75, 1])
    logger.info("Extracting patches ...")
    mean_patches, patches, locations = utils.generate_patches(scaled_imgs, constants)
    pairs = utils.generate_pairs(patches, constants)
    pairs = utils.filter_pairs(patches, pairs, constants)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
