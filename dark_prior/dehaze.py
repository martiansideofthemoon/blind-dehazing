import cv2
import logging
import sys
import yaml

from bunch import bunchify

# Our module imports
import steps

from config.arguments import parser
import tools


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    args = parser.parse_args()

    with open(args.constants, 'r') as f:
        constants = bunchify(yaml.load(f))

    logger.info("Loading image %s ..." % args.input)
    img = cv2.imread(args.input, flags=cv2.IMREAD_COLOR)
    # image scaled in 0-1 range
    img = img / 255.0

    logger.info("Generating dark channel prior ...")
    dark_channel = steps.generate_dark_channel(img, constants)

    logger.info("Estimating airlight ...")
    airlight = steps.estimate_airlight(img, dark_channel, constants)
    logger.info("Estimated airlight is %s", str(airlight))

    logger.info("Estimating transmission map ...")
    tmap = steps.estimate_tmap(dark_channel, constants)

    logger.info("Smooth transmission map ...")
    tmap = steps.smooth_tmap(img, tmap, constants)

    logger.info("Dehazing image ...")
    dehazed = steps.dehaze(img, airlight, tmap, constants)
    tools.show_img([img, dehazed])

if __name__ == '__main__':
    main()
