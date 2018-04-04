import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-ip", "--input", default="data/cityscape.png", type=str, help="Input image file")
parser.add_argument("-seed", "--seed", default=1, type=int, help="value of the random seed")
parser.add_argument("-op", "--output", default="dehazed.png", type=str, help="Output image file")
parser.add_argument("-c", "--constants", default="patch_recurrence/config/constants.yml", type=str, help="Predefined constants")
parser.add_argument("-no_cache", "--no-cache", default=False, action="store_true", help="Use cache or not")
