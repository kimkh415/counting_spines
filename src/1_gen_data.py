"""
Authors: Kwanho Kim, Saideep Gona, Jinke Liu

Contains code for generating training patches from original dendritic
spine microscopy images.
"""

from PIL import Image
import argparse
import numpy as np
import sys
import os
import re
from pathlib import Path
import configparser

from sklearn.preprocessing import scale


def get_file_paths(pwd):
    dirs = os.listdir(pwd)

    tifs = []
    infos = []

    for d in dirs:
        if os.path.isdir(os.path.join(pwd, d)):
            d_num = d[5:]
            im_path = os.path.join(pwd, d, "spine_image" + d_num + ".tif")
            info_path = os.path.join(pwd, d, "spine_info" + d_num + ".txt")
            tifs.append(im_path)
            infos.append(info_path)

    return tifs, infos


def extract_centers(fnames):
    spine_coords = []

    for info in fnames:
        f = open(info, 'r')
        lines = f.read()
        m = re.findall("x = (-?[0-9]*), y = (-?[0-9]*)", lines)
        spine_coords.append(m)

    return spine_coords


def create_training_images(tifs, coords, outname, patch_dim, norm_factor):

    pos_examples = []
    neg_examples = []

    pos_idx = 1
    neg_idx = 1

    if not os.path.exists(Path(outname)):
        os.makedirs(Path(outname))

    for i, tif in enumerate(tifs):
        print("Processing Image: ", i)
        im = Image.open(tif)
        ori_arr = np.array(im)
        arr = np.pad(ori_arr, (patch_dim, patch_dim), 'constant', constant_values=0)

        avoid = []
        cur_positives = []

        for coord in coords[i]:
            (x, y) = coord
            new_x = int(y) + patch_dim
            new_y = int(x) + patch_dim

            avoid.append((new_x, new_y))

            box = arr[new_x - int(patch_dim/2): new_x + int(patch_dim/2), new_y - int(patch_dim/2): new_y + int(patch_dim/2)]

            if not os.path.exists(Path(outname + "positive_examples")):
                os.makedirs(Path(outname + "positive_examples"))

            if not os.path.exists(Path(outname + "negative_examples")):
                os.makedirs(Path(outname + "negative_examples"))

            for i in range(4):
                pos = box.reshape((1, patch_dim, patch_dim)) / norm_factor
                pos_examples.append(pos)
                cur_positives.append(pos)

                tr_im = Image.fromarray(box)
                tr_im.save(os.path.join(outname,"positive_examples/","spine_image{}-{}.tif".format(pos_idx, i)))
                box = np.rot90(box)
                pos_idx += 1

            cur_pos_dist = analyze_positive_examples(cur_positives)
            strategy_dict = {
                "type": 1,
                "threshold": cur_pos_dist[-1] * 10.0
            }

        for _ in range(len(avoid)*4):

            too_dim = True              # Check if an image that does not overlap also is not too dim
            fails = 0
            while too_dim:
                fails += 1
                overlap = True          # Check if the current selection does not overlap with centers

                while overlap:
                    x = np.random.randint(ori_arr.shape[0]) + patch_dim
                    y = np.random.randint(ori_arr.shape[1]) + patch_dim

                    overlap = check_overlap((x, y), avoid, patch_dim)

                box = arr[x - int(patch_dim/2): x + int(patch_dim/2), y - int(patch_dim/2): y + int(patch_dim/2)]
                too_dim = not pass_negative_filter(box, strategy_dict)

            # print(fails)

            neg_examples.append(box.reshape((1, patch_dim, patch_dim))/norm_factor)
            tr_im = Image.fromarray(box)
            tr_im.save(outname + "negative_examples/" + "spine_image{}.tif".format(neg_idx))
            box = np.rot90(box)
            neg_idx += 1

    print(np.array(pos_examples).shape, "pos examples")
    print(np.array(neg_examples).shape, "neg examples")

    np.save(outname + "np_arr_pos_x.npy", np.array(pos_examples))
    np.save(outname + "np_arr_pos_y.npy", np.ones((len(pos_examples),1)))
    np.save(outname + "np_arr_neg_x.npy", np.array(neg_examples))
    np.save(outname + "np_arr_neg_y.npy", np.zeros((len(pos_examples),1)))


def check_overlap(point, box, patch_dim):
    (x, y) = point
    half_patch = patch_dim/2
    flex_factor = 0.8
    bound = int(half_patch * flex_factor)
    for coord in box:
        if coord[0] - bound < x < coord[0] + bound and coord[1] - bound < y < coord[1] + bound:
            return True

    return False


def analyze_positive_examples(positive_patches):
    """
    Analyzes positive training examples to figure out their distribution of intensity values

    :param positive_patches: List of positive patch images to analyze
    :return: summary distributions of positive examples
    """

    summed_intensities = []

    for x in range(len(positive_patches)):
        summed_intensities.append(np.sum(positive_patches[x]))

    return summed_intensities


def pass_negative_filter(image, strategy, pass_through=0.05):
    """
    Strategy 1: If the overall intensity of an image is above a threshold, return True
                {
                "strategy": 1,
                "threshold": summed intensity value
                }
    Strategy 2: If more than a given percentage of pixels in an image are above a threshold, return True
                {
                "strategy": 2,
                "bright_threshold": intensity value,
                "percent_threshold": decimal value
                }

    :param image: Numpy matrix image to be assessed
    :param strategy: Flexible strategy dictionary with information for executing given strategy
    :param pass_through: Random chance that a negative example makes it through the threshold for free
    :return: Boolean of whether the image is a valid negative training example
    """

    r = np.random.uniform(0,1)
    if r < pass_through:
        return True

    if strategy["type"] == 1:
        total_sum = np.sum(image)
        if total_sum < strategy["threshold"]:
            return False
        else:
            return True

    elif strategy["type"] == 2:
        is_above_thresh = image > strategy["bright_threshold"]
        percent_bright = np.sum(is_above_thresh)/is_above_thresh.size
        if percent_bright < strategy["percent_threshold"]:
            return False
        else:
            return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data Preprocessing and patch \
    generation code")
    # parser.add_argument("im_dir", help="Directory including the raw images to be \
    # processed")
    parser.add_argument("config_file", help="Path to config file for pipeline")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.sections()
    config.read(args.config_file)

    root_out = config['DEFAULT']['output_directory']

    tifs, infos = get_file_paths(config['DEFAULT']['image_directory'])
    # tifs, infos = get_file_paths(args.im_dir)

    coords = extract_centers(infos)

    patch_dim = int(config['DEFAULT']['patch_dim'])
    norm_factor = int(config['DEFAULT']['norm_factor'])

    create_training_images(tifs, coords, root_out + "/training_images/", patch_dim, norm_factor)

