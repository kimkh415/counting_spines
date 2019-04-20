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
    neg_exampels = []

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

        for coord in coords[i]:
            (x, y) = coord
            new_x = int(y) + patch_dim
            new_y = int(x) + patch_dim

            avoid.append((new_x, new_y))

            box = arr[new_x - int(patch_dim/2): new_x + int(patch_dim/2), new_y - int(patch_dim/2): new_y + int(patch_dim/2)]
            pos_examples.append(box.reshape((1, patch_dim, patch_dim))/norm_factor)

            if not os.path.exists(Path(outname + "positive_examples/")):
                        os.makedirs(Path(outname + "positive_examples/"))

            if not os.path.exists(Path(outname + "negative_examples/")):
                os.makedirs(Path(outname + "negative_examples/"))

            for i in range(4):  
                tr_im = Image.fromarray(box)
                tr_im.save(outname + "positive_examples/" + "spine_image{}-{}.tif".format(pos_idx, i))
                box = np.rot90(box)
                pos_idx += 1

        for _ in avoid:

            overlap = True

            while overlap:
                x = np.random.randint(ori_arr.shape[0]) + patch_dim
                y = np.random.randint(ori_arr.shape[1]) + patch_dim

                overlap = check_overlap((x, y), avoid, patch_dim)

            box = arr[x - int(patch_dim/2): x + int(patch_dim/2), y - int(patch_dim/2): y + int(patch_dim/2)]
            neg_exampels.append(box.reshape((1, patch_dim, patch_dim))/norm_factor)

            for i in range(4):
                tr_im = Image.fromarray(box)
                tr_im.save(outname + "negative_examples/" + "spine_image{}-{}.tif".format(neg_idx, i))
                box = np.rot90(box)
                neg_idx += 1

    print(np.array(pos_examples).shape, "pos examples")

    np.save(outname + "np_arr_pos_x.npy", np.array(pos_examples))
    np.save(outname + "np_arr_pos_y.npy", np.ones((len(pos_examples),1)))
    np.save(outname + "np_arr_neg_x.npy", np.array(neg_exampels))
    np.save(outname + "np_arr_neg_y.npy", np.zeros((len(pos_examples),1)))


def check_overlap(point, box, patch_dim):
    (x, y) = point

    for coord in box:
        if coord[0]-int(patch_dim/2) < x < coord[0] + int(patch_dim/2) and coord[1] - int(patch_dim/2) < y < coord[1] + int(patch_dim/2):
            return True

    return False


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Data Preprocessing and patch \
    generation code")
    parser.add_argument("im_dir", help="Directory including the raw images to be \
    processed")
    args = parser.parse_args()


    tifs, infos = get_file_paths(args.im_dir)

    coords = extract_centers(infos)

    patch_dim = 40
    norm_factor = 100

    create_training_images(tifs, coords, "training_images/", patch_dim, norm_factor)

