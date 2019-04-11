from PIL import Image
import numpy as np
import sys
import os
import re


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


def create_training_images(tifs, coords, outname):

    pos_examples = []
    neg_exampels = []

    pos_idx = 1
    neg_idx = 1

    for i, tif in enumerate(tifs):
        im = Image.open(tif)
        ori_arr = np.array(im)
        arr = np.pad(ori_arr, (40, 40), 'constant', constant_values=0)

        avoid = []

        for coord in coords[i]:
            (x, y) = coord
            new_x = int(y) + 40
            new_y = int(x) + 40

            avoid.append((new_x, new_y))

            box = arr[new_x - 20: new_x + 20, new_y - 20: new_y + 20]
            pos_examples.append(box + [1])

            tr_im = Image.fromarray(box)
            tr_im.save(outname + "positive_examples/" + "spine_image{}.tif".format(pos_idx))
            pos_idx += 1

        for _ in avoid:

            overlap = True

            while overlap:
                x = np.random.randint(ori_arr.shape[0]) + 40
                y = np.random.randint(ori_arr.shape[1]) + 40

                overlap = check_overlap((x, y), avoid)

            box = arr[x - 20: x + 20, y - 20: y + 20]
            neg_exampels.append(box + [0])

            tr_im = Image.fromarray(box)
            tr_im.save(outname + "negative_examples/" + "spine_image{}.tif".format(neg_idx))
            neg_idx += 1

    np.save(outname + "np_arr_pos.npy", np.array(pos_examples))
    np.save(outname + "np_arr_neg.npy", np.array(neg_exampels))


def check_overlap(point, box):
    (x, y) = point

    for coord in box:
        if coord[0]-20 < x < coord[0] + 20 and coord[1] - 20 < y < coord[1] + 20:
            return True

    return False


if __name__ == "__main__":
    im_dir = sys.argv[1]

    tifs, infos = get_file_paths(im_dir)

    coords = extract_centers(infos)

    create_training_images(tifs, coords, "training_images/")

