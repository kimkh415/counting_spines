import os, sys
import numpy as np
from PIL import Image
from gen_data import get_file_paths, extract_centers


def plot_centers_on_original(tifs, centers, outdir):
    for i, tif in enumerate(tifs):
        if i % 100 == 0:
            im = Image.open(tif)
            ori_arr = np.array(im)

            ori_im = Image.fromarray(ori_arr)
            ori_im.save(os.path.join(outdir, "{}_original.png".format(i)))

            ori_arr = np.pad(ori_arr, (40, 40), 'constant', constant_values=0)

            for c in centers[i]:
                c = (int(c[0]) + 40, int(c[1]) + 40)

                ori_arr[int(c[1])][int(c[0])] = 255
                ori_arr[int(c[1])+1][int(c[0])] = 255
                ori_arr[int(c[1])-1][int(c[0])] = 255
                ori_arr[int(c[1])][int(c[0])+1] = 255
                ori_arr[int(c[1])][int(c[0])-1] = 255
                ori_arr[int(c[1])+1][int(c[0])-1] = 255
                ori_arr[int(c[1])-1][int(c[0])+1] = 255
                ori_arr[int(c[1])+1][int(c[0])+1] = 255
                ori_arr[int(c[1])-1][int(c[0])-1] = 255

                for x in range(int(c[1])-20, int(c[1])+20):
                    ori_arr[x][int(c[0]) - 20] = 255
                    ori_arr[x][int(c[0]) + 20] = 255

                for y in range(int(c[0])-20, int(c[0])+20):
                    ori_arr[int(c[1])-20][y] = 255
                    ori_arr[int(c[1])+20][y] = 255

            new_im = Image.fromarray(ori_arr)
            new_im.save(os.path.join(outdir, "{}_center_box.png".format(i)))


if __name__ == "__main__":
    output_pwd = sys.argv[1]

    original_image = "D:\github_repos\counting_spines\Labeled_Spines_Tavita"
    tifs, infos = get_file_paths(original_image)
    centers = extract_centers(infos)

    plot_centers_on_original(tifs, centers, output_pwd)
