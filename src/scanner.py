"""
Authors: Kwanho Kim, Saideep Gona, Jinke Liu

Contains code for a scanner object which traveses provided images
and outputs predicted output maps
"""

import os, argparse
from pathlib import Path
import numpy as np
import torch
from cnn import ConvNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim
import random
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import re
import pickle
import matplotlib.pyplot as plt


class Scanner():

    def __init__(self, image_dir, model_path, patch_size, output_dir, norm_factor):

        self.norm_factor = norm_factor
        self.image_dir= image_dir
        print(image_dir)
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()            # Must be ON for running inference

        self.patch_size = int(patch_size)

        self.output_dir = output_dir    # RECOMMENDED: Store in the same directory as the model generation output!
        # self.output_dir = 
        # self.output_dir = Path("/".join(model_path.split("/")[:-1]) + "/")   # Recommendation from above

    def pad_image(self, image):

        padded_image = np.pad(image, (self.patch_size, self.patch_size), 'constant', constant_values=0)

        return padded_image

    def load_images_labels(self):
        """
        Loads in all the images and labels in the provided directory
        """

        def get_file_paths(imdir):
            dirs = os.listdir(imdir)

            tifs = []
            infos = []

            for d in dirs:
                if os.path.isdir(os.path.join(imdir, d)):
                    d_num = d[5:]
                    im_path = os.path.join(imdir, d, "spine_image" + d_num + ".tif")
                    info_path = os.path.join(imdir, d, "spine_info" + d_num + ".txt")
                    tifs.append(im_path)
                    infos.append(info_path)

            return tifs, infos

        def extract_center(fname):

            f = open(fname, 'r')
            lines = f.read()
            m = re.findall("x = (-?[0-9]*), y = (-?[0-9]*)", lines)
            return m


        image_paths, info_paths = get_file_paths(self.image_dir)

        data = []

        for i in range(len(image_paths)):

            im = Image.open(image_paths[i])
            im = np.array(im)
            coords = extract_center(info_paths[i])

            data.append(
                {
                    "image": im,
                    "centers": coords,
                    "count": len(coords),
                    "scanned output": np.zeros(im.shape)
                }
            )
        
        self.data = data

    def scan_single_image(self, image):
        """
        Scans an input image using the preloaded model and outputs
        a mapping

        :param image: Unpadded image to be processed
        :return: Scanning output map of size pre-padded image
        """

        image = image/self.norm_factor              #Normalize!
        out_map = np.zeros((image.shape))
        # print(out_map.shape)
        max_x = self.patch_size + image.shape[0]
        max_y = self.patch_size + image.shape[1]
        scan_interval_x = range(self.patch_size, max_x)
        scan_interval_y = range(self.patch_size, max_y)
        padded = self.pad_image(image)
        # print(padded.shape)
        # print(max(scan_interval_x))
        # print(max(scan_interval_y))

        for x in scan_interval_x[::1]:
            for y in scan_interval_y[::1]:
                # print(x, " , ", y, " patch center coords, ImDims: (", str(image.shape[0]), ",", str(image.shape[1]), ")")
                half = int(self.patch_size/2)
                patch = padded[x - half: x + half, y - half: y + half]
                patch = np.reshape(patch, (1,1,self.patch_size, self.patch_size))
                patch = torch.as_tensor(patch, dtype=torch.float, device=self.device)
                cur_out = self.model.forward(patch).detach().numpy()
                # print(cur_out.shape, "cur_out")
                soft_out = np.exp(cur_out)/np.sum(np.exp(cur_out))
                # print(soft_out, "soft_out")
                # softmax_maxout_bin = np.argmax(soft_out)
                # output_prob = soft_out[softmax_maxout_bin]
                # print(soft_out[1])
                # print(cur_out)
                # print(torch.max(cur_out,1))
                # if softmax_maxout_bin == 1:
                #     print("yes")
                #     out_map[x-self.patch_size,y-self.patch_size] = output_prob

                out_map[x - self.patch_size, y - self.patch_size] = soft_out[0, 1]

        return out_map

    def store_scanned_data(self):
        """
        Stores scanned image data structure as a pickle
        """
        store_path = Path.joinpath(self.output_dir, "scanned_data.p")
        pickle.dump(self.data, open(store_path, "wb"))

    def scan_all_images(self, outdir):
        """
        Scans all images in provided directory
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        self.device = device
        self.model.to(self.device)

        self.load_images_labels()       # Load image data

        for i in range(len(self.data)):
            # if (i%100) != 0:
            #     continue
            print("Scanning Image: ", i, " -------------------------------")
            cur_im = self.data[i]["image"]

            cur_out = self.scan_single_image(cur_im)
            self.data[i]["scanned output"] = cur_out

            for center in self.data[i]["centers"]:
                print(center)
                q_patch = 3
                e_patch = 1
                label_im = np.ones((q_patch, q_patch)) * 255
                label_out = np.ones((q_patch, q_patch)) * 1.3
                cur_im[int(center[1])-e_patch:int(center[1])+e_patch+1, 
                int(center[0])-e_patch:int(center[0])+e_patch+1] = label_im
                cur_out[int(center[1])-e_patch:int(center[1])+e_patch+1, 
                int(center[0])-e_patch:int(center[0])+e_patch+1] = label_out

            plt.imshow(cur_im, cmap='gray')
            plt.savefig(os.path.join(outdir, str(i) + "_cur_im.png"))
            plt.clf()
            plt.imshow(cur_out)
            plt.savefig(os.path.join(outdir, str(i) + "_cur_out.png"))
            plt.clf()

        self.store_scanned_data()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image scanner for  10-707(Deep Learning) project")
    parser.add_argument("patch_size", help="Size of model patch size")
    parser.add_argument("images_dir", help="Directory containing images to be processed")
    parser.add_argument("model_path", help="Path to saved model")
    parser.add_argument("output_dir", help="Output directory of scanned image maps")
    args = parser.parse_args()

    # python scanner.py 40 C:\Users\Saideep\Documents\Github_Repos\MSCB_Sem1\Deep_Learning\Project\Labeled_Spines_Tavita\ C:\Users\Saideep\Documents\Github_Repos\MSCB_Sem1\Deep_Learning\Project\counting_spines\src\training_sessions\2019-04-1515_00_29\weights.pt C:\Users\Saideep\Documents\Github_Repos\MSCB_Sem1\Deep_Learning\Project\counting_spines\src\training_sessions\2019-04-1515_00_29\
    # Create scanner object

    norm_factor = 100
    scanner = Scanner(Path(args.images_dir), Path(args.model_path), int(args.patch_size), Path(args.output_dir), norm_factor)
    figdir = os.path.join(args.output_dir, "prediction_figures")
    if os.path.isdir(figdir) is False:
        os.mkdir(figdir)
    scanner.scan_all_images(figdir)
