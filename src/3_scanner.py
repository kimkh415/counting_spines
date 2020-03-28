"""
Authors: Kwanho Kim, Saideep Gona, Jinke Liu

Contains code for a scanner object which traveses provided images
and outputs predicted output maps
"""

import os, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim
from PIL import Image
import re
import pickle
import matplotlib.pyplot as plt
import configparser
import glob

class ConvNet(nn.Module):
    """
    c1_out = convolutional layer 1 filter count
    c2_out = convolutional layer 2 filter count
    out_size = number of labels
    """

    def __init__(self, c1_out, c2_out, c3_out, l1_out, l2_out, out_size, kernel_size, patch_size, pool_size, pad, dropout_prob):
        super(ConvNet, self).__init__()
        # 1 input image channel
        print(patch_size)
        self.dp = dropout_prob

        self.conv1 = nn.Conv2d(1, c1_out, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size, padding=pad)
        # self.bn1 = nn.BatchNorm2d(c2_out)
        # self.do1 = nn.Dropout2d(self.dp)
        self.conv3 = nn.Conv2d(c2_out, c3_out, kernel_size, padding=pad)
        # self.bn2 = nn.BatchNorm2d(c3_out)
        # self.do2 = nn.Dropout2d(self.dp)

        self.pool_size = pool_size
        self.convout_size = int(c3_out * (patch_size/pool_size**3)**2)
        # self.convout_size = 3600
 
        print(self.convout_size, " size of convolution output")

        self.fc1 = nn.Linear(self.convout_size, l1_out)
        # self.bn3 = nn.BatchNorm1d(l1_out)
        # self.do3 = nn.Dropout(self.dp)
        self.fc2 = nn.Linear(l1_out, l2_out)
        # self.bn4 = nn.BatchNorm1d(l2_out)
        # self.do4 = nn.Dropout(self.dp)
        self.fc3 = nn.Linear(l2_out, out_size)

    def forward(self, x):
        # print(type(x))
        # Convolutions + Pooling
        # print(x.shape, " x")
        c1 = F.relu(self.conv1(x))
        # print(c1.shape, " c1")
        p1 = F.max_pool2d(c1, self.pool_size)
        # print(p1.shape, " p1")
        c2 = F.relu(self.conv2(p1))
        # print(c2.shape, " c2")
        # bn1 = self.bn1(c2)
        # do1 = self.do1(bn1)
        p2 = F.max_pool2d(c2, self.pool_size)
        # print(p2.shape, " p2")

        c3 = F.relu(self.conv3(p2))
        # print(c3.shape, " c3")
        # bn2 = self.bn2(c3)
        # do2 = self.do1(bn2)
        p3 = F.max_pool2d(c3, self.pool_size)
        # print(p3.shape, " p3")

        # Fully Connected
        flat = p3.view(-1, self.convout_size)
        # print(flat.shape, " flat")
        f1 = F.relu(self.fc1(flat))
        # print(f1.shape, " f1")l3
        # bn3 = self.bn3(f1)
        # do3 = self.do3(bn3)

        f2 = F.relu(self.fc2(f1))
        # bn4 = self.bn4(f2)
        # do4 = self.do4(bn4)
        f3 = self.fc3(f2)

        return f3

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save_model_weights(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model_weights(self, model_name):
        self.load_state_dict(torch.load(model_name))


class Scanner():

    def __init__(self, image_dir, model_path, patch_size, output_dir, norm_factor):

        self.norm_factor = norm_factor
        self.image_dir= image_dir
        # model = nn.Module.ConvNet()
        self.model = torch.load(model_path,map_location="cpu")
        # self.model = torch.load_state_dict(torch.load(model_path), map_location="cpu")
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
    # parser.add_argument("patch_size", help="Size of model patch size")
    # parser.add_argument("images_dir", help="Directory containing images to be processed")
    parser.add_argument("--model_dir", help="Path to directory containing trained model. Chooses the most recent by default")
    # parser.add_argument("output_dir", help="Output directory of scanned image maps")
    # args = parser.parse_args()

    parser.add_argument("config_file", help="Path to config file for pipeline")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.sections()
    config.read(args.config_file)

    image_dir = config['DEFAULT']['image_directory']

    training_sessions_dir = os.path.join(config['DEFAULT']['output_directory'], "training_sessions")

    output_dir = max(glob.glob(os.path.join(training_sessions_dir, '*/')), key=os.path.getmtime)

    most_recent_model = os.path.join(output_dir, "model.pb")

    if hasattr(args, "model_dir"):
        if args.model_dir != None:
            if os.path.isdir(args.model_dir):
                most_recent_model = args.model_dir
                output_dir = args.model_dir
            else:
                print("Model directory is not a directory")

    # python scanner.py 40 C:\Users\Saideep\Documents\Github_Repos\MSCB_Sem1\Deep_Learning\Project\Labeled_Spines_Tavita\ C:\Users\Saideep\Documents\Github_Repos\MSCB_Sem1\Deep_Learning\Project\counting_spines\src\training_sessions\2019-04-1515_00_29\weights.pt C:\Users\Saideep\Documents\Github_Repos\MSCB_Sem1\Deep_Learning\Project\counting_spines\src\training_sessions\2019-04-1515_00_29\
    # Create scanner object
    print("MOST RECENT")
    print(most_recent_model)
    # scanner = Scanner(Path(args.images_dir), Path(args.model_path), int(args.patch_size), Path(args.output_dir), norm_factor)
    scanner = Scanner(Path(image_dir), Path(most_recent_model), int(config['DEFAULT']['patch_dim']), Path(output_dir), int(config['DEFAULT']['norm_factor']))
    figdir = os.path.join(output_dir, "prediction_figures")
    if os.path.isdir(figdir) is False:
        os.mkdir(figdir)
    scanner.scan_all_images(figdir)
