"""
Authors: Kwanho Kim, Saideep Gona, Jinke Liu

Contains code for a scanner object which traveses provided images
and outputs predicted output maps
"""

import os, sys, argparse
from pathlib import Path
from cnn import ConvNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim
import random
import torchvision
import torchvision.transforms as transforms



class Scanner():

    def __init__(self, image_path, state_dict_path, patch_size, output_dir):
        
        model = ConvNet(*args, **kwargs)
        model.load_state_dict(torch.load(state_dict_path))
        model.eval()            # Must be on for running inference

        self.model = model      # Model used for scanning
        self.output_dir = output_dir    # Output directory for processed images

    def pad_image(self, image):

        padded_image = 0

        return padded_image

    def scan_single_image(self, image):
        """
        Scans an input image using the preloaded model and outputs
        a mapping

        :param image: Unpadded image to be processed
        :return: Scanning output map of size pre-padded image
        """

        out_map = np.zeros((image.shape))

        return out_map

    def scan_all_images(self):
        """
        Scans all images in provided directory
        """

        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image scanner for  10-707(Deep Learning) project")
    parser.add_argument("patch_size", help="Size of model patch size")
    parser.add_argument("images_dir", help="Directory containing images to be processed")
    parser.add_argument("state_dict_path", help="Path to model state dictionary")
    parser.add_argument("output_dir", help="Output directory of scanned image maps")
    args = parser.parse_args()

    # Create scanner object

    scanner = Scanner(Path(args.image_dir), Path(args.state_dict_path), Path(args.output_dir))
    scanner.scan_all_images()