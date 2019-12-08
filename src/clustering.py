"""
Authors: Kwanho Kim, Saideep Gona, Jinke Liu

Contains code for doing clustering on scanned output. 
"""

import os, sys, argparse, pathlib

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clustering algorithm for  10-707(Deep Learning) project")
    parser.add_argument("input_dir", help="Directory containing input(output of scanning step)")
    parser.add_argument("source_dir", help="Directory of original source images and the true annotations")
    args = parser.parse_args()

    # Create scanner object

    scanner = Scanner(args.image_dir, state_dict_path)