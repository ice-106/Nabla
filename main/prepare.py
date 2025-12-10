import argparse
import os
import os.path as osp
import sys
import zipfile
from pathlib import Path

import gdown

from configs.extraction.prepare import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos_url", type=str,
                        help="URL of the videos to download.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent.parent

    # Check if destination directory exists, create if not
    destination_folder = config['destination_folder']
    if not osp.exists(destination_folder):
        print(f"Creating destination folder: {destination_folder}")
        os.makedirs(destination_folder, exist_ok=True)

    # Download videos using the URL from command line argument
    gdown.download(args.videos_url, config["filename"], fuzzy=True)
    val_zip_path = config["filename"]
    if not osp.exists(val_zip_path):
        print(f"✗ File not found: {val_zip_path}")
        return
    if os.system(f"unzip -q {val_zip_path} -d {config['destination_folder']}") == 0:
        print(f"✓ Successfully extracted to: {config['destination_folder']}")
        os.system(f"rm {val_zip_path}")


if __name__ == "__main__":
    main()
