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
    parser.add_argument("videos_url", type=str, help="URL of the videos to download.")
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
    if os.system(f"unzip {val_zip_path} -d {config['destination_folder']}") == 0:
        print(f"✓ Successfully extracted to: {config['destination_folder']}")  
        os.system(f"rm {val_zip_path}")

        






# Example usage:
if __name__ == "__main__":
    # gdown.download('https://drive.google.com/file/d/1hEocwdsGE9nhrym6P-9uf0rOvXAwATbC/view?usp=sharing', 'val_videos.zip', fuzzy=True)
    # val_videos = config["video_url"]
    # destination_folder = config["destination_folder"]
    # print(f"Video URL: {val_videos}")
    # print(f"Destination Folder: {destination_folder}")

    main()
    
    # file_id = extract_file_id(val_videos)
    # print(f"Extracted file ID: {file_id}")

    # downloaded_file = download_from_gdrive(
    #     file_id=file_id,
    #     output_path=destination_folder,
    #     filename=config["filename"]
    # )

    # if downloaded_file:
    #     print(f"Downloaded file path: {downloaded_file}")

    #     # Extract the zip file
    #     if downloaded_file.endswith('.zip'):
    #         extract_to = os.path.join(destination_folder, 'extracted')
    #         os.makedirs(extract_to, exist_ok=True)

    #         try:
    #             with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
    #                 zip_ref.extractall(extract_to)
    #             print(f"✓ Successfully extracted to: {extract_to}")
    #         except Exception as e:
    #             print(f"✗ Error extracting file: {str(e)}")


