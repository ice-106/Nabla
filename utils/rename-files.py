import argparse
import os
import shutil

def parseargs():
    parser = argparse.ArgumentParser(
        description="Rename output to original filenames"
    )
    parser.add_argument(
        '--original',
        type=str,
        required=True,
        help='Path to original data folder'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Path to target data to rename folder'
    )
    parser.add_argument(
        '--smplx',
        type=bool,
        help='smplerx model'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()
    original_path = args.original
    target_path = args.target
    # Walk through the target directory structure
    if args.smplx:
        for root, dirs, files in os.walk(target_path):
            npz_files = sorted([f for f in files if f.endswith('.npz')])
            if npz_files:
                rel_path = os.path.relpath(root, target_path)
                # Remove 'smplx' from the relative path to find original directory
                rel_path_parts = rel_path.split(os.sep)
                if 'smplx' in rel_path_parts:
                    rel_path_parts.remove('smplx')
                original_rel_path = os.sep.join(rel_path_parts) if rel_path_parts else ''
                original_dir = os.path.join(original_path, original_rel_path)
                
                if os.path.exists(original_dir):
                    png_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.png')])
                    
                    for i, npz_file in enumerate(npz_files):
                        if i < len(png_files):
                            original_name = os.path.splitext(png_files[i])[0]
                            new_filename = original_name + '.npz'
                            
                            old_path = os.path.join(root, npz_file)
                            new_path = os.path.join(root, new_filename)
                            
                            os.rename(old_path, new_path)
                            print(f"Renamed: {old_path} -> {new_path}")
                        else:
                            print(f"Warning: No corresponding PNG file for {npz_file} in {original_dir}")
    else:
        for root, dirs, files in os.walk(target_path):
            # Get all pkl files and sort them
            pkl_files = sorted([f for f in files if f.endswith('.pkl')])
            
            if pkl_files:
                # Get the relative path from target_path
                rel_path = os.path.relpath(root, target_path)
                
                # Find corresponding original directory
                original_dir = os.path.join(original_path, rel_path)
                
                if os.path.exists(original_dir):
                    # Get all png files in the original directory and sort them
                    png_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.png')])
                    
                    # Match pkl files with png files by index
                    for i, pkl_file in enumerate(pkl_files):
                        if i < len(png_files):
                            # Get the base name without extension from the corresponding .png file
                            original_name = os.path.splitext(png_files[i])[0]
                            
                            # Extract the suffix from the pkl file (everything after the first underscore)
                            pkl_parts = pkl_file.split('_', 1)
                            if len(pkl_parts) > 1:
                                suffix = '_' + pkl_parts[1]
                            else:
                                suffix = '.pkl'
                            
                            # Create new filename
                            new_filename = original_name + suffix
                            
                            # Rename the file
                            old_path = os.path.join(root, pkl_file)
                            new_path = os.path.join(root, new_filename)
                            
                            os.rename(old_path, new_path)
                            print(f"Renamed: {old_path} -> {new_path}")
                        else:
                            print(f"Warning: No corresponding PNG file for {pkl_file} in {original_dir}")
