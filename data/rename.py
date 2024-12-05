"""
Created on 2024-12-4
Author: Yuejiahe
Description: This script is used to rename the files in the dataset.
"""
import os
import re

# Rename files in a directory
def rename_files_in_directory(root_dir, ext='.flo'):
    for subdir, _, files in os.walk(root_dir):
        jpg_files = [f for f in files if f.endswith(ext)]
        if jpg_files:
            # Extract the numeric part from the filenames and sort them
            sorted_files = sorted(jpg_files, key=lambda x: int(re.search(r'\d+', x).group()))
            for file in sorted_files:
                old_path = os.path.join(subdir, file)
                # Extract the numeric part and format it to 5 digits
                num = int(re.search(r'\d+', file).group())
                new_filename = f"{num:05d}"+ext
                new_path = os.path.join(subdir, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed {old_path} to {new_path}")

if __name__ == "__main__":
    root_directory = '/media/yjh/yjh/LightReBlur/VDV/test/BackwardFlows_NewCT'  # 替换为你的根目录
    rename_files_in_directory(root_directory,ext = '.flo')