# -*- coding: utf-8 -*-
"""
Created on 2024-12-4
Author: Yuejiahe
Description: This script is used to generate text data for the dataset.
"""

import os

def generate_file_list(root_dir, output_file):
    root_dir_name = os.path.basename(root_dir)
    with open(output_file, 'w') as f:
        for subdir, _, files in os.walk(root_dir):
            if files:
                relative_path = os.path.relpath(subdir, root_dir)
                full_path = os.path.join(root_dir_name, relative_path)
                file_list = ' '.join(sorted(files))
                f.write(f"{full_path}/ {file_list}\n")

if __name__ == "__main__":
    root_directory = '/media/yjh/yjh/LightReBlur/VDV/test/SharpImages'  # Replace with your root directory
    output_txt = 'data_vdv/val.txt'
    generate_file_list(root_directory, output_txt)