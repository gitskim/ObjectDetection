import os
import numpy as np

source_root = '/Users/Senpai/git'
key_frame_directory = '/Users/Senpai/git'
file_type = '.mp4'
file_types = ['.mp4', '.MXF', '.mxf']

# list to store the names of the files
mfiles = []

# count how many .mp4, .mxf, .MXF are in the source root - don't count duplicates
file_counter = 0
for root, dirs, files in os.walk(source_root):
    for file in files:
        # for ft in file_types:
        for f_type in file_types:
            if file.endswith(f_type) and file not in mfiles:
                file_counter += 1
                mfiles.append(file)

print(f'Total # of files that end with .mp4, .MXF, .mxf: {len(mfiles)} \n from {mfiles}.')

# count how many .mp4, .MXF, .mxf folders are there under keyframes - don't count duplicates; count how many pngs are there in each folder
dir_counter = 0
mdirs = []
problematic_dir_list=[]
for root, dirs, files in os.walk(source_root):
    for file in files:
        # for ft in file_types:
        for f_type in file_types:
            if root.endswith(f_type):
                dir_name = root.split('/')[-1]
                if dir_name not in mdirs:
                    dir_counter += 1
                    mdirs.append(root.split('/')[-1])
                    if len(files) < 2:
                        problematic_dir_name = os.path.join(root, file)
                        problematic_dir_list.append(problematic_dir_name)
                        print(f'extraction required: {problematic_dir_name}')

print(f'Total # of dirs that end with .mp4, .MXF, .mxf: {len(mdirs)} \n from {mdirs}.')

print('-------Summary-------')
print(f'Total # of files that end with .mp4, .MXF, .mxf: {len(mfiles)} \n from {mfiles}.')
print(f'Total # of dirs that end with .mp4, .MXF, .mxf: {len(mdirs)} \n from {mdirs}.')
