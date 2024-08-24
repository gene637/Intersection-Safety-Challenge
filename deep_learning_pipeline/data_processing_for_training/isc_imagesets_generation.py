import random
import os

def split_text_file(input_file, output_file1, output_file2, split_ratio=0.8):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Calculate the split index
    split_index = int(len(lines) * split_ratio)

    # Shuffle the list randomly
    random.shuffle(lines)

    # Split into two parts
    part1 = lines[:split_index]
    part2 = lines[split_index:]

    # Write to the first output file
    with open(output_file1, 'w') as f1:
        f1.writelines(part1)

    # Write to the second output file
    with open(output_file2, 'w') as f2:
        f2.writelines(part2)


def list_files_to_txt(folder_path, output_file):
    with open(output_file, 'w') as f:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                f.write(filename[:-4] + '\n')

# Folder paths
folder_path_train = '/home/gene/mmdetection3d/data/isc_full/training/velodyne/'
folder_path_test = '/home/gene/mmdetection3d/data/isc_full/testing/velodyne/'
# Output file paths
output_file_train = '/home/gene/mmdetection3d/data/isc_full/ImageSets/trainval.txt'
output_file_test = '/home/gene/mmdetection3d/data/isc_full/ImageSets/test.txt'

# Call the function to write filenames from the folder to text files
list_files_to_txt(folder_path_train, output_file_train)
list_files_to_txt(folder_path_test, output_file_test)

print(f"File list from folder {folder_path_test} has been written to {output_file_test}")

# Input file and output file paths
input_file = '/home/gene/mmdetection3d/data/isc_full/ImageSets/trainval.txt'
output_file1 = '/home/gene/mmdetection3d/data/isc_full/ImageSets/train.txt'
output_file2 = '/home/gene/mmdetection3d/data/isc_full/ImageSets/val.txt'
# Call the function to randomly split the content of the input file into two output files
split_text_file(input_file, output_file1, output_file2)

print(f"Content of file {input_file} has been randomly split and saved to {output_file1} and {output_file2}")
