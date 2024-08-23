import os
import shutil
import random


def find_different_extension_file(folder, file_path):
    # Get the file name and extension of the given file
    base_name = os.path.basename(file_path)  # Get the file name part
    file_name, file_ext = os.path.splitext(base_name)  # Separate the file name and extension
    
    # Iterate over files in the folder
    for filename in os.listdir(folder):
        if filename.startswith(file_name) and filename != base_name:
            # Find a file with the same name but different extension
            _, ext = os.path.splitext(filename)
            if ext != file_ext:
                return True, filename  # Return True indicating a matching file exists
    return False, None  # No matching file found


def save_training_data(loc, isctrainloc, isclabelloc):
    # Get a list of all items in the directory
    sub_srcs = os.listdir(loc)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(loc, sub_src))]
    for sub_dir in sub_dirs:

        path = loc + '/' + sub_dir + '/Lidar12_bin_reduced/strongest'
        pathgt = loc + '/' + sub_dir + '/Kitti_GT/'
        sub_items = os.listdir(path)
        sub_dir = sub_dir.replace('Run_', '')
        sub_dir = sub_dir.replace('?', '')
        i = 0
        for item in sub_items:
            item_path = os.path.join(path, item)
            _, gt_item = find_different_extension_file(pathgt, item_path)
            if _:
                # i += 1
                # array = list(range(1, len(sub_items) - 5))
                # random.shuffle(array)
                # split_index = int(len(array) * 0.2)
                # first_part = array[:split_index]
                # if i in first_part:
                #     # Original file name and new file name (if renaming within the same directory, you can set target path to current path)
                #     old_name = item_path
                #     new_name = isctestloc + sub_dir + item
                # else:
                
                shutil.copy(pathgt + gt_item, isclabelloc + sub_dir + gt_item)
                # Use shutil.copy() to copy the file
                old_name = item_path
                new_name = isctrainloc + sub_dir + item
                shutil.copy(old_name, new_name)


# Testing data

def save_test_data(pathtest, isctestloc, dir):
    sub_items = os.listdir(pathtest)
    i = 0
    for item in sub_items:
        item_path = os.path.join(pathtest, item)
        # i += 1
        # array = list(range(1, len(sub_items) - 5))
        # random.shuffle(array)
        # split_index = int(len(array) * 0.3)
        # first_part = array[:split_index]
        # if i in first_part:
        #     # Original file name and new file name (if renaming within the same directory, you can set target path to current path)
        old_name = item_path
        new_name = isctestloc + dir + item
        shutil.copy(old_name, new_name)


if __name__ == '__main__':

    isctrainloc = '/home/gene/mmdetection3d/data/isc_full/training/velodyne/'
    isctestloc = '/home/gene/mmdetection3d/data/isc_full/testing/velodyne/'
    isclabelloc = '/home/gene/mmdetection3d/data/isc_full/training/label_2/'
    loc = '/home/gene/Documents/Validation Data2'
    pathtest_1 = '/home/gene/Documents/Training Data/?Run_465/Lidar2_bin/strongest'
    dir_1 = '465'
    pathtest_2 = '/home/gene/Documents/Training Data/Run_3/Lidar2_bin/strongest'
    dir_2 = '3'
    pathtest_3 = '/home/gene/Documents/Training Data/Run_26/Lidar2_bin/strongest'
    dir_3 = '26'
    pathtest_4 = '/home/gene/Documents/Training Data/Run_48/Lidar2_bin/strongest'
    dir_4 = '48'

    test = 0

    if test == 1:
        save_test_data(pathtest_1, isctestloc, dir_1)
        save_test_data(pathtest_2, isctestloc, dir_2)
        save_test_data(pathtest_3, isctestloc, dir_3)
        save_test_data(pathtest_4, isctestloc, dir_4)
    else:
        save_training_data(loc, isctrainloc, isclabelloc)
