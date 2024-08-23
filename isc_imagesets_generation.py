import random
import os

def split_text_file(input_file, output_file1, output_file2, split_ratio=0.8):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 计算拆分的位置
    split_index = int(len(lines) * split_ratio)

    # 随机打乱列表
    random.shuffle(lines)

    # 分割成两部分
    part1 = lines[:split_index]
    part2 = lines[split_index:]

    # 写入第一个输出文件
    with open(output_file1, 'w') as f1:
        f1.writelines(part1)

    # 写入第二个输出文件
    with open(output_file2, 'w') as f2:
        f2.writelines(part2)


def list_files_to_txt(folder_path, output_file):
    with open(output_file, 'w') as f:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                f.write(filename[:-4]+'\n')

# 文件夹路径
folder_path_train = '/home/gene/mmdetection3d/data/isc_full/training/velodyne/'
folder_path_test = '/home/gene/mmdetection3d/data/isc_full/testing/velodyne/'
# 输出文件路径
output_file_train = '/home/gene/mmdetection3d/data/isc_full/ImageSets/trainval.txt'
output_file_test = '/home/gene/mmdetection3d/data/isc_full/ImageSets/test.txt'

# 调用函数，将文件夹下文件名写入文本文件
list_files_to_txt(folder_path_train, output_file_train)
list_files_to_txt(folder_path_test, output_file_test)

print(f"已将文件夹 {folder_path_test} 下的文件列表写入到 {output_file_test}")

# 输入文件和输出文件路径
input_file = '/home/gene/mmdetection3d/data/isc_full/ImageSets/trainval.txt'
output_file1 = '/home/gene/mmdetection3d/data/isc_full/ImageSets/train.txt'
output_file2 = '/home/gene/mmdetection3d/data/isc_full/ImageSets/val.txt'
# 调用函数，将输入文件内容随机拆分为两个输出文件
split_text_file(input_file, output_file1, output_file2)

print(f"已将文件 {input_file} 中的内容随机拆分并保存到 {output_file1} 和 {output_file2}")
