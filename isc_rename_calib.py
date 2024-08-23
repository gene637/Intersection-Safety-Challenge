import os
import shutil

def copy_and_rename_file(input, original_file, output_directory):
    # 获取原始文件名和扩展名
    base_name, extension = os.path.splitext(original_file)
    
    # 构建输出文件路径
    output_file = os.path.join(output_directory, os.path.basename(base_name) + extension)
    
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # clear sub-files of the previous dataset
    clear_directory(output_directory)

    # 复制并重命名文件
    with open(input, 'r') as f:
        lines = f.readlines()
    for i in lines:
        i = i.replace('\n','')
        output_file = os.path.join(output_directory, i + extension)
        # 执行复制操作
        with open(original_file, 'rb') as fin, open(output_file, 'wb') as fout:
            fout.write(fin.read())
    
    return output_file


def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)



# 示例用法
if __name__ == "__main__":
    input_file_train = '/home/gene/mmdetection3d/data/isc_full/ImageSets/trainval.txt'
    input_file_test = '/home/gene/mmdetection3d/data/isc_full/ImageSets/test.txt'
    original_file = '/home/gene/Documents/000000.txt'  # 原始文本文件名
    output_directory1 = '/home/gene/mmdetection3d/data/isc_full/training/calib/'   # 输出目录
    output_directory2 = '/home/gene/mmdetection3d/data/isc_full/testing/calib/'   # 输出目录
    
    copied_file = copy_and_rename_file(input_file_train, original_file, output_directory1)
    copied_file = copy_and_rename_file(input_file_test, original_file, output_directory2)

    print(f"复制文件成功: {copied_file}")

    origin_image = '/home/gene/Documents/000000.png'
    output_image1 = '/home/gene/mmdetection3d/data/isc_full/training/image_2/'
    output_image2 = '/home/gene/mmdetection3d/data/isc_full/testing/image_2/'
    copied_file = copy_and_rename_file(input_file_train, origin_image, output_image1)
    copied_file = copy_and_rename_file(input_file_test, origin_image, output_image2)

    print(f"复制文件成功: {copied_file}")
    
