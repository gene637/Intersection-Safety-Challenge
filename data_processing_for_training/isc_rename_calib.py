import os
import shutil

def copy_and_rename_file(input, original_file, output_directory):
    # Get the original file name and extension
    base_name, extension = os.path.splitext(original_file)
    
    # Build the output file path
    output_file = os.path.join(output_directory, os.path.basename(base_name) + extension)
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Clear sub-files of the previous dataset
    clear_directory(output_directory)

    # Copy and rename the file
    with open(input, 'r') as f:
        lines = f.readlines()
    for i in lines:
        i = i.replace('\n', '')
        output_file = os.path.join(output_directory, i + extension)
        # Perform the copy operation
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



# Example usage
if __name__ == "__main__":
    input_file_train = '/home/gene/mmdetection3d/data/isc_full/ImageSets/trainval.txt'
    input_file_test = '/home/gene/mmdetection3d/data/isc_full/ImageSets/test.txt'
    original_file = '/home/gene/Documents/000000.txt'  # Original file name
    output_directory1 = '/home/gene/mmdetection3d/data/isc_full/training/calib/'   # Output directory
    output_directory2 = '/home/gene/mmdetection3d/data/isc_full/testing/calib/'   # Output directory
    
    copied_file = copy_and_rename_file(input_file_train, original_file, output_directory1)
    copied_file = copy_and_rename_file(input_file_test, original_file, output_directory2)

    print(f"File copied successfully: {copied_file}")

    origin_image = '/home/gene/Documents/000000.png'
    output_image1 = '/home/gene/mmdetection3d/data/isc_full/training/image_2/'
    output_image2 = '/home/gene/mmdetection3d/data/isc_full/testing/image_2/'
    copied_file = copy_and_rename_file(input_file_train, origin_image, output_image1)
    copied_file = copy_and_rename_file(input_file_test, origin_image, output_image2)

    print(f"File copied successfully: {copied_file}")
