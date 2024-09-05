# This script is for generating final 3d detections after getting the results from 2D images
# How to get results from 2d: 3d point cloud (merged and filtered point cloud) -> project to 2d image -> cluster points -> get centroids -> project to lidar2 coordinates with vehicle/VRU class and confidence
import numpy as np
import pandas as pd
import torch

def limit_period(val,
                 offset: float = 0.5,
                 period: float = np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val

def find_subclass(loc2d, loc3d, search_period, search_radius, name_to_bbox_size):
    content_value_2d = np.genfromtxt(loc2d, delimiter=',', usecols=range(2,8))
    content_str_2d = np.genfromtxt(loc2d, delimiter=',', dtype='|U', usecols=range(0,2))
    frames_2d = content_str_2d[:,0]
    frames_2d = np.char.replace(frames_2d, '.pid', '')
    classes_2d = content_str_2d[:,1]
    classes_2d = np.char.replace(classes_2d, ' ', '')
    centers_2d = content_value_2d[:,0:3]
    confidence_scores = content_value_2d[:,3]
    raws = content_value_2d[:,5]

    content_value_3d = np.genfromtxt(loc3d, delimiter=',', usecols=range(2,10))
    content_str_3d = np.genfromtxt(loc3d, delimiter=',', dtype='|U', usecols=range(0,2))
    frames_3d = content_str_3d[:,0]
    frames_3d = np.char.replace(frames_3d, '.pcd', '')
    classes_3d = content_str_3d[:,1]
    classes_3d = np.char.replace(classes_3d, ' ', '')
    centers_3d = content_value_3d[:,[0,2,4]]

    detections = []

    for i in range(int(frames_2d[0]), int(frames_2d[-1])+1):
        print(i)
        if i>=int(frames_2d[0])+search_period and i<=int(frames_2d[-1])+1-search_period:
            indices_2d = np.where(frames_2d == str(i).zfill(6))[0]
            indices_3d = np.array([])
            for ii in range(i-search_period, i+search_period+1):
                frame = str(ii).zfill(6)
                indices_3d_tmp = np.where(frames_3d == frame)[0]
                indices_3d = np.append(indices_3d, indices_3d_tmp).astype(int)
        else:
            frame = str(i).zfill(6)
            indices_2d = np.where(frames_2d == frame)[0]
            indices_3d = np.where(frames_3d == frame)[0]
        for j in indices_2d:
            # print(classes_2d[j])
            if classes_2d[j] == 'car' or classes_2d[j] == 'truck':
                subclass = 'Passenger_Vehicle'
            elif classes_2d[j] == 'bicycle':
                subclass = 'VRU_Adult_Using_Manual_Bicycle'
            else:
                within_range_indices_3d = calcualte_near_objects(centers_2d, centers_3d, j, indices_3d, search_radius)
                if len(within_range_indices_3d) == 0:
                    indices_3d2 = np.arange(len(frames_3d))
                    within_range_indices_3d2 = calcualte_near_objects(centers_2d, centers_3d, j, indices_3d2, search_radius = 30)
                    print(within_range_indices_3d2)
                    series = pd.Series(classes_3d[within_range_indices_3d2])
                    print(series)
                    frequency = series.value_counts()
                    print(frequency)
                    tmp = 0
                    while frequency.index[tmp] == 'VRU_Adult_Using_Manual_Bicycle' or frequency.index[tmp] == 'Passenger_Vehicle':
                        tmp += 1
                    subclass = frequency.index[tmp]
                else:
                    # filter the most frequent one
                    # print(within_range_indices_3d)
                    series = pd.Series(classes_3d[within_range_indices_3d])
                    # print(series)
                    frequency = series.value_counts()
                    # print(frequency)
                    highest_frequency = frequency.iloc[0]
                    most_frequent_classes = frequency[frequency == highest_frequency].index
                    # print(most_frequent_classes)
                    most_frequent_class = most_frequent_classes[0]
                    # if there are two highest frequency classes
                    if len(most_frequent_classes)>1:
                        most_frequent_class = most_frequent_classes[0]
                    # if class is bicycle or vehicle
                    if len(frequency) > 1:
                        if most_frequent_class=='VRU_Adult_Using_Manual_Bicycle' or most_frequent_class=='Passenger_Vehicle':
                            most_frequent_class = frequency.index[1]
                    subclass = most_frequent_class
            size = name_to_bbox_size[subclass]
            center = centers_2d[j]
            detection = [frames_2d[j], subclass, center[0], size[0], center[1], size[1], center[2], size[2], raws[j]]
            detections.append(detection)
    return detections

def calcualte_near_objects(centers_2d, centers_3d, index_2d, indices_3d, search_radius):    
    # print(indices_3d)
    points = centers_3d[indices_3d]
    reference_point = centers_2d[index_2d]
    # calculate euc dis
    distances = np.linalg.norm(points - reference_point, axis=1)
    # filter the centers within range
    within_range_indices_3d = indices_3d[distances < search_radius]

    return within_range_indices_3d

def main():
    # open detections from 2D
    loc2d = '/home/gene/Downloads/fused_label_lidar12_cam1_5/Run_48/fusion_table/fusion_label/Run_48_fused_result.txt'
    loc3d = '/home/gene/Documents/Validation Data2/Run_48/detections/Run_48_detections_axis_aligned_lidar2.txt'

    name_to_bbox_size = {
        'VRU_Adult_Using_Motorized_Bicycle': [0.96, 1.64, 1.65],
        'Passenger_Vehicle': [2.60, 5.08, 1.85],
        'VRU_Child': [0.68, 0.65, 1.08],
        'VRU_Adult': [0.83, 0.75, 1.59],
        'VRU_Adult_Using_Cane': [0.75, 0.69, 1.45],
        'VRU_Adult_Using_Manual_Scooter': [1.13, 1.23, 1.76],
        'VRU_Adult_Using_Crutches': [1.14, 1.09, 2.08],
        'VRU_Adult_Using_Cardboard_Box': [1.11, 1.15, 1.68],
        'VRU_Adult_Using_Walker': [0.99, 1.12, 1.58],
        'VRU_Adult_Using_Manual_Wheelchair': [1.08, 1.20, 1.28],
        'VRU_Adult_Using_Stroller': [0.97, 1.56, 1.68],
        'VRU_Adult_Using_Skateboard': [1.31, 1.71, 1.40],
        'VRU_Adult_Using_Manual_Bicycle': [0.89, 1.53, 1.38],
        }
    
    # search range of frames
    search_range =  5 #odd number only
    search_period = int((search_range-1)/2)

    # search radius of object location from 2d
    search_radius = 1.0

    detections = find_subclass(loc2d, loc3d, search_period, search_radius, name_to_bbox_size)
    print(detections)
    


if __name__ == "__main__":
    main()