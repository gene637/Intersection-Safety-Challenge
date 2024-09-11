# This script is for generating final 3d detections after getting the results from 2D images
# How to get results from 2d: 3d point cloud (merged and filtered point cloud) -> project to 2d image -> cluster points -> get centroids -> project to lidar2 coordinates with vehicle/VRU class and confidence
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
import pytz

def cal_timestamp(loc):
    # Open and read the CSV file
    with open(loc, mode='r', newline='') as file:
        reader = csv.reader(file)
        
        # Loop through each row in the CSV file
        for row in reader:
            if 'Lidar2' in row:
                start_time = row[4]  # Start time

    # Time string (Eastern Time)
    time_str = start_time

    # Define the Eastern Time zone
    eastern = pytz.timezone('America/New_York')

    # Convert the string to a datetime object (local time, without timezone info)
    dt_local = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")

    # Localize the datetime object to Eastern Time
    dt_eastern = eastern.localize(dt_local)

    # Convert Eastern Time to UTC
    dt_utc = dt_eastern.astimezone(pytz.utc)

    # Convert the UTC datetime object to a Unix timestamp (seconds since epoch)
    timestamp = dt_utc.timestamp()

    return timestamp

def find_subclass(timestamp_start, loc2d, loc3d, search_period, search_radius, name_to_bbox_size):
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
                    within_range_indices_3d2 = calcualte_near_objects(centers_2d, centers_3d, j, indices_3d2, search_radius = 100) #enlarge the search_radius if no matching objects detected
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
            timestamp = float(timestamp_start) + float(frames_2d[j])*0.1
            detection = [timestamp, subclass, center[0], size[0], center[1], size[1], center[2], size[2], raws[j]]
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

    # open detections from 2D
    # Get a list of all items in the directory
    loc = './sample_detections_validation/' #replace it to your own folder
    res2d_loc = './fused_label_lidar12_cam1_5/' #replace it to your own folder
    sub_srcs = os.listdir(res2d_loc)
    # Filter out only directories
    sub_dirs = [sub_src for sub_src in sub_srcs if os.path.isdir(os.path.join(res2d_loc, sub_src))]
    for sub_dir in sub_dirs:
        #input
        loc2d = res2d_loc+sub_dir+'/fusion_table/fusion_label/'+sub_dir+'_fused_result.txt'
        if not os.path.exists(loc2d):
            continue
        loc3d = loc+sub_dir+'_detections_axis_aligned_lidar2.txt'
        timestamp_loc = '/home/gene/Documents/Validation Data2/'+sub_dir+'/ISC_'+sub_dir+'_ISC_all_timing.csv' #replace it to your own folder with timestamp.csv
        timestamp_start = cal_timestamp(timestamp_loc)

        #output
        file_name = res2d_loc+sub_dir+'/fusion_table/fusion_label/'+sub_dir+'_detections_fusion_lidar12_camera_search-based.csv'
        header = ['Timestamps', 'subclass', 'x_center', 'x_length', 'y_center', 'y_length', 'z_center', 'z_length', 'z_rotation']
        detections = find_subclass(timestamp_start, loc2d, loc3d, search_period, search_radius, name_to_bbox_size)
        # save as CSV
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(detections)
        # save as txt
        # with open(file_name, "w", encoding="utf-8") as file:
        #     for item in detections:
        #         file.write(', '.join(map(str, item)) + '\n')
        # print(detections)
    


if __name__ == "__main__":
    main()