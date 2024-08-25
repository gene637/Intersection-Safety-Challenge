### Folder Structure
```
├── conventional_pipeline
│   ├── clustering_classification_bbox_video_generation.py
│   ├── clustering_labelfeature_extractor.py
│   ├── cut_pcd_range.py
│   ├── dataset_background_filter.py
│   ├── isc2kitti.py
│   ├── label_features_12_all.npy
│   ├── main.py
│   ├── merge_lidar1lidar2_cloud.py
│   ├── __pycache__
│   │   ├── clustering_classification_bbox_video_generation.cpython-38.pyc
│   │   ├── clustering_labelfeature_extractor.cpython-38.pyc
│   │   ├── cut_pcd_range.cpython-38.pyc
│   │   ├── dataset_background_filter.cpython-38.pyc
│   │   ├── isc2kitti.cpython-38.pyc
│   │   └── merge_lidar1lidar2_cloud.cpython-38.pyc
│   └── sample_detections_validation
├── deep_learning_pipeline
│   └── data_processing_for_training
│       ├── 000000.png
│       ├── 000000.txt
│       ├── box_3d_mode.py
│       ├── cut_pcd_range.py
│       ├── isc2kitti.py
│       ├── isc_imagesets_generation.py
│       ├── isc_rename_bin_moveto_isc_dataset.py
│       ├── isc_rename_calib.py
│       ├── isc_video_clipping.py
│       ├── main.py
│       ├── merge_lidar1lidar2_cloud.py
│       ├── pcd2bin.py
│       ├── __pycache__
│       │   ├── cut_pcd_range.cpython-38.pyc
│       │   ├── isc2kitti.cpython-38.pyc
│       │   ├── isc_imagesets_generation.cpython-38.pyc
│       │   ├── isc_rename_bin_moveto_isc_dataset.cpython-38.pyc
│       │   ├── isc_rename_calib.cpython-38.pyc
│       │   ├── merge_lidar1lidar2_cloud.cpython-38.pyc
│       │   └── pcd2bin.cpython-38.pyc
│       └── README.md
├── environment.yml
├── LICENSE
├── README.md
└── unwrap_pcap2pcd.py
```

### Environment Installation
If you need to install MMDetection3D, please refer to [MMDetection Doc](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) and environment.yml.

You can install the packages using `conda env create -f environment.yml` to copy my conda environment. 

Regarding the environment.yml, please make sure your GPU aligns with the cuda toolkit version. When you meet some issues about cuda, it may be because the cuda toolkit in conda environment is incomplete and you may need to setup the complete cuda in the base environment as a supplement with the same version in the conda environment. 

### *.pcap Unwrapping
Before detection using deep learning pipeline or conventional pipeline, please unwrap the *.pcap files first. With [PCD-TOOL](https://github.com/NEWSLabNTU/pcd-tool), we can tranform the pcap files to pcd/bin files quickly. After you install the pcd-tool, please move the `unwrap_pcap2pcd.py` under the pcd-tool folder and edit the `loc` as the parent folder of pcap files.

### Training and Testing
The two following both use the cut pcd range and merged point cloud with lidar1 and lidar2.

#### Conventional Pipeline
The conventional one utilized background filtering (Difference Comparison), clustering (DBSCAN), classification (Random Forest) and bbox generation to get the detections.
##### Step:
Under conventional_pipeline folder, find `main.py`.

```Command: python main.py <path>```

Note: Please provide the folder location of testing data with '\\' before blank and '/' in the end.

1. Get the classifier (Validation/Training Data with labels): 
```isc2kitti->cut_pcd_range->merge_lidar1_and_lidar2->filter_background->get_training_feature_and_label```
We have gathered the `label_feature_12_all.npy` here, so you do not need to process step by step.
2. Test the data, get the detections (Testing Data without labels):
If you do not have the Lidar12_pcd_filtered, you need to process the Lidar1_pcd and Lidar2_pcd with `cut_pcd_range->merge_lidar1_and_lidar2->filter_background->clustering_classification_bbox_generation`.

##### Results:
Detection format (in lidar2 coordinate): frame, subclass, x_center, x_length, y_center, y_length, z_center, z_length, yaw, score

`Run_*/detections/Run_*_detections_axis_aligned_lidar2.txt`

`Run_*/detections/Run_*_detections_oriented_lidar2.txt`

Also, for tracking, we leave a interface as the return of `clustering_classification_bbox_video_generation` function.

Note: you can finetune the filtering degree with scale, nb_points and radius; finetune the clustering degree with eps, min_points; save BEV video with `video = True`.

#### Deep Learning Pipeline
The deep learning pipeline is developed under MMDetection3D framework and Kitti format, using pointpillars detector. To run this pipeline, please first install the MMDetection3D environment with correponding version packages.
##### Step:
Under deep_learning_pipeline folder, use code under data_processing_for_training to process the ISC data to Kitti format.
Under data_processing_for_training folder, find `main.py`.

```Command: python main.py <src_path> <target_path>```

Note: Please provide the folder location of source testing data and target testing data with '\\' before blank and '/' in the end.
I.e., `/home/gene/Documents/Validation\\ Data2/ /home/gene/mmdetection3d/data/isc_full/`

Steps:
```generate kitti label->cut pcd range->merge lidar1 and lidar2->pcd2bin for training in Kitti format->remove data to isc_full->generate imagesets (split train and val randomly)->rename calib files according to imagesets```

After generating the files for training in MMDet3D, please refer to command in README.md under data_processing_for_training to train the model.
