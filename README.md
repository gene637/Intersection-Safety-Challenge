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
Under conventional_methods folder, find `main.py`.
```Command: python main.py <path>```
Note: Please provide the folder location of testing data with '\\' before blank and '/' in the end.
##### Results:
`Run_\*/detections/Run_\*_detections_axis_aligned_lidar2.txt`
`Run_\*/detections/Run_\*_detections_oriented_lidar2.txt`

#### Deep Learning Pipeline
The deep learning pipeline is developed under MMDetection3D framework and Kitti format, using pointpillars detector. To run this pipeline, please first install the MMDetection3D environment with correponding version packages. 
