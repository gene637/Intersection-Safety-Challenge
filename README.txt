Edit each path in the file as your own data location and mmdetection3d data location

#### 1. remove data to isc_full
ISC/isc_rename_bin_moveto_isc_dataset.py
Edit line 25 and 92 (train/test)

#### 2. generate imagesets (split train and val randomly)
ISC/isc_imagesets_generation.py
No Edit.

#### 3. rename calib files according to imagesets
ISC/isc_rename_calib.py
No Edit.

#### 4. generate kitti pickles
python tools/create_data.py kitti --root-path ./data/isc_full --out-dir ./data/isc_full --extra-tag isc_full_lidar12

#### 5. train
python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class.py --work-dir work_dirs/xxx

#### 6. test
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class.py work_dirs/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class/latest.pth --work-dir work_dirs/xxx

#### 7. continue training from last pth
python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class.py --work-dir work_dirs/lidar12 --resume work_dirs/pointpillars_hv_secfpn_8xb6-160e_isc-3d-3class/epoch_100_lidar12.pth