```commandline
python3 spatial_mapping.py -h
usage: spatial_mapping.py [-h] [--input_svo_file INPUT_SVO_FILE] [--ip_address IP_ADDRESS] [--resolution RESOLUTION] [--build_mesh]

optional arguments:
  -h, --help            show this help message and exit
  --input_svo_file INPUT_SVO_FILE
                        Path to an .svo file, if you want to replay it
  --ip_address IP_ADDRESS
                        IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup
  --resolution RESOLUTION
                        Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
  --build_mesh          Either the script should plot a mesh or point clouds of surroundings
```


--build_mesh を指定したときに得られるファイル。
```commandline
mesh_gen.obj
mesh_gen_material0000_map_Kd.png
mesh_gen.mtl
```

mtl: マテリアル テンプレート ライブラリ
mesh_gen_material0000_map_Kd.png: メッシュと対応付けられた範囲のRGBデータ。（３チャネル）

--build_mesh を指定しないときに得られるファイル。

mesh_gen.obj
点群だけが得られる。

## 撮影状況の改善手法
- meshを生成する刻みを小さくする。
  - SpatialMappingParameters.resolution_meter [m]
- meshを生成する空間範囲を狭める。
MAPPING_RANGE::NEAR: integrates depth up to 3.5 meters.
MAPPING_RANGE::MEDIUM: integrates depth up to 5 meters.
MAPPING_RANGE::FAR: integrates depth up to 10 meters.
- メモリの使用量を増やす。
  - SpatialMappingParameters.max_memory_usage
- 点を返すための条件を緩める。
  - confidenceの値を大きくすると点を返しやすくなる。
  -  runtime_parameters.confidence_threshold = 50


## 
normals: 法線ベクトル

## Q: obj ファイルで座標の原点はどうなっているか？
Meshlab での表示との関係がわからない。

## 影響する項目
enabl_fill_mode: Falseの場合、欠損値を生じる。Trueにすると欠損値を生じないが、物体の存在しない場所に点を生じる。
removed_saturated_area: Trueにすると白飛びした画素が欠損値となる。例：ペットボトルの白いキャップが未検出になる。

#### カメラのゲインの設定

## depth計算の負荷を減らすには
- これらのメソッドを利用して計算負荷を減らすこと
- 以下のサイトの情報を参考にすること。
https://www.stereolabs.com/docs/ros2/region-of-interest

```commandline
Camera.get_region_of_interest(self, Mat py_mat: Mat, resolution=Resolution(0, 0), module=MODULE.ALL) -> ERROR_CODE
Camera.get_region_of_interest_auto_detection_status(self) -> REGION_OF_INTEREST_AUTO_DETECTION_STATE
Camera.set_region_of_interest(self, Mat py_mat: Mat, modules=[MODULE.ALL]) -> ERROR_CODE
Camera.start_region_of_interest_auto_detection(self, roi_param=RegionOfInterestParameters()) -> ERROR_CODE
```
