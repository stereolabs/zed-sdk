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
