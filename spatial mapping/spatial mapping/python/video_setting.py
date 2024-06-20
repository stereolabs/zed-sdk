"""
inspect.getdoc()　を使ってカメラの設定に関する情報を取得するためのスクリプト
"""
import inspect

import pyzed.sl as sl

zed = sl.Camera()
for k, v in inspect.getmembers(zed):
    if k.find("__") > -1:
        continue
    print("###")
    print(f"{inspect.getdoc(v)}")

    for k2, v2 in inspect.getmembers(v):
        if k2.find("__") > -1:
            continue

        print(f"\t{inspect.getdoc(v2)}")

for k, v in  inspect.getmembers(zed):
    if k.find("region") > -1:
        print(f"{inspect.getdoc(v)}")
