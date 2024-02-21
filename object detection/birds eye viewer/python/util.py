def bbox_to_xyxy(bounding_box_2d, as_int=True):
    """
    4点の座標から、左上、右下の座標に変換する。

    Args:
        bounding_box_2d: ４点の座標
        as_int: Trueのとき、int型で返す。

    Returns:　左上、右下の座標

    """
    xlist = [x for x, _ in bounding_box_2d]
    ylist = [y for _, y in bounding_box_2d]
    xmin = min(xlist)
    xmax = max(xlist)

    ymin = min(ylist)
    ymax = max(ylist)

    if as_int:
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

    return ((xmin, ymin), (xmax, ymax))

def bbox_to_xyzxyz(bounding_box_3d, as_int=False):
    """
    4点の座標から、左上、右下の座標に変換する。

    Args:
        bounding_box_2d: ４点の座標
        as_int: Trueのとき、int型で返す。

    Returns:　左上、右下の座標

    """
    xlist = [x for x, _, _ in bounding_box_3d]
    ylist = [y for _, y, _ in bounding_box_3d]
    zlist = [z for _, _, z in bounding_box_3d]
    xmin = min(xlist)
    xmax = max(xlist)

    ymin = min(ylist)
    ymax = max(ylist)

    zmin = min(zlist)
    zmax = max(zlist)

    if as_int:
        xmin, xmax, ymin, ymax, zmin, zmax = int(xmin), int(xmax), int(ymin), int(ymax), int(zmin), int(zmax)

    return ((xmin, ymin, zmin), (xmax, ymax, zmax))


if __name__ == "__main__":
    bounding_box_2d = [[ 553.,  232.],
                       [1275., 232.],
                       [1275., 716.],
                       [ 553., 716.]
                       ]
    point = bbox_to_xyxy(bounding_box_2d)
    print(point)

    bounding_box=[[-0.24210832,  0.20973869, -0.25170186],
                    [-0.23471525, 0.06486525, -1.1031609],
                    [0.6278899, 0.10888501, -1.1031609],
                    [0.62049675, 0.25375843, -0.25170186],
                    [-0.21306634, -0.3593632, -0.15461853],
                    [-0.20567328, -0.50423664, -1.0060775],
                    [0.6569318, -0.46021688, -1.0060775],
                    [0.64953876, -0.31534344, -0.15461853]
                  ]

    box = bbox_to_xyzxyz(bounding_box, as_int=False)
    print(box)
