def bbox_to_xyxy(bounding_box_2d):
    xlist = [x for x, _ in bounding_box_2d]
    ylist = [y for _, y in bounding_box_2d]
    xmin = min(xlist)
    xmax = max(xlist)

    ymin = min(ylist)
    ymax = max(ylist)

    return ((xmin, ymin), (xmax, ymax))

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
