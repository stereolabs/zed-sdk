from pathlib import Path
from pprint import pprint

import cv2

import faceme_wrapper

def process_image_dir(img_dir: Path, out_dir: Path):
    """
    img_dir: image source directory
    out_dir: output image directory
    """
    names = list(sorted(img_dir.glob("*.jpg")))  # ascii, Unicode multi byte names
    print(list(names))
    cvimgs = [cv2.imread(str(p)) for p in names]
    t0 = cv2.getTickCount()
    recognize_results, search_results = faceme_wrapper.process_images(cvimgs)
    t1 = cv2.getTickCount()
    used = (t1 - t0) / cv2.getTickFrequency()
    print(f"{used=}")
    print(faceme_wrapper.bbox_and_name(recognize_results, search_results))

    assert len(recognize_results) == len(search_results)
    # imageIndex が同じ画像の範囲でフィルタリングして、その範囲にあるrecognized, searched の結果をあわせて描画する
    for i in range(len(names)):
        outname = out_dir / f"{names[i].name}"
        oimg = cvimgs[i]
        indexes = [j for j, recognized in enumerate(recognize_results) if recognized["imageIndex"] == i]
        for j in indexes:
            pt1, pt2 = recognize_results[j]["boundingBox"]
            cv2.rectangle(oimg, pt1, pt2, (0, 255, 0), thickness=3)
            person = search_results[j][1][0]["name"] if search_results[j][1] else ""
            oimg = faceme_wrapper.putText_utf(oimg, unicode_text=person, org=pt1, font_size=36, color=(255, 0, 0))
        cv2.imwrite(str(outname), oimg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="face searcher")
    parser.add_argument("src", help="source image (/dev/video0 for camera)")
    args = parser.parse_args()

    faceme_wrapper.initialize_SDK()

    if args.src.find("/dev/video") == 0:
        device_number = int(args.src.replace("/dev/video", ""))
        cap = cv2.VideoCapture(0)
        while True:
            r, img = cap.read()
            recognize_results, search_results = faceme_wrapper.process_images([img])
            print(faceme_wrapper.bbox_and_name(recognize_results, search_results))
            out_cvimg = img
            out_cvimg = faceme_wrapper.draw_recognized(out_cvimg, recognize_results, search_results)
            cv2.imwrite("out_cvimg.jpg", out_cvimg)
            cv2.imshow("out", out_cvimg)
            key = cv2.waitKey(1)
            if key & 0xff == ord('q'):
                break
    elif Path(args.src).is_dir():
        img_dir = Path(args.src)
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        process_image_dir(img_dir, out_dir)
    else:
        img = cv2.imread(args.src)
        recognize_results, search_results = faceme_wrapper.process_images([img])
        print(faceme_wrapper.bbox_and_name(recognize_results, search_results))
        out_cvimg = img
        out_cvimg = faceme_wrapper.draw_recognized(out_cvimg, recognize_results, search_results)
        cv2.imwrite("out_cvimg.jpg", out_cvimg)
        cv2.imshow("out", out_cvimg)
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()
