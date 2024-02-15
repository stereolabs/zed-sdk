import cv2

import faceme_wrapper

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
            recognize_results, search_results = faceme_wrapper.process_image(img)
            out_cvimg = img
            out_cvimg = faceme_wrapper.draw_recognized(out_cvimg, recognize_results, search_results)
            cv2.imwrite("out_cvimg.jpg", out_cvimg)
            cv2.imshow("out", out_cvimg)
            key = cv2.waitKey(1)
            if key & 0xff == ord('q'):
                break
    else:
        img = cv2.imread(args.src)
        recognize_results, search_results = faceme_wrapper.process_image(img)
        out_cvimg = img
        out_cvimg = faceme_wrapper.draw_recognized(out_cvimg, recognize_results, search_results)
        cv2.imwrite("out_cvimg.jpg", out_cvimg)
        cv2.imshow("out", out_cvimg)
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()
