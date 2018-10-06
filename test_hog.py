from objectrecognition import HOG
from cameramanager import CameraManager
import cv2

cam = None

if __name__ == '__main__':
    # cam = CameraManager()
    # cam.initialize_depth_stream()
    # cam.initialize_color_stream()

    cv2.namedWindow("color_image")

    hog = HOG()
    im = cv2.imread("data/sift_test.jpg")

    while True:
        # cam.compute_depth_frame()
        # cam.compute_color_frame()
        #
        # depth_frame = cam.get_current_cropped_depth_frame()
        # color_frame = cam.get_current_cropped_color_frame()

        # cv2.imshow("depth_image", depth_frame)
        # cv2.imshow("color_image", color_frame)

        h = hog.get_hog_from_image(im)

        print(h.shape)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

    cv2.destroyAllWindows()
    cam.unload()
