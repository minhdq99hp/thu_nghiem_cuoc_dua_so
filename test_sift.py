from objectrecognition import SIFT
from cameramanager import CameraManager
import cv2

cam = None

if __name__ == '__main__':
	cam = CameraManager()
	cam.initialize_depth_stream()
	cam.initialize_color_stream()

	cv2.namedWindow("color_image")

	sift = SIFT()
	sift_test = cv2.imread("data/traffic_signs/turn_left_sign.png")
	while True:
		cam.compute_depth_frame()
		cam.compute_color_frame()

		depth_frame = cam.get_current_cropped_depth_frame()
		color_frame = cam.get_current_cropped_color_frame()

		# SIFT
		sift.run(sift_test, cam.color_frame)
		cv2.imshow("SIFT", sift.get_result_image())

		cv2.imshow("depth_image", depth_frame)
		cv2.imshow("color_image", color_frame)


		key = cv2.waitKey(1) & 0xFF
		if key == ord("c"):
			break

	cv2.destroyAllWindows()
	cam.unload()