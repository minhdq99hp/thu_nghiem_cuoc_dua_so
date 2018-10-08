#!/usr/bin/python
import cv2
import numpy as np
from cameramanager import CameraManager
from test import Tester
from objectsegmentation import ObjectSegmentationAlgorithm, LayerIndex
from objectrecognition import SIFT
import time
import timeit

def get_color_road_region(color_frame):
	frame_height = color_frame.shape[0]
	frame_width = color_frame.shape[1]
	road_region = color_frame[frame_height//2 + 48:frame_height, :, :]

	return road_region

def get_depth_road_region(depth_frame):
	frame_height = depth_frame.shape[0]
	frame_width = depth_frame.shape[1]

	road_region = depth_frame[frame_height//2 + 48:frame_height, :, :].copy()

	for row in range(road_region.shape[0]):
		for col in range(0,204):
			if(row < get_left_lane_y_position(col)):
				road_region[row, col, 0] = 0
		for col in range(357, 549):
			if(row < get_right_lane_y_position(col)):
				road_region[row, col, 0] = 0

	return road_region

def get_right_lane_y_position(x):
	return (19/29 * x - 6574/29) // 1
def get_left_lane_y_position(x):
	return (-133/204 * x + 133) // 1


if __name__ == "__main__":
	tester = Tester()

	cam = CameraManager()
	cam.initialize_depth_stream()
	cam.initialize_color_stream()

	objectSegmentation = ObjectSegmentationAlgorithm()

	# Settings
	run_only_once = False

	while True:
		# GETTING DATA
		cam.compute_color_frame()
		cam.compute_depth_frame()

		if cam.cropEnable:
			cam.depth_frame = cam.get_current_cropped_depth_frame()
			cam.color_frame = cam.get_current_cropped_color_frame()

		cam.compute_rois() # compute left_roi, right_roi, main_roi

		# PROCESS DATA

		origin_depth_frame = cam.depth_frame.copy()

		# UPPER THRESHOLD
		objectSegmentation.upper_threshold(cam.depth_left_roi, 20000)
		objectSegmentation.upper_threshold(cam.depth_right_roi, 20000)

		# REMOVE GROUND
		objectSegmentation.load_frames(cam.depth_frame, cam.color_frame)
		objectSegmentation.remove_ground_2(0, 211, 549, 419)

		# TRAFFIC SIGNS LOCALIZATION
		# objectSegmentation.find_contours(depth_left_roi)

		# x, y, x2, y2 = objectSegmentation.get_bounding_box_coordinate(cam.depth_left_roi)
		# cv2.rectangle(cam.color_left_roi, (x, y), (x2, y2), (0, 255, 0), 2)

		# TEST RUNTIME BLOCK
		# start = timeit.default_timer()
		# stop = timeit.default_timer()
		# tester.print_interval(start, stop)

		color_road_region = get_color_road_region(cam.color_frame)
		depth_road_region = get_depth_road_region(cam.depth_frame)


		# DRAWING
		cv2.line(color_road_region, (549, 133), (346, 0), (255, 0, 0), 3)
		cv2.line(color_road_region, (0, 133), (204, 0), (255, 0, 0), 3)

		cv2.rectangle(cam.color_frame, (10, 210), (540, 419), (0, 255, 0), 2) # main_roi
		cv2.rectangle(cam.color_frame, (10, 10), (260, 209), (0, 255, 0), 2) # left_roi
		cv2.rectangle(cam.color_frame, (539, 10), (290, 209), (0, 255, 0), 2) # right_roi

		cv2.rectangle(cam.depth_frame[:, :, 0], (539, 10), (290, 209), 65535, 2)
		cv2.rectangle(cam.depth_frame[:, :, 0], (10, 10), (260, 209), 65535, 2)
		cv2.rectangle(cam.depth_frame[:, :, 0], (10, 210), (540, 419), 65535, 2)

		# SHOWING
		# cv2.imshow("origin_depth_frame", origin_depth_frame)
		cv2.imshow("color_image", cam.color_frame)
		cv2.imshow("depth_image", cam.depth_frame)

		if run_only_once:
			cv2.waitKey(0)
			break
		else:
			key = cv2.waitKey(1) & 0xFF
			if key == ord("c"):
				break

	cam.unload()
	cv2.destroyAllWindows()
