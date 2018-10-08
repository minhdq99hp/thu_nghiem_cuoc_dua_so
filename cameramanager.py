import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
from datetime import datetime


class CameraManager:
	def __init__(self):
		# Initialize the depth device
		openni2.initialize(dll_directories="/home/minhdq99hp/OpenNi/OpenNI-Linux-x64-2.3/Redist/")
		self.dev = openni2.Device.open_any()

		self.depth_stream = None
		self.color_stream = None

		self.depth_frame = None
		self.color_frame = None

		self.depth_left_roi = None
		self.color_left_roi = None
		self.depth_right_roi = None
		self.color_right_roi = None
		self.depth_main_roi = None
		self.color_main_roi = None

		# Settings
		self.noMirror = True
		self.cropEnable = True

		self.cropX = 46
		self.cropY = 27
		self.cropWidth = 550
		self.cropHeight = 420

	def initialize_depth_stream(self):
		self.depth_stream = self.dev.create_depth_stream()
		self.depth_stream.start()
		self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))

		if self.noMirror:
			self.depth_stream.set_mirroring_enabled(False)
		self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

	def initialize_color_stream(self):
		self.color_stream = self.dev.create_color_stream()
		self.color_stream.start()
		self.color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 640, resolutionY = 480, fps = 30))

		if self.noMirror:
			self.color_stream.set_mirroring_enabled(False)

	def compute_depth_frame(self):
		self.depth_frame = self.depth_stream.read_frame()
		frame_data = self.depth_frame.get_buffer_as_uint16()

		img = np.frombuffer(frame_data, dtype=np.uint16)
		img.shape = (1, 480, 640)

		img = np.swapaxes(img, 0, 2)
		img = np.swapaxes(img, 0, 1)

		self.depth_frame = img

	def get_current_cropped_depth_frame(self):
		return self.depth_frame[self.cropY:self.cropY+self.cropHeight, self.cropX:self.cropX+self.cropWidth, :]

	def compute_color_frame(self):
		self.color_frame = self.color_stream.read_frame()
		frame_data = self.color_frame.get_buffer_as_uint8()

		img = np.frombuffer(frame_data, dtype=np.uint8)
		img.shape = (307200, 3)
		b = img[:, 0].reshape(480, 640)
		g = img[:, 1].reshape(480, 640)
		r = img[:, 2].reshape(480, 640)

		rgb = (r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis])
		img = np.concatenate(rgb, axis=-1)

		self.color_frame = img

	def get_current_color_frame(self):
		return self.color_frame

	def compute_left_roi(self):
		self.depth_left_roi = self.depth_frame[10:210, 10:260, :]
		self.color_left_roi = self.color_frame[10:210, 10:260, :]

	def compute_right_roi(self):
		self.depth_right_roi = self.depth_frame[10:210, 290:540, :]
		self.color_right_roi = self.color_frame[10:210, 290:540, :]

	def compute_main_roi(self):
		self.depth_main_roi = self.depth_frame[210: 420, 10: 540, :]
		self.color_main_roi = self.color_frame[210: 420, 10: 540, :]

	def compute_rois(self):
		self.compute_left_roi()
		self.compute_right_roi()
		self.compute_main_roi()

	def get_current_cropped_color_frame(self):
		return self.get_current_color_frame()[self.cropY:self.cropY+self.cropHeight, self.cropX:self.cropX+self.cropWidth, :]

	def draw_color_left_roi_bounding_box(self):
		cv2.rectangle(self.color_frame, (10, 10), (260, 209), (0, 255, 0), 2)


	@staticmethod
	def get_frame_from(file):
		return np.load(file)

	@staticmethod
	def unload():
		openni2.unload()


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global mouse_x, mouse_y, flag

	if event == cv2.EVENT_LBUTTONDOWN:
		mouse_x = x
		mouse_y = y
		flag = True


mouse_x = 0
mouse_y = 0
flag = False

test_camera = True
cam = None

if __name__ == '__main__':
	if test_camera:
		cam = CameraManager()
		cam.initialize_depth_stream()
		cam.initialize_color_stream()

	print("Press C to exit.\nPress S to save depth\color frame")

	cv2.namedWindow("color_image")
	cv2.setMouseCallback("color_image", click_and_crop)

	gotData = False

	while True:
		if test_camera:
			cam.compute_depth_frame()
			cam.compute_color_frame()

			depth_frame = cam.get_current_cropped_depth_frame()
			color_frame = cam.get_current_cropped_color_frame()
		else:
			depth_frame = np.load("data/depth_frame.npy")
			color_frame = np.load("data/color_frame.npy")

		if flag:
			print("Position: ({x}, {y}) - Depth: {d}".format(x=mouse_x, y=mouse_y, d=depth_frame[mouse_y, mouse_x, 0]))
			flag = False

		cv2.line(color_frame, (275, 210), (275, 419), (0, 0, 255), 2)

		# LAY GROUND_DATA, CHI CHAY 1 LAN
		# if not gotData:
		# 	data = open("data.txt", 'w')
		# 	x_data = []
		# 	y_data = []
		# 	for row in range(210, 419):
		# 		if depth_frame[row, 275] != 0:
		# 			x_data.append(row)
		# 			y_data.append(depth_frame[row, 275, 0])
		#
		# 	data.write(' '.join(str(x) for x in x_data) + '\n' + ' '.join(str(y) for y in y_data))
		#
		# 	data.close()
		# 	gotData = True

		cv2.imshow("depth_image", depth_frame)
		cv2.imshow("color_image", color_frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("c"):
			break
		elif key == ord("s"):
			time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
			np.save("data/{0}_depth_frame.npy".format(time), depth_frame)
			np.save("data/{0}_color_frame.npy".format(time), color_frame)
			print("Saved !")

	cv2.destroyAllWindows()

	if test_camera:
		cam.unload()
