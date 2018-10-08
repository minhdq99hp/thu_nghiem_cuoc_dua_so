import numpy as np
import matplotlib.pyplot as plt
import cv2
from enum import Enum


class LayerIndex(Enum):
    FIRST_LAYER = 0
    SECOND_LAYER = 1
    THIRD_LAYER = 2
    LAST_LAYER = -1


class ObjectSegmentationAlgorithm:
    def __init__(self):
        self.histogram_of_depth = None
        self.output_image = None
        self.bounding_box_info = None
        self.valid_points = None
        self.local_peaks = None
        self.global_peaks = None
        self.group_of_regions = None

        self.depth_frame = None
        self.color_frame = None

        self.ground_offset = 3000
        self.depth_ground = np.load('data/depth_ground_data.npy')

        # BUG: Overflow and Underflow occured here ! Because the data type is np.uint16
        self.upper_depth_ground = self.depth_ground.astype(np.int) + self.ground_offset
        self.lower_depth_ground = self.depth_ground.astype(np.int) - self.ground_offset

        self.d = 3000

    def load_frames(self, depth, color):
        self.depth_frame = depth
        self.color_frame = color

    def compute_histogram_of_depth(self):
        frame_width = self.depth_frame.shape[0]
        frame_height = self.depth_frame.shape[1]

        hod = np.zeros((65536,), dtype=int)

        for row in range(frame_width):
            for col in range(frame_height):
                hod[self.depth_frame[row, col, 0]] += 1

        self.histogram_of_depth = hod

    def get_histogram_of_depth(self):
        return self.histogram_of_depth

    def is_valid_point(self, point):
        if self.histogram_of_depth[point] > 0:
            return True
        else:
            return False

    def compute_valid_points(self):
        valid_points = []
        for point in range(1, self.histogram_of_depth.shape[0]):  # The value 0 is ignored
            if self.is_valid_point(point):
                valid_points.append(point)

        self.valid_points = valid_points

    def get_valid_points(self):
        return self.valid_points

    def compute_peaks(self, points):
        peaks = []
        hod = self.histogram_of_depth
        pt = points

        if len(pt) < 2:
            return points

        if hod[pt[0]] >= hod[pt[1]]:
            peaks.append(pt[0])

        for i in range(1, len(pt) - 1):
            if hod[pt[i]] >= hod[pt[i + 1]] and hod[pt[i]] >= hod[pt[i - 1]]:
                peaks.append(pt[i])

        if hod[pt[-1]] >= hod[pt[-2]]:
            peaks.append(pt[-1])

        return peaks

    def compute_local_peaks(self):
        self.local_peaks = self.compute_peaks(self.valid_points);

    def get_local_peaks(self):
        return self.local_peaks

    def filter(self, points):
        hod = self.histogram_of_depth

        if len(points) < 2 or abs(points[0] - points[-1]) <= self.d:
            return points

        for i in range(len(points)-1):
            if abs(points[i + 1] - points[i]) > self.d:

                # Split regions at index i
                # Process left_points
                l_peaks = self.compute_peaks(points[:i])
                # Process right_points
                r_peaks = self.compute_peaks(points[i:])

                # self.show_histogram_of_depth_points(points)
                return self.filter(l_peaks) + self.filter(r_peaks)

        # self.show_histogram_of_depth_points(points)
        return self.filter(self.compute_peaks(points))

    def compute_global_peaks(self):

        self.global_peaks = self.filter(self.local_peaks)

    def get_global_peaks(self):
        return self.global_peaks

    def compute_group_of_regions(self):
        group_of_regions = []

        if len(self.global_peaks) == 1:
            return [(self.valid_points[0], self.valid_points[-1])]

        start_point = self.valid_points[0]
        end_point = self.global_peaks[0]

        for i in range(1, len(self.global_peaks)-1):
            if abs(self.global_peaks[i] - end_point) <= self.d:
                end_point = self.global_peaks[i]

            if abs(self.global_peaks[i] - self.global_peaks[i+1]) > self.d:
                end_point = (self.global_peaks[i] + self.global_peaks[i+1])//2
                group_of_regions.append((start_point, end_point))
                start_point = end_point + 1
                end_point = self.global_peaks[i+1]
        end_point = self.valid_points[-1]
        group_of_regions.append((start_point, end_point))

        self.group_of_regions = group_of_regions

    def get_group_of_regions(self):
        return self.group_of_regions

    def run(self):
        self.compute_histogram_of_depth()
        self.compute_valid_points()
        self.compute_local_peaks()
        self.compute_global_peaks()
        self.compute_group_of_regions()

    def compute_depth_layer_from_region(self, region):
        frame = self.depth_frame[:, :, 0].copy()

        frame[frame > region[1]] = 0
        frame[frame < region[0]] = 0

        return frame

    def get_depth_layer_from_region(self, region):
        return self.compute_depth_layer_from_region(region)

    def compute_color_layer_from_depth_layer(self, d_layer):
        c_frame = self.color_frame.copy()
        for row in range(self.color_frame.shape[0]):
            for col in range(self.color_frame.shape[1]):
                if d_layer[row, col] == 0:
                    c_frame[row, col, :] = [0,0,0]

        return c_frame

    def get_color_layer_from_region(self, d_layer):
        return self.compute_color_layer_from_depth_layer(d_layer)

    def get_output_image(self):
        return self.output_image

    def get_object_bounding_box_info(self):
        return self.bounding_box_info

    def show_histogram_of_depth_points(self, points):
        plt.plot(points, [self.histogram_of_depth[points[x]] for x in range(len(points))], 'ro')
        plt.show()

    def get_layer(self, index):
        return self.get_depth_layer_from_region(self.group_of_regions[index])

    def show_layer(self, index):
        cv2.imshow(str(index), self.get_color_layer_from_region(self.get_layer(index)))

    def show_all_layers(self):
        for i in range(len(self.group_of_regions)):
            d_layer = self.get_depth_layer_from_region(self.group_of_regions[i])

            cv2.imshow(str(i), self.get_color_layer_from_region(d_layer))

    def upper_threshold(self, roi, value):
        roi[roi>value] = 0

    def find_contours(self, roi):

        img = (roi / 65535 * 255) // 1

        image = roi.astype(np.uint8)

        ret, thresh = cv2.threshold(image, 127, 255, 0)

        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # find bounding box coordinates
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("r", img)

        return contours

    def get_bounding_box_coordinate(self, roi):
        x = None
        y = None
        x2 = None
        y2 = None

        # Find Y
        for row in range(roi.shape[0]):
            found_y = False
            for col in range(roi.shape[1]):
                if roi[row, col, 0] != 0:
                    y = col
                    found_y = True
                    break

            if found_y:
                break

        # Find X
        for col in range(roi.shape[1]):
            found_x = False
            for row in range(roi.shape[0]):
                if roi[row, col, 0] != 0:
                    x = row
                    found_x = True
                    break
            if found_x:
                break

        # Find Y2
        for row in range(roi.shape[0]-1, -1, -1):
            found_y2 = False
            for col in range(roi.shape[1]):
                if roi[row, col, 0] != 0:
                    y2 = col
                    found_y2 = True
                    break
            if found_y2:
                break

        # Find X2
        for col in range(roi.shape[1]-1, -1, -1):
            found_x2 = False
            for row in range(roi.shape[0]):
                if roi[row, col, 0] != 0:
                    x2 = row
                    found_x2 = True
                    break
            if found_x2:
                break

        return x, y, x2, y2

    def is_ground(self, row, col):
        return self.depth_ground[row, col] - self.ground_offset < self.depth_frame[row, col] < self.depth_ground[row, col] + self.ground_offset

    def remove_ground(self, x, y, x2, y2):
        for row in range(y, y2+1):
            for col in range(x, x2+1):
                if self.depth_frame[row, col, 0] != 0:
                    if self.is_ground(row, col):
                        self.depth_frame[row, col, 0] = 0

    def remove_ground_2(self, x, y, x2, y2):
        roi = self.depth_frame[y:y2+1, x:x2+1, 0]

        # Create masks
        diff_up = roi - self.upper_depth_ground[y:y2+1, x:x2+1, 0]
        diff_down = roi - self.lower_depth_ground[y:y2+1, x:x2+1, 0]

        diff_up[diff_up >= 0] = 1
        diff_up[diff_up < 0] = 0

        diff_down[diff_down <= 0] = -2 # set to a value != 1 to avoid the effect of the next step
        diff_down[diff_down > 0] = 0
        diff_down[diff_down == -2] = 1

        self.depth_frame[y:y2 + 1, x:x2 + 1, 0] = roi * (diff_up + diff_down)


    # Heuristic
    def compute_depth_ground(self):
        depth_ground = np.zeros(self.depth_frame.shape, dtype=np.uint16)

        # for row in range(depth_ground.shape[0]):
        for row in range(211, depth_ground.shape[0]):
            depth_of_center_row = self.get_depth_of_center_row(row)

            for col in range(depth_ground.shape[1]):
                depth_ground[row, col, 0] = depth_of_center_row + abs(28 / 11 * col - 700) // 1

        self.depth_ground = depth_ground

        # np.save('data/depth_ground_data.npy', depth_ground)

    # Heuristic
    def get_depth_of_center_row(self, row):
        # The value of these local variables is got from test_ground/fit_curve_based_on_ground_data
        a = 1.94926806e+06
        b = -1.90139644e+02

        result = (row + a)/(row + b)

        if result > 65535:
            result = 65535
        return np.uint16(result)


def test_ground_removal():
    global objectSegmentation

    depth_frame = np.load("data/2018-10-08-11-01-49_depth_frame.npy")
    color_frame = np.load("data/2018-10-08-11-01-49_color_frame.npy")

    objectSegmentation.load_frames(depth_frame, color_frame)
    # objectSegmentation.compute_depth_ground()
    objectSegmentation.remove_ground_2(0, 211, 549, 419)

    cv2.imshow("depth_frame", objectSegmentation.depth_frame)

    depth_frame = np.load("data/2018-10-08-11-01-49_depth_frame.npy")
    cv2.imshow("origin_depth_frame", depth_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_segmentation():
    global objectSegmentation

    depth_frame = np.load("data/depth_frame.npy")
    color_frame = np.load("data/color_frame.npy")

    objectSegmentation.run()

    print("Group of Regions: " + str(objectSegmentation.get_group_of_regions()))

    objectSegmentation.show_all_layers()

    objectSegmentation.show_histogram_of_depth_points(objectSegmentation.get_global_peaks())

    cv2.destroyAllWindows()


if __name__ == "__main__":
    objectSegmentation = ObjectSegmentationAlgorithm()

    test_ground_removal()
    # test_segmentation()


