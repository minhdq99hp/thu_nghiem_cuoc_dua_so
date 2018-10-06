import numpy as np
import cv2
from matplotlib import pyplot as plt


class SIFT:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.img1 = None
        self.img2 = None
        self.good = None
        self.kp1 = None
        self.kp2 = None

    def run(self, img1, img2):
        self.img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # find the keypoints and descriptors with SIFT
        self.kp1, des1 = self.sift.detectAndCompute(img1, None)
        self.kp2, des2 = self.sift.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        self.good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                self.good.append([m])

    def get_result_image(self):
        img3 = cv2.drawMatchesKnn(self.img1, self.kp1, self.img2, self.kp2, self.good, None, flags=2)

        return img3


class HOG:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        return

    def get_hog_from_image(self, image):

        return self.hog.compute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
