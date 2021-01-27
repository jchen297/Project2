"""
Image Stitching Problem
(Due date: Nov. 9, 11:59 P.M., 2020)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """
    # Some of the codes are adapted from: https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)

    # find key points
    sift = cv2.xfeatures2d.SIFT_create()
    right_kp, right_des = sift.detectAndCompute(right, None)
    left_kp, left_des = sift.detectAndCompute(left, None)

    # get match points index with distance lower than 10 ---- hand writen, so it take time
    matched_points_index = []
    len_img = len(left_des)
    for left_pos in range(len_img-1,int(len_img/5*4),-1):
        distance = list(map(np.linalg.norm, left_des[left_pos] - right_des))
        dis = min(distance)
        best_position = distance.index(dis)
        if dis < 10:
            matched_points_index.append([left_kp[left_pos], right_kp[best_position]])
            if len(matched_points_index) > 40:
                break

    # find the matched points by index that we get above
    # Some of the codes are adapted from: https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    src_pts = np.float32([p[1].pt for p in matched_points_index]).reshape(-1, 1, 2)
    dst_pts = np.float32([p[0].pt for p in matched_points_index]).reshape(-1, 1, 2)

    # get Homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # wrap the image with Homography matrix
    # Some of the codes are adapted from: https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    warped_img = cv2.warpPerspective(right_img, H, (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
    warped_img[0:left_img.shape[0], 0:left_img.shape[1]] = left_img

    # define a function to crop the black area of the image
    def crop_black(img):
        # Some of the codes are adapted from: https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = img[y:y + h, x:x + w]
        return crop

    cropped = crop_black(warped_img)
    return cropped


if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg',result_image)
