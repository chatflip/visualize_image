import os

import cv2
import numpy as np
from dense_feature_detector import DenseFeatureDetector
from skimage import feature


class Visualizer(object):
    def __init__(self, src_path, dst_path):
        self.color_image = cv2.imread(src_path)
        # TODO: grayだったらcolorにする
        try:
            _ = self.color_image.shape
        except Exception:
            print("can't read image: {}".format(src_path))
        name, ext = os.path.splitext(src_path)
        self.filename = "{}/{}_%s{}".format(dst_path, name, ext)

    @staticmethod
    def show_image(image, filename, is_last=True):
        if is_last:
            cv2.imshow(image, filename)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imshow(image, filename)

    def gray(self, save=False, show=False):
        gray = cv2.cvtColor(self.color_image.copy(), cv2.COLOR_BGR2GRAY)

        if save:
            cv2.imwrite(self.filename % "gray", gray)

        if show:
            self.show_image(self.filename % "gray", gray)

    def color_channel(self, save=False, show=False):
        color_image = self.color_image.copy()
        blue = np.zeros_like(color_image)
        green = np.zeros_like(color_image)
        red = np.zeros_like(color_image)
        blue[:, :, 0] = color_image[:, :, 0]
        green[:, :, 1] = color_image[:, :, 1]
        red[:, :, 2] = color_image[:, :, 2]

        if save:
            cv2.imwrite(self.filename % "blue", blue)
            cv2.imwrite(self.filename % "green", green)
            cv2.imwrite(self.filename % "red", red)

        if show:
            self.show_image(self.filename % "blue", blue, is_last=False)
            self.show_image(self.filename % "green", green, is_last=False)
            self.show_image(self.filename % "red", red)

    def gray_gradient(self, save=False, show=False):
        gray = cv2.cvtColor(self.color_image.copy(), cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], np.float32)
        kernel_y = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], np.float32)
        kernel_xy = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]], np.float32)
        img_x = cv2.filter2D(gray, -1, kernel_x)
        img_y = cv2.filter2D(gray, -1, kernel_y)
        img_xy = cv2.filter2D(gray, -1, kernel_xy)

        if save:
            cv2.imwrite(self.filename % "gradient_x", img_x)
            cv2.imwrite(self.filename % "gradient_y", img_y)
            cv2.imwrite(self.filename % "gradient_xy", img_xy)

        if show:
            self.show_image(self.filename % "gradient_x", img_x, is_last=False)
            self.show_image(self.filename % "gradient_y", img_y, is_last=False)
            self.show_image(self.filename % "gradient_xy", img_xy)

    def color_gradient(self, save=False, show=False):
        color_image = self.color_image.copy()
        img_x = np.zeros_like(self.color_image.copy())
        img_y = np.zeros_like(self.color_image.copy())
        img_xy = np.zeros_like(self.color_image.copy())
        kernel_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], np.float32)
        kernel_y = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], np.float32)
        kernel_xy = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]], np.float32)
        img_x[:, :, 0] = cv2.filter2D(color_image[:, :, 0], -1, kernel_x)
        img_y[:, :, 0] = cv2.filter2D(color_image[:, :, 0], -1, kernel_y)
        img_xy[:, :, 0] = cv2.filter2D(color_image[:, :, 0], -1, kernel_xy)
        img_x[:, :, 1] = cv2.filter2D(color_image[:, :, 1], -1, kernel_x)
        img_y[:, :, 1] = cv2.filter2D(color_image[:, :, 1], -1, kernel_y)
        img_xy[:, :, 1] = cv2.filter2D(color_image[:, :, 1], -1, kernel_xy)
        img_x[:, :, 2] = cv2.filter2D(color_image[:, :, 2], -1, kernel_x)
        img_y[:, :, 2] = cv2.filter2D(color_image[:, :, 2], -1, kernel_y)
        img_xy[:, :, 2] = cv2.filter2D(color_image[:, :, 2], -1, kernel_xy)

        if save:
            cv2.imwrite(self.filename % "gradient_color_x", img_x)
            cv2.imwrite(self.filename % "gradient_color_y", img_y)
            cv2.imwrite(self.filename % "gradient_color_xy", img_xy)

        if show:
            self.show_image(self.filename % "gradient_color_x", img_x, is_last=False)
            self.show_image(self.filename % "gradient_color_y", img_y, is_last=False)
            self.show_image(self.filename % "gradient_color_xy", img_xy)

    def power_spectrum(self, save=False, show=False):
        gray = cv2.cvtColor(self.color_image.copy(), cv2.COLOR_BGR2GRAY)
        gray = np.array(gray)
        fft = np.fft.fft2(gray)
        fft = np.fft.fftshift(fft)
        Pow = np.abs(fft) ** 2
        Pow = np.log10(Pow)
        Pmax = np.max(Pow)
        Pow = Pow / Pmax * 255
        pow_img = np.array(np.uint8(Pow))

        if save:
            cv2.imwrite(self.filename % "power_spectrum", pow_img)

    def draw_keypoint(self, save=False, show=False):
        sift_img, rich_sift_img, sift_keypoint_num = self.draw_sift(
            self.color_image.copy()
        )
        akaze_img, rich_akaze_img, akaze_keypoint_num = self.draw_akaze(
            self.color_image.copy()
        )
        dense_img, rich_dense_img, dense_keypoint_num = self.draw_dense(
            self.color_image.copy()
        )
        print("num of kps(SIFT) : {:5d}".format(sift_keypoint_num))
        print("num of kps(AKAZE): {:5d}".format(akaze_keypoint_num))
        print("num of kps(DENSE): {:5d}".format(dense_keypoint_num))

        if save:
            cv2.imwrite(self.filename % "sift", sift_img)
            cv2.imwrite(self.filename % "sift_rich", rich_sift_img)
            cv2.imwrite(self.filename % "akaze", akaze_img)
            cv2.imwrite(self.filename % "akaze_rich", rich_akaze_img)
            cv2.imwrite(self.filename % "dense", dense_img)
            cv2.imwrite(self.filename % "dense_rich", rich_dense_img)

        if show:
            self.show_image(self.filename % "sift", sift_img, is_last=False)
            self.show_image(self.filename % "sift_rich", rich_sift_img, is_last=False)
            self.show_image(self.filename % "akaze", akaze_img, is_last=False)
            self.show_image(self.filename % "akaze_rich", rich_akaze_img)
            self.show_image(self.filename % "dense", dense_img, is_last=False)
            self.show_image(self.filename % "dense_rich", rich_dense_img)

    def draw_sift(self, original_img):
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = original_img.copy()
        rich_img = original_img.copy()
        detector = cv2.SIFT_create()
        keypoints = detector.detect(gray)
        for key in keypoints:
            cv2.circle(
                img, (np.uint64(key.pt[0]), np.uint64(key.pt[1])), 3, (255, 255, 0), 1
            )
        cv2.drawKeypoints(
            rich_img,
            keypoints,
            rich_img,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return img, rich_img, len(keypoints)

    def draw_akaze(self, original_img):
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = original_img.copy()
        rich_img = original_img.copy()
        detector = cv2.AKAZE_create()
        keypoints = detector.detect(gray)
        for key in keypoints:
            cv2.circle(
                img, (np.uint64(key.pt[0]), np.uint64(key.pt[1])), 3, (255, 255, 0), 1
            )
        cv2.drawKeypoints(
            rich_img,
            keypoints,
            rich_img,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return img, rich_img, len(keypoints)

    def draw_dense(self, original_img):
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = original_img.copy()
        rich_img = original_img.copy()
        detector = DenseFeatureDetector(cv2.SIFT_create(), step=5, scale=5, start=0)
        keypoints, _ = detector.detect(gray)
        for key in keypoints:
            cv2.circle(
                img, (np.uint64(key.pt[0]), np.uint64(key.pt[1])), 3, (255, 255, 0), 1
            )
        cv2.drawKeypoints(
            rich_img,
            keypoints,
            rich_img,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return img, rich_img, len(keypoints)

    def draw_hog(self, save=False, show=False):
        gray = cv2.cvtColor(self.color_image.copy(), cv2.COLOR_BGR2GRAY)
        _, hog_image = feature.hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
        )
        hog_image = np.uint8(hog_image * 255)
        if save:
            cv2.imwrite(self.filename % "hog", hog_image)

        if show:
            self.show_image(self.filename % "hog", hog_image)

    def draw_lbp(self, save=False, show=False):
        img = cv2.copyMakeBorder(
            self.color_image.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        counter = 0
        lbp = 8 * [0]
        lbp_picture = np.zeros((img.shape[0] - 2, img.shape[1] - 2), dtype=np.uint8)
        for centerY in range(1, img.shape[0] - 1):
            for centerX in range(1, img.shape[1] - 1):
                for yy in range(centerY - 1, centerY + 2):
                    for xx in range(centerX - 1, centerX + 2):
                        if (xx != centerX) or (yy != centerY):
                            if img[centerY, centerX] >= img[yy, xx]:
                                lbp[counter] = 0
                            else:
                                lbp[counter] = 1
                            counter += 1
                lbp_pix = (
                    lbp[0] * 2 ** 7
                    + lbp[1] * 2 ** 6
                    + lbp[2] * 2 ** 5
                    + lbp[4] * 2 ** 4
                    + lbp[7] * 2 ** 3
                    + lbp[6] * 2 ** 2
                    + lbp[5] * 2 ** 1
                    + lbp[3] * 2 ** 0
                )
                lbp_picture[centerY - 1, centerX - 1] = lbp_pix
                counter = 0

        if save:
            cv2.imwrite(self.filename % "lbp", lbp_picture)

        if show:
            self.show_image(self.filename % "lbp", lbp_picture)

    def save_all(self):
        self.gray(save=True)
        self.color_channel(save=True)
        self.gray_gradient(save=True)
        self.color_gradient(save=True)
        self.power_spectrum(save=True)
        self.draw_keypoint(save=True)
        self.draw_hog(save=True)
        self.draw_lbp(save=True)

    def show_all(self):
        self.gray(show=True)
        self.color_channel(show=True)
        self.gray_gradient(show=True)
        self.color_gradient(show=True)
        self.power_spectrum(show=True)
        self.draw_keypoint(show=True)
        self.draw_hog(show=True)
        self.draw_lbp(show=True)
