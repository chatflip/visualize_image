import os

import cv2
import numpy as np
from skimage import feature


class Visualizer(object):

    def __init__(self, src_path, dst_path):
        self.color_image = cv2.imread(src_path)
        # TODO: grayだったらcolorにする
        try:
            _ = self.color_image.shape
        except:
            print("can't read image: {}".format(src_path))
        name, ext = os.path.splitext(src_path)
        self.filename = '{}/{}_%s{}'.format(dst_path, name, ext)

    def gray(self, save=False, show=False):
        gray = cv2.cvtColor(self.color_image.copy(), cv2.COLOR_BGR2GRAY)
        if save:
            cv2.imwrite(self.filename % 'gray', gray)

    def color_channel(self, save=False, show=False):
        color_image = self.color_image.copy()
        blue = np.zeros_like(color_image)
        green = np.zeros_like(color_image)
        red = np.zeros_like(color_image)
        # 各チャネルの輝度値を格納
        blue[:, :, 0] = color_image[:, :, 0]
        green[:, :, 1] = color_image[:, :, 1]
        red[:, :, 2] = color_image[:, :, 2]

        if save:
            cv2.imwrite(self.filename % 'blue', blue)
            cv2.imwrite(self.filename % 'green', green)
            cv2.imwrite(self.filename % 'red', red)

    def gray_gradient(self, save=False, show=False):
        gray = cv2.cvtColor(self.color_image.copy(), cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], np.float32)
        kernel_y = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], np.float32)
        kernel_xy = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]], np.float32)
        img_x = cv2.filter2D(gray, -1, kernel_x)
        img_y = cv2.filter2D(gray, -1, kernel_y)
        img_xy = cv2.filter2D(gray, -1, kernel_xy)

        if save:
            cv2.imwrite(self.filename % 'Xgradient', img_x)
            cv2.imwrite(self.filename % 'Ygradient', img_y)
            cv2.imwrite(self.filename % 'XYgradient', img_xy)

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
            cv2.imwrite(self.filename % 'XcolorGradient', img_x)
            cv2.imwrite(self.filename % 'YcolorGradient', img_y)
            cv2.imwrite(self.filename % 'XYcolorGradient', img_xy)

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
            cv2.imwrite(self.filename % 'power_spectrum', pow_img)

    def drawKeypoint(self, save=False, show=False):
        SIFT_img, RichSIFT_img, SIFT_num = self.draw_SIFT(self.color_image.copy())
        #Dense_img, RichDense_img, Dense_num = self.draw_Dense(self.color_image.copy())
        #print('SIFT  = %5d\nDense = %5d' % (SIFT_num, Dense_num))

        if save:
            cv2.imwrite(self.filename % 'SIFT', SIFT_img)
            cv2.imwrite(self.filename % 'SIFTRich', RichSIFT_img)
            #cv2.imwrite(self.filename % 'Dense', Dense_img)
            #cv2.imwrite(self.filename % 'DenseRich', RichDense_img)

    def draw_SIFT(self, original_img):
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = original_img.copy()
        rich_img = original_img.copy()
        detector = cv2.SIFT_create()
        keypoints = detector.detect(gray)
        for key in keypoints:
            cv2.circle(img, (np.uint64(key.pt[0]), np.uint64(key.pt[1])), 3, (255, 255, 0), 1)
        cv2.drawKeypoints(rich_img, keypoints, rich_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img, rich_img, len(keypoints)

    def draw_Dense(self, original_img):
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = original_img.copy()
        rich_img = original_img.copy()
        detector = cv2.xfeatures2d.SIFT_create()
        #detector = cv2.FeatureDetector_create("Dense")
        keypoints = detector.detect(gray)
        for key in keypoints:
            cv2.circle(img, (np.uint64(key.pt[0]), np.uint64(key.pt[1])), 3, (255, 255, 0), 1)
        cv2.drawKeypoints(rich_img, keypoints, rich_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img, rich_img, len(keypoints)

    def drawHoG(self, save=False, show=False):
        # グレースケール化
        gray = cv2.cvtColor(self.color_image.copy(), cv2.COLOR_BGR2GRAY)
        _, hog_image = feature.hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                                   block_norm="L2-Hys", visualize=True)
        hog_image = np.uint8(hog_image * 255)
        if save:
            cv2.imwrite(self.filename % 'hog', hog_image)

    def trancelbp(self, save=False, show=False):
        img = cv2.copyMakeBorder(self.color_image.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        # グレースケール化
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
                lbp_pix = lbp[0] * 2 ** 7 + lbp[1] * 2 ** 6 + lbp[2] * 2 ** 5 + lbp[4] * 2 ** 4 + lbp[7] * 2 ** 3 + lbp[
                    6] * 2 ** 2 + lbp[5] * 2 ** 1 + lbp[3] * 2 ** 0
                lbp_picture[centerY - 1, centerX - 1] = lbp_pix
                counter = 0
        if save:
            cv2.imwrite(self.filename % 'lbp', lbp_picture)

    def save_all(self):
        self.gray(save=True)
        self.color_channel(save=True)
        self.gray_gradient(save=True)
        self.color_gradient(save=True)
        self.power_spectrum(save=True)
        self.drawKeypoint(save=True)
        self.drawHoG(save=True)
        self.trancelbp(save=True)