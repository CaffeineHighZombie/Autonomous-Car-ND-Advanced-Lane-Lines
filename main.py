import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


class CalibrateCamera:
    '''
    Class to take in calibration images, calculate calibration parameters,
    and undistort any given images
    '''
    def __init__(self, nx=9, ny=6, debug=False):
        '''
        '''
        self.nx = nx
        self.ny = ny
        self.objpoints = []
        self.imgpoints = []
        self.objp = np.zeros((nx*ny,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        self.debug = debug
        self.image_shape = None
        self.mtx = None
        self.dist = None
    
    def calculate_imgpoints(self, fname):
        '''
        '''
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_shape = gray.shape
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
        if ret == True:
            self.imgpoints.append(corners)
            self.objpoints.append(self.objp)
            img = cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
            if self.debug:
                plt.imshow(img)
                plt.show()
    
    def calculate_imgpoints_over_directory(self, directory="./camera_cal", fname_prototype="calibration*.jpg"):
        '''
        '''
        directory_path = directory + '/' + fname_prototype
        if self.debug: print("Directory path", directory_path)
        images = glob.glob(directory_path)
        for image in images:
            self.calculate_imgpoints(image)

    def calculate_calibration_parameters(self):
        '''
        '''
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_shape[::-1], None, None)
    
    def undistort_image(self, image):
        '''
        '''
        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        if self.debug:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(image)
            ax1.set_title("Original Image", fontsize=50)
            ax2.imshow(undist)
            ax2.set_title("Undistorted Image", fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()
        return undist

if __name__ == "__main__":
    ## Testing camera calibration and image undistortion
    image_calibration = CalibrateCamera(debug=True)
    image_calibration.calculate_imgpoints_over_directory()
    image_calibration.calculate_calibration_parameters()
    test_image = mpimg.imread("./camera_cal/calibration1.jpg")
    image_calibration.undistort_image(test_image)
        
