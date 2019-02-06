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

## Functions for image thresholding
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    grad_output = np.zeros_like(scaled_sobel)
    grad_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return grad_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobelxy/np.max(sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    mag_output = np.zeros_like(scaled_sobel)
    mag_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mag_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    arc_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_output = np.zeros_like(arc_sobel)
    dir_output[(arc_sobel >= thresh[0]) & (arc_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return dir_output

def combining_thresholds():
    pass

def plot_image_side_by_side(image1, image2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('Combined Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

if __name__ == "__main__":
    ## Testing camera calibration and image undistortion
    image_calibration = CalibrateCamera(debug=True)
    image_calibration.calculate_imgpoints_over_directory()
    image_calibration.calculate_calibration_parameters()
    test_image = mpimg.imread("./camera_cal/calibration1.jpg")
    image_calibration.undistort_image(test_image)
        
