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

class ExtractLaneLines():
    '''
    Class to extract lane line pixels for a given image
    Applying gradient, color thresholding techniques
    '''

    def __init__(self, sobel_kernel=15, x_thresh=(20, 100), y_thresh=(20, 100), m_thresh=(30, 100), d_thresh=(0.7, 1.3), s_thresh=(170, 255), sx_thresh=(20, 100)):
        '''
        '''
        self.sobel_kernel = sobel_kernel
        self.x_thresh = x_thresh
        self.y_thresh = y_thresh
        self.m_thresh = m_thresh
        self.d_thresh = d_thresh
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh

    def abs_sobel_thresh(self, gray, orient='x', thresh=(0, 255)):
        # Take the derivative in x or y given orient = 'x' or 'y'
        if orient=='x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
        grad_output = np.zeros_like(scaled_sobel)
        grad_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return grad_output

    def mag_thresh(self, gray, thresh=(0, 255)):
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # Calculate the magnitude 
        sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*sobelxy/np.max(sobelxy))
        # 5) Create a binary mask where mag thresholds are met
        mag_output = np.zeros_like(scaled_sobel)
        mag_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return mag_output

    def dir_thresh(self, gray, thresh=(0, np.pi/2)):
        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        arc_sobel = np.arctan2(abs_sobely, abs_sobelx)
        # Create a binary mask where direction thresholds are met
        dir_output = np.zeros_like(arc_sobel)
        dir_output[(arc_sobel >= thresh[0]) & (arc_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return dir_output

    def hsi_thresh(self, img, thresh=(170, 255), x_thresh=(20, 100)):
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
        sxbinary[(scaled_sobel >= x_thresh[0]) & (scaled_sobel <= x_thresh[1])] = 1
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        # Stack each channel
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        return color_binary

    def sobel_plus_hsv_thresh_debug(self, img):
        # Get the HSI thresholded image
        hsi_image = self.hsi_thresh(img, self.s_thresh, self.sx_thresh)
        # Convert image to grayscale
        gray = cv2.cvtColor(hsi_image, cv2.COLOR_RGB2GRAY)
        # Get x axis threshold gradient image
        gradx = self.abs_sobel_thresh(gray, orient='x', thresh=self.x_thresh)
        # Get y axis threshold gradient image
        grady = self.abs_sobel_thresh(gray, orient='y', thresh=self.y_thresh)
        # Get magnitude threshold gradient image
        mag_binary = self.mag_thresh(gray, thresh=self.m_thresh)
        # Get directional threshold gradient image
        dir_binary = self.dir_thresh(gray, thresh=self.d_thresh)
        # Create a composite image of thresholded outputs
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        # Plotting the output for debugging purpose
        f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax2.imshow(hsi_image)
        ax3.imshow(gradx, cmap="gray")
        ax4.imshow(grady, cmap="gray")
        ax5.imshow(mag_binary, cmap="gray")
        ax6.imshow(dir_binary, cmap="gray")
        ax7.imshow(combined, cmap="gray")
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        return combined

    def sobel_plus_hsv_thresh(self, img):
        # Get the HSI thresholded image
        hsi_image = self.hsi_thresh(img, self.s_thresh, self.sx_thresh)
        # Convert image to grayscale
        gray = cv2.cvtColor(hsi_image, cv2.COLOR_RGB2GRAY)
        # Get x axis threshold gradient image
        gradx = self.abs_sobel_thresh(gray, orient='x', thresh=self.x_thresh)
        # Get y axis threshold gradient image
        grady = self.abs_sobel_thresh(gray, orient='y', thresh=self.y_thresh)
        # Get magnitude threshold gradient image
        mag_binary = self.mag_thresh(gray, thresh=self.m_thresh)
        # Get directional threshold gradient image
        dir_binary = self.dir_thresh(gray, thresh=self.d_thresh)
        # Create a composite image of thresholded outputs
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined

if __name__ == "__main__":
    ## Testing camera calibration and image undistortion
    image_calibration = CalibrateCamera(debug=True)
    image_calibration.calculate_imgpoints_over_directory()
    image_calibration.calculate_calibration_parameters()
    test_image = mpimg.imread("./camera_cal/calibration1.jpg")
    image_calibration.undistort_image(test_image)
        
