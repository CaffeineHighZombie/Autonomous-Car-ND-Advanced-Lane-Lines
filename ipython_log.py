# IPython log file

get_ipython().run_line_magic('logstart', '')
from main import *
pipeline
image_cal = CalibrateCamera()
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from main import *
check
from main import *
check
check
check
from main import *
check
image_cal = CalibrateCamera()
image_cal.calculate_imgpoints_over_directory()
image_cal.calculate_calibration_parameters()
img = mpimg.imread("./test_images/straight_lines2.jpg")
plt.imshow(img)
plt.show()
ksize = 5
from main import *
from main import *
ksize = 5
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plot_image_side_by_side(img, gray)
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3)))
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
plot_image_side_by_side(img, gradx)
plot_image_side_by_side(img, grady)
plot_image_side_by_side(img, mag_binary)
plot_image_side_by_side(img, dir_binary)
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
plot_image_side_by_side(img, combined)
plot_image_side_by_side(img, dir_binary)
plot_image_side_by_side(img, mag_binary)
plot_image_side_by_side(img, grady)
plot_image_side_by_side(img, gradx)
plot_image_side_by_side(img, combined)
color_binary = pipeline(img)
plt.imshow(color_binary)
plt.show()
gray = cv2.cvtColor(color_binary, cv2.RGB2GRAY)
gray = cv2.cvtColor(color_binary, cv2.COLOR_RGB2GRAY)
plt.imshow(gray)
plt.show()
plt.imshow(gray, cmap="gray"))
plt.imshow(gray, cmap="gray")
plt.show()
def sobel_test(img, ksize=5):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    plot_image_side_by_side(img, combined)
    
image_list = glob.glob("./test_images/*.jpg")
image_list
for image in image_list:
    sobel_test(image)
    
for image in image_list:
    image_read = mpimg.imread(image)
    sobel_test(image_read)
    
for image in image_list:
    image_read = mpimg.imread(image)
    plt.imshow(image_read)
    plt.show()
    sobel_test(image_read)
    
image_list[0]
image = mpimg.imread(image_list[0])
gradx = abs_sobel_thresh(image, orient="x", sobel_kernel=ksize, thresh=(20. 100))
gradx = abs_sobel_thresh(image, orient="x", sobel_kernel=ksize, thresh=(20, 100))
sobel_test(image)
for image in image_list:
    image_read = mpimg.imread(image)
    sobel_test(image_read)
    
sobel_test(mpimg.imread(image_list[5]))
image_list[0]
img_l = image_list[5]
img_l
img_l_read = mpimg.imread(img_l)
sobel_tes(img_l)
sobel_test(img_l)
sobel_test(img_l_read)
def sobel_test(image, ksize=5):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    plot_image_side_by_side(image, combined)
    
for image in image_list:
    image_read = mpimg.imread(image)
    sobel_test(image_read)
    
def hsv_test(image):
    color_binary = pipeline(image)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(color_binary)
    ax2.set_title('Combined Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
for image in image_list:
    image_read = mpimg.imread(image)
    hsv_test(image_read)
    
def sobel_plus_hsv_test(inp_img, ksize=5):
    image = pipeline(inp_img)
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    plot_image_side_by_side(image, combined)
    
for image in image_list:
    image_read = mpimg.imread(image)
    soble_plus_hsv_test(image_read)
    
for image in image_list:
    image_read = mpimg.imread(image)
    sobel_plus_hsv_test(image_read)
    
def sobel_plus_hsv_test(inp_img, ksize=5):
    image = pipeline(inp_img)
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    plot_image_side_by_side(inp_img, combined)
    
for image in image_list:
    image_read = mpimg.imread(image)
    sobel_plus_hsv_test(image_read)
    
def sobel_plus_hsv_test(inp_img, ksize=5):
    image = pipeline(inp_img)
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(inp_img)
    ax2.imshow(image)
    ax3.imshow(combined, cmap="gray")
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
for image in image_list:
    image_read = mpimg.imread(image)
    sobel_plus_hsv_test(image_read)
    
def sobel_plus_hsv_test(inp_img, ksize=5):
    image = pipeline(inp_img)
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(2, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(inp_img)
    ax2.imshow(image)
    ax3.imshow(gradx, cmap="gray")
    ax4.imshow(grady, cmap="gray")
    ax5.imshow(dir_binary, cmap="gray")
    ax6.imshow(combined, cmap="gray")
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
for image in image_list:
    image_read = mpimg.imread(image)
    sobel_plus_hsv_test(image_read)
    
def sobel_plus_hsv_test(inp_img, ksize=5):
    image = pipeline(inp_img)
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(inp_img)
    ax2.imshow(image)
    ax3.imshow(gradx, cmap="gray")
    ax4.imshow(grady, cmap="gray")
    ax5.imshow(dir_binary, cmap="gray")
    ax6.imshow(combined, cmap="gray")
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
for image in image_list:
    image_read = mpimg.imread(image)
    sobel_plus_hsv_test(image_read)
    
