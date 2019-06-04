import cv2
import numpy
import scipy.signal

shells = cv2.imread('shells.jpg', 0)
cv2.imwrite('shells_gray.jpg', shells)

kernel_smooth = numpy.ones((3,3))/9
shells_smooth = scipy.signal.convolve2d(shells, kernel_smooth)
cv2.imwrite('shells_smooth.jpg', shells_smooth)

kernel_sharp = numpy.zeros((3,3)) - numpy.ones((3,3))/9
kernel_sharp[1,1] += 2
shells_sharp = scipy.signal.convolve2d(shells, kernel_sharp)
cv2.imwrite('shells_sharp.jpg', shells_sharp)

kernel_sobel_h = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
shells_sobel_h = scipy.signal.convolve2d(shells, kernel_sobel_h)
cv2.imwrite('shells_sobel_h.jpg', shells_sobel_h)

kernel_sobel_v = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
shells_sobel_v = scipy.signal.convolve2d(shells, kernel_sobel_v)
cv2.imwrite('shells_sobel_v.jpg', shells_sobel_v)