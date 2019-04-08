import warnings
warnings.filterwarnings(action='ignore')

import cv2
import keras
import numpy
import scipy.stats
import multiprocessing
import warnings
import collections
import shapelib

warnings.filterwarnings(action='ignore')

method = [('neighborhood', 6, 6),
          ('neighborhood', 8, 2),
          ('neighborhood', 10, 2),
          ('neighborhood', 10, 3),
          ('contour_portion', 5, 6),
          ('contour_portion', 20, 2),
          ('angle', 5, 4),
          ('angle', 10, 2),
          ('angle', 15, 7),
          ('angle', 20, 5),
          ('angle_plus', 5, 6),
          ('angle_plus', 25, 7)]

print('Descriptor size: %d' % (sum([n+1 for _,_,n in method])))

stack = [shapelib.ContourDescriptor(mode=m[0], params=(m[1],), neurons=m[2]) for m in method]
descriptor = shapelib.StackedContourDescriptor(stack)

# load json and create model
with open('model_s.json', 'r') as json_file:
    model_json = json_file.read()
    model_s = keras.models.model_from_json(model_json)
# load weights into new model
model_s.load_weights("model_s.h5")

# load json and create model
with open('model_g.json', 'r') as json_file:
    model_json = json_file.read()
    model_g = keras.models.model_from_json(model_json)
# load weights into new model
model_g.load_weights("model_g.h5")

video = cv2.VideoCapture(0)
# Set properties. Each returns === True on success (i.e. correct resolution)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

kernel_1 = numpy.ones((3,3), numpy.uint8)
kernel_2 = numpy.ones((5,5), numpy.uint8)
kernel_3 = numpy.ones((1,5), numpy.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
TopLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (0,255,0)
lineType               = 2

g_ = collections.deque(maxlen=10)

class_ = {
    -1: ' ',
     0: '0',
     1: '1',
     2: '2',
     3: '3',
     4: 'hang loose'
}

descriptor_size = 8

counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = video.read()

    # Draw region of interest (a 300x300 square region) + small 2px border
    cv2.rectangle(frame, (50-2, 50-2), (350+2, 350+2), (0,255,0), 2)

    # Get region of interest (roi), save a raw version (raw)
    roi = frame[50:350,50:350]
    raw = roi.copy()

    roi = cv2.resize(roi, (0,0), fx=0.5, fy=0.5)
    h, w, d = roi.shape
    x = roi.reshape(roi.shape[0]*roi.shape[1], 3)

    y = model_s.predict(x)

    segm = y[:,0]
    idx = numpy.argwhere(segm < 0.7)
    segm[idx] = 0
    segm = segm*255
    segm = segm.reshape(roi.shape[0], roi.shape[1], 1)

    segm = cv2.erode(segm, kernel_1, iterations=1)
    segm = cv2.morphologyEx(segm, cv2.MORPH_CLOSE, kernel_2)
    segm = cv2.dilate(segm, kernel_1, iterations=2)
    
    segm_ = segm.copy()
    segm_ = cv2.cvtColor(segm_, cv2.COLOR_GRAY2BGR)

    g = -1
    segm = segm.astype(numpy.uint8)
    _, segm = cv2.threshold(segm, 127, 255, 0)
    contours, _ = cv2.findContours(segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours = [cnt for cnt in contours if numpy.linalg.norm(numpy.mean(cnt, axis=0) - [w/2, h/2]) < 50]
    if len(contours) > 0:
        main_contour = max(contours, key=lambda x:len(x))
        main_contour = numpy.reshape(main_contour, (len(main_contour), 2))
        if len(main_contour) > 100:
            cv2.drawContours(segm_, [main_contour], 0, (0,255,0), 2)
            features = descriptor.extract_contour_features(contour=main_contour)
            # main_contour = main_contour.reshape(main_contour.shape[0], 1, main_contour.shape[1])
            # main_contour = numpy.asarray(main_contour, numpy.float32)
            # features = cv2.ximgproc.fourierDescriptor(main_contour, 0, int(descriptor_size/2)).flatten()
            out = model_g.predict(numpy.array([features]))
            g_.append(numpy.argmax(out))
            g = scipy.stats.mode(g_)[0][0]
            
    # Display the resulting segmentation
    segm_ = cv2.resize(segm_, (0,0), fx=2.0, fy=2.0)
    # print(frame.shape)
    # print(segm_.shape)
    frame[50:350,50:350] = segm_
    cv2.putText(frame, class_[g], TopLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.imshow('F', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    counter += 1

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()