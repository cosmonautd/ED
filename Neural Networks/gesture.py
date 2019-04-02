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

kernel_1 = numpy.ones((3,3), numpy.uint8)
kernel_2 = numpy.ones((5,5), numpy.uint8)
kernel_3 = numpy.ones((1,5), numpy.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
TopLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (0,255,0)
lineType               = 2

g_ = collections.deque(maxlen=10)

counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = video.read()
    frame = cv2.resize(frame, (0,0), fx=0.15, fy=0.15)
    h, w, d = frame.shape
    x = frame.reshape(frame.shape[0]*frame.shape[1], 3)

    y = model_s.predict(x)

    segm = y[:,0]
    idx = numpy.argwhere(segm < 0.2)
    segm[idx] = 0
    segm = segm*255
    segm = segm.reshape(frame.shape[0], frame.shape[1], 1)

    segm = cv2.erode(segm, kernel_1, iterations=1)
    segm = cv2.morphologyEx(segm, cv2.MORPH_CLOSE, kernel_2)
    segm = cv2.dilate(segm, kernel_1, iterations=2)
    
    segm_ = segm.copy()
    segm_ = cv2.cvtColor(segm_, cv2.COLOR_GRAY2BGR)

    g = -1
    segm = segm.astype(numpy.uint8)
    _, segm = cv2.threshold(segm, 127, 255, 0)
    contours, _ = cv2.findContours(segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if numpy.linalg.norm(numpy.mean(cnt, axis=0) - [w/2, h/2]) < 25]
    if len(contours) > 0:
        main_contour = max(contours, key=lambda x:len(x))
        main_contour = numpy.reshape(main_contour, (len(main_contour), 2))
        if len(main_contour) > 100:
            cv2.drawContours(segm_, [main_contour], 0, (0,255,0), 2)
            features = descriptor.extract_contour_features(contour=main_contour)
            out = model_g.predict(numpy.array([features]))
            g_.append(numpy.argmax(out))
            g = scipy.stats.mode(g_)[0][0]
            
    # Display the resulting segmentation
    # segm_ = cv2.resize(segm_, (0,0), fx=3, fy=3)
    cv2.putText(segm_, str(g), TopLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.imshow('S', segm_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    counter += 1

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()