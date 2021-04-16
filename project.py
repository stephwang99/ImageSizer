# pip install h5py
# pip3 install keras && pip3 install tensorflow
# pip3 install 

import cv2 as cv
import numpy as np
import imutils
import argparse
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims

# YOLO labels for identification in specific order
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", 
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator","book", "clock", 
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# Provided dictionary of actual dimensions
size_dict = {"apple" : (4, 4), "umbrella": (5, 20), "banana": (10, 5), "keyboard": (20, 12), "book": (15, 9), "scissors": (9, 7)}

# YOLO Model Classification
class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1
 
	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)
 
		return self.label
 
	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]
 
		return self.score

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

def detect_object(img_path):
    model = load_model('model.h5')

    img = load_img(img_path)
    width, height = img.size
    img =  load_img(img_path, target_size=(416,416))
    img =  img_to_array(img)
    img = img.astype('float32')
    img /= 255
    img = expand_dims(img, 0)
    p = model.predict(img)
    return classify_object(p)

def classify_object(predict):
    # define the anchors
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    # define the threshold for detected objects
    threshold = 0.7
    boxes = list()
    for i in range(len(predict)):
        # decode the output of the network
        boxes += decode_netout(predict[i][0], anchors[i], threshold, 416, 416)

    flabels, scores = list(), list()
    for box in boxes:
	    for i in range(len(labels)):
		    if box.classes[i] > threshold:
		    	flabels.append(labels[i])
		    	scores.append(box.classes[i]*100)
    print("labels", flabels, "scores", scores)
    return flabels

def midpoint(a, b):
	return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def calculate_dimensions(img_path):
    width = 6
    dim = []
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # edge detection
    gray = cv.GaussianBlur(gray, (7, 7), 0)
    edged = cv.Canny(gray, 50, 100)
    edged = cv.dilate(edged, None, iterations=1)
    edged = cv.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    for contour in cnts:
        # if the contour is too small, ignore it
        if cv.contourArea(contour) < 100:
            continue
        # compute the rotated bounding box of the contour
        box = cv.minAreaRect(contour)
        box = cv.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points
        box = perspective.order_points(box)
        (pointA, pointB, pointC, pointD) = box

        # compute the Euclidean distance between the midpoints
        w = dist.euclidean(midpoint(pointA, pointB), midpoint(pointC, pointD))
        h = dist.euclidean(midpoint(pointA, pointD), midpoint(pointB, pointC))

        # initialize pixelsPerMetric
        if pixelsPerMetric is None:
            pixelsPerMetric = h / width
            # compute the size of the object
            dimX = w / pixelsPerMetric
            dimY = h / pixelsPerMetric
            dim.append((dimX,dimY))
            break
    return dim, pixelsPerMetric

def resize_object(img_path, dim, pixels):
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

    width = int(dim[0] * pixels)
    height = int(dim[1] * pixels)

    resized = cv.resize(img, (width,height), interpolation = cv.INTER_AREA)
    # Crop img
    resize_img = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    th, threshed = cv.threshold(resize_img, 240, 255, cv.THRESH_BINARY_INV)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)

    cnts = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv.contourArea)[-1]

    x,y,w,h = cv.boundingRect(cnt)
    resize_img = resize_img[y:y+h, x:x+w]
    return resize_img

parser = argparse.ArgumentParser(description='Process image  path')
parser.add_argument('--path', type=str)
args = parser.parse_args()

obj_path = "./banana.jpeg"

if args.path != None:
    obj_path = args.path

obj =  detect_object(obj_path)
print("obj", obj)

dim, pixels = calculate_dimensions("./desk_background.jpeg")
print("dim", dim)

resize_img = resize_object(obj_path, size_dict[obj[0]], pixels)

final_img = cv.imread("./desk_background.jpeg")
final_img = cv.cvtColor(final_img, cv.COLOR_BGR2GRAY)
print("resize_img size", resize_img.shape)
print("final_img size", final_img.shape)

def click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        print("click L", x, y)
        for i in range(resize_img.shape[0]):
            for j in range(resize_img.shape[1]):
                if int(resize_img[i,j]) <= 250:
                    if  i+y < final_img.shape[0] and j+x < final_img.shape[1]:
                        final_img[i+y, j+x] = resize_img[i,j]
    cv.imshow("img", cv.cvtColor(final_img, cv.COLOR_GRAY2RGB))

# Place img on mouse click
cv.namedWindow("img")
cv.imshow("img", cv.cvtColor(final_img, cv.COLOR_GRAY2RGB))
cv.setMouseCallback("img", click)

cv.waitKey(0)