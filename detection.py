import numpy as np 
import cv2
import getdata
import pickle
import re
'''
step by step 
1 : train SVM model to detecter
'''

WINDOW = (30,60)
BLOCK = (20,20)
BLOCK_STRIDE = (10,10)
CELL = (10,10)
NBIN = 9

HUMAN_DAT      =  "/home/thangnx/HD/code/data/human/"
NONHUMAN_DAT   =  "/home/thangnx/HD/code/data/non-human/"

SVM_MODEL = "model.xml"

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

'''
classifier human/non-human for the hog detector

'''

def trainSVM():
	model = cv2.SVM()
	params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C = 1)

	trainingdata = np.array([])
	labels = np.array([])

	pos_imgs = getdata.getfiles(HUMAN_DAT)
	pos_hog = compute_hog(pos_imgs)
	pos_labels = np.ones((len(pos_imgs),1), dtype = np.int32)

	neg_imgs = getdata.getfiles(NONHUMAN_DAT)
	neg_hog = compute_hog(neg_imgs)
	neg_labels = np.zeros((len(neg_imgs), 1), dtype = np.int32)

	trainingdata = np.vstack([pos_hog,neg_hog])
	labels = np.vstack([pos_labels,neg_labels])

	model.train(trainingdata, labels, params = params)
	model.save(SVM_MODEL)
#	print np.asarray(model)

	#convert xml to pickle

	import xml.etree.ElementTree as ET 
	tree = ET.parse(SVM_MODEL)
	root = tree.getroot()
	
	SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0] 
	rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
	svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
	svmvec.append(-rho)
	pickle.dump(svmvec, open("svm.pickle", 'w'))



'''
compute hog feature of all images
gradients: training data for SVM model 
'''
def compute_hog(imgs):
	gradients = []
	hog = cv2.HOGDescriptor(WINDOW,BLOCK, BLOCK_STRIDE, CELL, NBIN)
	#print "hog computing......."
	for img in imgs:
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#img = img.reshape((1,col*row))
		#print img.shape
		grad = hog.compute(img)
		#print grad
		#print grad.shape

		if grad.shape[1] == 1 :
			# reshape grad to 1*d array 
			pass

		gradients.append(grad)

	return np.array(gradients)

'''
def getSVMmodel():
	import xml.etree.ElementTree as ET 
	tree = ET.parse(SVM_MODEL)
	root = tree.getroot()
	SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0] 
	rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
	svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
	svmvec.append(-rho)
	pickle.dump(svmvec, open("svm.pickle", 'w'))
'''

trainSVM()

model = pickle.load(open("svm.pickle"))

#model = cv2.SVM()
#model.load(SVM_MODEL) 

#print "****"
model = np.array(model, dtype = np.float32)
model = model.reshape((model.shape[0],1))
#print model.shape
hog = cv2.HOGDescriptor(WINDOW,BLOCK, BLOCK_STRIDE, CELL, NBIN)
hog.setSVMDetector(model)
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector() )


cam = cv2.VideoCapture("video.mp4")

while(cam.isOpened()):
	ret, frame = cam.read()
	frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#cv2.imshow("video",frame)

	found, w = hog.detectMultiScale(frame1)
	print found
#	print "w"
#	print w
	draw_detections(frame,found)
	cv2.imshow("output",frame)

	if cv2.waitKey(10) == 27:
		break


cv2.DestroyAllWindows()





