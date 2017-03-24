import numpy as np 
import cv2
import os

def getfiles(path):
	file = []

	for name in os.listdir(path):
		#print name
		img = cv2.imread(path+name,0)
		#cv2.imshow("before",img)
		#img = cv2.resize(img, (60,30), interpolation = cv2.INTER_AREA)
		#cv2.imshow("img", img) 
		file.append(img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		#print img.shape

	return file

def writeimage(filepath, img):
	files = getfiles(filepath)
	name = len(files)

	cv2.imwrite(filepath + str(name + 1)+".jpg",img)


def draw(event, x, y, flags, params):
	global ix, iy, drawing
	
	if event == cv2.EVENT_LBUTTONDOWN:
		#cv2.circle(img, (x,y), 100, (255,0,0), -1)
		#drawing = True
		ix,iy = x,y
	elif event == cv2.EVENT_MOUSEMOVE:
		pass

	elif event == cv2.EVENT_LBUTTONUP:
		cv2.rectangle(frame, (ix,iy),(x,y),(0,200,0),0)
		up, down, left,right = 0,0,0,0
		if ix > x:
			up, down = x,ix
		else:
			up,down = ix,x

		if iy > y:
			left,right = y,iy
		else:
			left,right = iy,y


		dat = frame[left:right,up:down,:]
		#print dat.shape
		
		cv2.imshow("data",dat)
		dat = cv2.resize(dat, (30,60), interpolation = cv2.INTER_AREA)
		dat = cv2.cvtColor(dat, cv2.COLOR_BGR2GRAY)


		writeimage("/home/thangnx/HD/code/data/non-human/",dat)
'''
def getdata(video_name):
	cam = cv2.VideoCapture(video_name)
	cv2.namedWindow("get_data_from_video")

	while(cam.isOpened()):
		ret,frame = cam.read()
		cv2.setMouseCallback("get_data_from_video",draw)

		k = cv2.waitKey(0)

		if k == 'n':
			continue
'''

'''
cam = cv2.VideoCapture("video.mp4")

while(cam.isOpened()):
	ret,frame = cam.read()
	cv2.namedWindow("video")
	cv2.imshow("video",frame)

	cv2.setMouseCallback("video",draw)

	if cv2.waitKey(0) & 0xFF == ord('n'):
		continue

	if cv2.waitKey(0) & 0xFF == ord('q') :
		break



cam.release()
cv2.destroyAllWindows()
'''
#img = cv2.imread("download.jpg",1)
#img = cv2.imread("/home/thangnx/HD/code/data/human/1.jpg",1)
#cv2.namedWindow("anh")
#cv2.imshow("anh", img)
#print img.shape

#cv2.waitKey(0)
#cv2.destroyAllWindows()

#cv2.setMouseCallback("anh", draw)

#while 1 :
#	cv2.imshow("anh", img)
#	if cv2.waitKey(20) & 0xFF == 27:
#		break

#cv2.waitKey(0)
#cv2.destroyAllWindows()

#file =  getfiles("/home/thangnx/HD/code")
#print file