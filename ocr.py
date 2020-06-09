import pandas as pd 
import urllib.request
import sys
from PIL import Image as Image
import numpy as np
import cv2 as cv
import sys
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import os
from statistics import mean 
import pytesseract as tess	
import shutil  
import GSheets as gs
import re
Directory='images/'
images = os.listdir(Directory)
final_images = {}
final_images['Image_name'] = []
final_images['Id'] = []
print(len(images))

def show_image(name,image):
	cv.imshow(name,image)
	cv.waitKey(0)
	cv.destroyWindow(name)

def import_images(links_path,uuids_path):
    in_df = pd.read_csv(uuids_path)#'input2.csv'
    df = pd.read_csv(links_path)#'hesham@uber.com_1587908514.csv'
    merged_df = in_df.merge(df,left_on = 'uuid',right_on = 'UUID')
    for uuid,link,format in zip(merged_df['UUID'],merged_df['Doc Link'],merged_df['format']):
        urllib.request.urlretrieve(link, 'images/'+uuid+'.'+format)

def detect_face(image):
	img = image.copy()
	face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	XstartF,YstartF = 0,0
	XendF,YendF = 0,0
	area = 0 
	for (x, y, w, h) in faces:
		cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
		if w*h >area:
			area = w*h
			XendF,YendF = x+w,y+h
			XstartF,YstartF = x,y
	return XstartF,YstartF,XendF,YendF,area,img


def detect_text(image):
	args={}
	args["width"] = 320
	args["height"] = 320
	args["east"] = 'frozen_east_text_detection.pb'
	args["min_confidence"] = 0.5
	# load the input image and grab the image dimensions
	# image = cv.imread(args["image"])
	orig = image.copy()
	(H, W) = image.shape[:2]

	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (args["width"], args["height"])
	rW = W / float(newW)
	rH = H / float(newH)

	# resize the image and grab the new image dimensions
	image = cv.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	net = cv.dnn.readNet(args["east"])

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()

	# show timing information on text prediction
	print("[INFO] text detection took {:.6f} seconds".format(end - start))

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# loop over the bounding boxes
	xs,ys,xe,ye =sys.maxsize,sys.maxsize, 0,0
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# draw the bounding box on the image
		cv.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		# print(startX,startY,endX,endY)
		xs,ys,xe,ye = min(startX,xs),min(startY,ys),max(endX,xe),max(endY,ye)

	# show the output image
	# cv.imshow("Text Detection", orig)
	# cv.waitKey(0)
	return xs,ys,xe,ye,orig
def rotate_bound(img, angle):
    # grab the dimensions of the image and then determine the
    # center
    image = img.copy()
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))

def crop_image(img,x1,y1,x2,y2):
#     img = cv.imread(image_path)
    cropped = img[y1:y2, x1:x2]
    return cropped
# import_images('hesham@uber.com_1587908514.csv','input2.csv')

def get_angle(image,image_name,w,h,box,flag):
	ratio = 10
	print(w,h,box)
	try:
		# print('cropping')
		cropped = crop_image(image,box[0],box[1],box[2],box[3]) ## crop text area to check for rotation
		gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

		cv.imwrite('cropped/'+image_name,gray)

		# show_image('cropped',cropped)
	except:
		print('CROP FAILURE.......')
		# shutil.move('images/'+image_name,'FReading/'+image_name)

		return 0
	try:
		im = Image.open('cropped/'+image_name)
		# show_image('getting_angle',cropped)
		print('APPLYING TESSERACT FOR ANGLE FETCHING.....')

		newdata=tess.image_to_osd(im)
		angle = int(re.search('(?<=Rotate: )\d+', newdata).group(0))
		print('angle:', angle)
		return angle
	except:
		if flag:
			startXtext,startYtext,endXtext,endYtext,text_image = detect_text(image)
			box =[startXtext,startYtext,endXtext,endYtext]
			flag = False
			# show_image('text_detection_image',text_image)
			cv.imwrite('facetext/Text/'+image_name,text_image)
			# return get_angle(image,image_name,w,h,box,flag)

		if box[0]>0: box[0]= max(box[0] - ratio,0)
		else: box[0]=0

		if box[1]>0: box[1]= max(box[1] - ratio,0)
		else: box[1]=0

		if box[2]<w: box[2]= min(box[2] + ratio,w)
		else: box[2]=w

		if box[3]<h: box[3]= min(box[3] + ratio,h)
		else: box[3]=h

		if box[0]==0 and box[1]==0 and box[2]==w and box[3]==h:
			# print('Donee')
			return 0
		else:
			return get_angle(image,image_name,w,h,box,flag)
		# print('box',box)

def get_id(image_path):
	text = tess.image_to_string(Image.open(image_path))
	words = text.split()
	print (words)
	for word in words:
		# print(word)
		if word.isdigit():
			if len(word)==14:
				print('ID : ',word)
				return word,True
	return 0,False

def Failure_Handling(image):
	XstartF,YstartF,XendF,YendF,area,face_image = detect_face(image)
	startXtext,startYtext,endXtext,endYtext,text_image = detect_text(face_image)

	pre_final_image = text_image

	show_image('prefinal_image',pre_final_image)
	cv.imwrite('facetext/'+image_name,pre_final_image)
	cv.imwrite('facetext/Face/'+image_name,face_image)


	startX = min(XstartF,startXtext)
	startY = min(YstartF,startYtext)
	endX = endXtext 
	endY = max(YendF,endYtext)
	try:
		cropped = crop_image(image,startX,startY,endX,endY)
		gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
		cv.imwrite('cropped/'+image_name,gray)
		cv.imwrite('facetext/cropped/'+image_name,gray)
		# show_image('cropped',cropped)

		id,flag = get_id('cropped/'+image_name)
		if flag:
			final_images['Image_name'].append(image_name)
			final_images['Id'].append(id)
		else:
			print('ID NOT FOUND.......')
			# shutil.move('images/'+image_name,'FDetection/'+image_name)

	except:
		print('CROP FAILURE......')
		# shutil.move('images/'+image_name,'FReading/'+image_name)

for i in range(1):

	image_name = '3c6d1c6a-add4-48dc-b03f-094b1dfdad36.jpeg'#'3c6b5658-46b5-4761-b559-b961c9d39ada.jpeg'#'05d75378-cf24-4c86-9e34-ea6c31415eab.jpeg'#images[i]#'05e2b636-b091-473e-b1b8-273039980da7.jpeg'#'05d75378-cf24-4c86-9e34-ea6c31415eab.jpeg'#images[i]#'05e2b636-b091-473e-b1b8-273039980da7.jpeg'#images[i]#'05ea52fe-0258-42e4-bb6b-b69654bf6a22.jpeg'#'05e711a2-3f7b-4df4-9552-853c0b982a4e.jpeg'#'05e85b37-c2a1-4266-b740-8e49d5dfce7c.jpeg'#'05e29e6b-e0d3-4419-bf52-be7b0f00173f.jpeg'#'05da6f12-92e2-4ad9-8d2a-263d7ae92e8c.jpeg'
	print(image_name)
	
	try:
		image =cv.imread(Directory+image_name)	
		# show_image('original_image',image)
		startXtext,startYtext,endXtext,endYtext,text_image = detect_text(image)
		box =[startXtext,startYtext,endXtext,endYtext]
		w,h = image.shape[:2]
		# box=[0,0,h,w]
		angle = get_angle(image,image_name,w,h,box,True)
		cv.imwrite('facetext/Text/'+image_name,text_image)
		print('returned: ',angle)

		try:
			if angle > 0:
				rotated_image = rotate_bound(image,angle)
				image = rotated_image
				# show_image('rotated',image)
				cv.imwrite('facetext/Rotated/'+image_name,image)

			gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
			cv.imwrite('output/'+image_name,gray)
			id,flag =get_id('output/'+image_name)
			if flag:
				final_images['Image_name'].append(image_name)
				final_images['Id'].append(id)

			
			if flag ==False:
				Failure_Handling(image)
		
			
		except:
			Failure_Handling(image)
	except:
		print('FAILED READING IMAGE.......')
		print('IMAGE NAME: ',image_name)	
	
		# shutil.move('images/'+image_name,'FReading/'+image_name)
		# print(newdata)

print('DETECTED: ', len(final_images['Id']),' IDs..........')
print(final_images)
final_images = pd.DataFrame.from_dict(final_images)
# final_images.to_csv('final.csv')
sheet_key = '1NIx9XpsqKCwX4Mel5auSyEsn_D1wSSvC3IspNSTIGNc'
# gs.publish_sheets_Woverwrite(df=final_images, key=sheet_key, target_cell='A1', sheet_name='Sheet4', head=True,fit=False)


