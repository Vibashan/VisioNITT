import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('612_small.png',cv.IMREAD_GRAYSCALE)          # queryImage

out = cv.VideoWriter('cxuv.mp4',cv.VideoWriter_fourcc(*'MP4V'), 17, (700,600))
img_array = []

for i in range(1070,1200):	
	img2 = cv.imread('./xuv back/%s'%i+'.png',cv.IMREAD_GRAYSCALE)
	img2 = img2[:,0:500]

#for i in range(330,460):
#	img2 = cv.imread('./swift back/%s'%i+'.png',cv.IMREAD_GRAYSCALE)
	
	# Initiate SIFT detector
	sift = cv.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	# Need to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in range(len(matches))]
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.725*n.distance:
			matchesMask[i]=[1,0]
	draw_params = dict(matchColor = (0,255,0),
					singlePointColor = None,
					matchesMask = matchesMask,
					flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
	img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
	height, width, layers = img3.shape

	lol = float(np.sum(np.array(matchesMask))/100)*1.7
	cv.putText(img3,"SIM:" + str(lol), (int(width/2),int((3*height)/4)), cv.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2)
	
	fimg = cv.resize(img3,(700,600))
	img_array.append(fimg)
for i in range(len(img_array)):
	out.write(img_array[i])
out.release()
	#cv.imshow('img',fimg)
	#cv.waitKey(50)
