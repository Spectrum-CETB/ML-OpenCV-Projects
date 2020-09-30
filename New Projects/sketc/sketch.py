import cv2
import numpy as np
#insert your image
img=cv2.imread("akp.jpg")
scale=0.40
height=int(img.shape[0]*scale)
width=int(img.shape[1]*scale)
#step-1:resize the large image
dim=(width,height)
resize=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
cv2.imshow("photo",resize) #SHOW RESIZED IMAGE
#step-2:sharpen the resized image
kernel=np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
sharp=cv2.filter2D(resize,-1,kernel)
#cv2.imshow("sharp",sharp) see the sharpen image

#apply bgr to hsv
#object=cv2.cvtColor(sharp,cv2.COLOR_BGR2HSV)
#cv2.imshow("sharp",object) see the hsv image

#step-3:get the sketch from inversing images
#a)inversion on gray image
gray=cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
inv=255-gray
#apply bluring
blur=cv2.GaussianBlur(src=inv,ksize=(15,15),sigmaX=0,sigmaY=0)
#draw sketch
sketch=cv2.divide(gray,255-blur,scale=256)
cv2.imshow("sketch",sketch)
cv2.waitKey(0)
cv2.destroyAllWindows()
