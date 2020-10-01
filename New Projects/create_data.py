import cv2,sys,numpy,os
hf='haarcascade_frontalface_default.xml'
datasets='datasets'
sub_data='deb' #Type a persons name
#creating folder inside the data set file 
path=os.path.join(datasets,sub_data)

if not os.path.isdir(path):
    os.mkdir(path)
# width and height of the image stored   
(width,height)=(270,400)
#referencing all the facial features using hf
face_cas=cv2.CascadeClassifier(hf)
#accesing web cam to capture picture,0=>inbuilt camera access
web=cv2.VideoCapture(0)

count=1

while count<50:
    (_,im)=web.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    face=face_cas.detectMultiScale(gray,1.3,4)
    #
    for(x,y,w,h) in face:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        faces=gray[y:y+h,x:x+w]
        
        #cropping the face part out of the whole image   
        fce_re=cv2.resize(faces,(width,height))
        #saving the cropped image 
        cv2.imwrite('% s/% s.png' % (path,count),fce_re)
        count+=1
        cv2.imshow('OpenCV',im)
        key=cv2.waitKey(100)
        if key==40:
            break
cv2.destroyAllWindows()
