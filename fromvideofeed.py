#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import operator
import dlib
import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
import cvutils
import csv
from PIL import Image,ImageStat, ImageEnhance
import math
from matplotlib import pyplot as plt
import csv


#For Equalizing a frame...equalizing is not helpful like normalizing....not being used
def equalize(im):
    equ = cv2.equalizeHist(im)
    return(equ)

#For Performing morphological closing...Won't help....not being used
def closing(gray):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    return(res)


#Next two functions are For trying out gabor filters...not being used
#BUILD Filter
def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters


#Apply Filter
def process(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

#Similar to equalization..not being used
def clacheequalize(im):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(im)
    return(cl1)

#This function is for NORMALIZING the image before calculating the entropy.....->play around with the enhancement values and the threshold to make normalization better
def normalize(im):                      
    im=Image.fromarray(im)
    im=im.convert('L')
    contr = ImageEnhance.Contrast(im)
    im = contr.enhance(1)      #1
    bright = ImageEnhance.Brightness(im)
    im = bright.enhance(2.9)    #3
    #im.show()
    return np.array(im)

#same normalization different enhancement values for a different feature
def normalize2(im):                      
    im=Image.fromarray(im)
    im=im.convert('L')
    contr = ImageEnhance.Contrast(im)
    im = contr.enhance(.9)      #1
    bright = ImageEnhance.Brightness(im)
    im = bright.enhance(1.9)    #3
    #im.show()
    return np.array(im)

#same normalization different enhancement values for a different feature
def normalize3(im):                      
    im=Image.fromarray(im)
    contr = ImageEnhance.Contrast(im)
    im = contr.enhance(1)      #1
    bright = ImageEnhance.Brightness(im)
    im = bright.enhance(3)    #3
    #im.show()
    return np.array(im)

#This function is for calculating the entropy VALUE of the image
def ent2(im):
    """calculate the entropy of an image"""
    im=Image.fromarray(im)
    histogram = im.histogram()
    histogram_length = sum(histogram)
    samples_probability = [float(h) / histogram_length for h in histogram]
    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])
   
#The next two functions are used to find the entropy heat map for an image....dealt with in the other code..not being used
def entropyhelp(signal):
   lensig=signal.size
   symset=list(set(signal))
   numsym=len(symset)
   propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
   ent=np.sum([p*np.log2(1.0/p) for p in propab])
   return ent

def entropy(signal):
   
   N=5
   S=signal.shape
   E=np.array(signal)
   for row in range(S[0]):
      for col in range(S[1]):
         Lx=np.max([0,col-N])
         Ux=np.min([S[1],col+N])
         Ly=np.max([0,row-N])
         Uy=np.min([S[0],row+N])
         region=signal[Ly:Uy,Lx:Ux].flatten()
         E[row,col]=entropyhelp(region)
         return E

#CONVERTS the rgb format to ycbcr but gives NO VALUE (only for converting image for observation purpose)     
def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    # Y
    cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # Cb
    cbcr[:,:,1] = 128 - .169 * r - .331 * g + .5 * b
    # Cr
    cbcr[:,:,2] = 128 + .5 * r - .419 * g - .081 * b#changed from 2 to 1
    return np.uint8(cbcr)


#This function gives the MEAN value of ycbcr
def mycrcb(im):
   im=Image.fromarray(im)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   y= .299 * r + .587 * g + .114 * b
   cb= 128 - .169 * r - .331 * g + .5 * b
   cr= 128 + .5 * r - .419 * g - .081 * b
   return (y+cb+cr)/3

#returns the value of y of ycrcb
def yellow(im):
   im=Image.fromarray(im)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   y= .299 * r + .587 * g + .114 * b
   #print("CR is",128 + .5 * r - .419 * g - .081 * b)
   return (y)

#returns the value of cr of ycrcb
def cr(im):
   im=Image.fromarray(im)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   cr1= 128 + .5 * r - .419 * g - .081 * b
   return (cr1)

#returns the value of cb of ycrcb
def cb(im):
   im=Image.fromarray(im)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   cb1= 128 - .169 * r - .331 * g + .5 * b
   return (cb1)


#This function calculates the brightness value
def bright(im):
    im=Image.fromarray(im)
    greyscale_image = im.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

#Return mean rgb value
def mrgb(im):
   im=Image.fromarray(im)
   stat = ImageStat.Stat(im)
   r,g,b = stat.mean
   return ((r) + (g) + (b))/3,r,g,b
   

#To resize the image for faster detection
def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

#This function plots the facial landmark points....NOT BEING USED
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

camera = cv2.VideoCapture(0)

#predictor_path = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(predictor_path)

outfile = open('fakedata.csv', 'a') #write the name of the file to which data is to be written
i=0

print("Setting light levels") #Let the camera adjust to the light and wait for 50 frames before using the feed
while True:
    

    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break
    i+=1
    if(i<50):
        print("|",end ="")
        continue

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)


   


    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)
    print(len(dets))
    if (len(dets) > 0):
        for k, d in enumerate(dets):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            #shape = predictor(frame_resized, d)
            #shape = shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            #for (x, y) in shape:
            #   cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
            cv2.rectangle(frame, (int(d.left()/ratio), int(d.top()/ratio)),(int(d.right()/ratio), int(d.bottom()/ratio)), (0, 255, 0), 1)#rectangle around face is drawn
            if(int(d.top()/ratio)>=0 and int(d.bottom()/ratio)>=0 and int(d.left()/ratio)>=0 and int(d.right()/ratio)>=0 ):
                face=frame[int(d.top()/ratio): int(d.bottom()/ratio),int(d.left()/ratio):int(d.right()/ratio)]
                #facepil=Image.fromarray(face)
                newim=rgb2ycbcr(frame)
                normframecolor=normalize3(frame)
                normfacecolor=normframecolor[int(d.top()/ratio): int(d.bottom()/ratio),int(d.left()/ratio):int(d.right()/ratio)]
                ent=ent2(face)
                normim=normalize(frame)
                facenorm=normim[int(d.top()/ratio): int(d.bottom()/ratio)-70,int(d.left()/ratio):int(d.right()/ratio)]
                entnorm=ent2(facenorm)
                brt=bright(face)
                rgb,r,g,b=mrgb(face)
                ycrcb=mycrcb(face)
                faceycrcb=newim[int(d.top()/ratio): int(d.bottom()/ratio)-70,int(d.left()/ratio):int(d.right()/ratio)]
                normnewim=normalize2(faceycrcb)
                entycrcb=ent2(faceycrcb)
                entnormycrcb=ent2(normnewim)
                #and bright(face)>0.44 and bright(face)<0.56 and mrgb(face)>120 and mrgb(face)<160 and mycrcb(face)>120 and mycrcb(face)<145
                luma=yellow(face)
                crvalue=cr(face)
                cbvalue=cb(face)
                rgbnormalized,r2,g2,b2=mrgb(normfacecolor)
                entcolor=ent2(normfacecolor)

                
                if(luma>150):#<---- the THRESHOLD for live/fake face
                    cv2.putText(frame,"FAKE", (int(d.left()/ratio),int(d.bottom()/ratio)+30), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
                    print("FAKE (HIGH LUMA)")
                elif(entnorm>1):#<---- the THRESHOLD for live/fake face
                    cv2.putText(frame,"LIVE", (int(d.left()/ratio),int(d.bottom()/ratio)+30), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
                    print("LIVE (GOOD NORM ENTROPY)")
                elif(luma<=125 ):#and entycrcb>=1.8 and crvalue<=124):#<---- the THRESHOLD for live/fake face
                    cv2.putText(frame,"LIVE", (int(d.left()/ratio),int(d.bottom()/ratio)+30), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
                    print("LIVE")
                elif(entcolor>=2 and crvalue<=124):#<---- the THRESHOLD for live/fake face
                   cv2.putText(frame,"LIVE", (int(d.left()/ratio),int(d.bottom()/ratio)+30), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
                   print("LIVE")
                #elif(crvalue<=123):#<---- the THRESHOLD for live/fake face
                 #  cv2.putText(frame,"LIVE", (int(d.left()/ratio),int(d.bottom()/ratio)+30), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)   
                else:
                   cv2.putText(frame,"FAKE", (int(d.left()/ratio),int(d.bottom()/ratio)+30), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
                   print("FAKE")

                #write real or fake in the next line as first column based on what you are recording               
                #myData=(['real/fake',ent,entycrcb,entnorm,entnormycrcb,luma,crvalue,cbvalue,ycrcb,brt,r,g,b,rgb,entcolor,rgbnormalized,r2,g2,b2])
                #filew = csv.writer(outfile)
                #filew.writerow(myData)


                print("Entropy of normalized face is ", entnorm)
                print("Entropy of coloured normalized face is ", entcolor)
                print("Mean rgb of normalized face is", rgbnormalized)
                print("Entropy of normalized ycrcb face is", entnormycrcb)
                print("Entropy of ycrcb face is", entycrcb)
                print("luma value is ",  luma)
                print("Entropy is ", ent)
                print(" Brightness is ",brt)
                print("Mean rgb is ",rgb)
                print("Mean ycrcb is ",ycrcb)
                #cv2.imshow("cropped face", face)#save instead?
                #cv2.imshow("image in ycrcb", newim)
                #cv2.imshow("b new stream", normnewim)
                #cv2.imshow("normalized and cropped image for entropy", facenorm)
                cv2.imshow("normalized rgb", normframecolor)
                #continue
                #cv2.imshow("closed frame", closing(frame_grey))
                #cv2.imshow("clache_equalized frame", clacheequalize(frame_grey))
                #cv2.imshow("equalized frame", equalize(frame_grey))
                #cv2.imshow("normalized_equalized frame", normalize(equalize(frame_grey)))
                #filters = build_filters()                #For gabor filter
                #p = process(frame_grey, filters)
                #cv2.imshow("Gabor filtered", p)
                #cv2.imshow("resized", frame_resized)
                #cv2.imshow("normalized frame full", normim)
                continue
    cv2.imshow("image", frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):#PRESS Q TO EXIT CODE WHILE IT IS RUNNING
        cv2.destroyAllWindows()
        camera.release()
        outfile.close()
        break



