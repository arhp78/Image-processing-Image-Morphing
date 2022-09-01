# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 10:11:10 2021

@author: hatam
"""

import numpy as np
import cv2
import dlib
from imutils import face_utils
from PIL import Image
from scipy.spatial import Delaunay
import os
import ffmpeg

def Keypoint(img):
    size=img.shape
    p = "shape_predictor_68_face_landmarks.dat"
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

       
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    # For each  face
    
    for (i, rect) in enumerate(rects):
      
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        
        points = [];
        for (x, y) in shape:
            points.append((x, y))
            #cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
           
    points.append((1,1))
    points.append((size[1]-1,1))
    points.append(((size[1]-1)//2,1))
    points.append((1,size[0]-1))
    points.append((1,(size[0]-1)//2))
    points.append(((size[1]-1)//2,size[0]-1))
    points.append((size[1]-1,size[0]-1))
    points.append(((size[1]-1),(size[0]-1)//2))
    #cv2.imwrite("Output.jpg", img) 
    
    return points
    # Show the image
    #cv2.imwrite("Output.jpg", img1)
def rescale_image (img1 , img2):
    size1 = img1.shape
    size2 = img2.shape
    if(size1[0]>=size2[0] and size1[1]>=size2[1]):
         scale1 = size1[0]/size2[0]
         scale0 = size1[1]/size2[1]
         res = cv2.resize(img2,None,fx=scale0,fy=scale1,interpolation=cv2.INTER_AREA)
         return img1 , res
    elif(size1[0]<size2[0] and size1[1]<size2[1]):
         scale1 = size2[0]/size1[0]
         scale0 = size2[1]/size1[1]
         res = cv2.resize(img1,None,fx=scale0,fy=scale1,interpolation=cv2.INTER_AREA)
         return  res,img2
    elif(size1[0]>size2[0] and size1[1]<size2[1]):
         scale1 = size1[0]/size2[0]
         scale0 = size2[1]/size1[1]
         res2 = cv2.resize(img2,None,fx=scale0,interpolation=cv2.INTER_AREA)
         res1 = cv2.resize(img1,None,fy=scale1,interpolation=cv2.INTER_AREA)
         return  res1,res2   
    elif(size1[0]<size2[0] and size1[1]>size2[1]):
         scale1 = size2[0]/size1[0]
         scale0 = size1[1]/size2[1]
         res1 = cv2.resize(img1,None,fx=scale0,interpolation=cv2.INTER_AREA)
         res2 = cv2.resize(img2,None,fy=scale1,interpolation=cv2.INTER_AREA)
         return  res1,res2   

def affine_triengle(tringle1,tringle2,tringle3,img1,img2,alpha,size,morphed_image_final):
        morphed_image=np.zeros_like(img1)
       
        
        
        dstTri = np.array( [[tringle3[0][0],tringle3[0][1]],[tringle3[1][0],tringle3[1][1]],[tringle3[2][0],tringle3[2][1]]]  ).astype(np.float32)
        srcTri = np.array( [[tringle1[0][0],tringle1[0][1]],[tringle1[1][0],tringle1[1][1]],[tringle1[2][0],tringle1[2][1]] ] ).astype(np.float32)
        
        warpMat1 = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
        dst1 = cv2.warpAffine(img1, warpMat1, (size[1], size[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        
        dstTri = np.array( [[tringle3[0][0],tringle3[0][1]],[tringle3[1][0],tringle3[1][1]],[tringle3[2][0],tringle3[2][1]]]  ).astype(np.float32)
        srcTri = np.array( [[tringle2[0][0],tringle2[0][1]],[tringle2[1][0],tringle2[1][1]],[tringle2[2][0],tringle2[2][1]] ] ).astype(np.float32)
        warpMat2 = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
        dst2 = cv2.warpAffine(img2, warpMat2, (size[1], size[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        
        mask=np.zeros_like(img1)
        triangle  = np.int32(dstTri)
        cv2.fillConvexPoly(mask, triangle, (1.0, 1.0, 1.0))
        
        mask1 = morphed_image_final == 0
        morphed_image=mask1*mask*((1-alpha)*dst1+alpha*dst2)
        morphed_image_final+=morphed_image
        
        return morphed_image_final



def morphing(img1,img2,n,points1,points2,triengle,size):
    img1=img1.astype('float32')
    img2=img2.astype('float32')
    
    
    for i in range(0,n):
     morphed_image_final=np.zeros_like(img1)
     alpha=i/(n+1)
     point=[]
     for j in range(0,len(points1)):
         x=(1-alpha)*points1[j][0]+alpha*points2[j][0]
         y=(1-alpha)*points1[j][1]+alpha*points2[j][1]
         point.append((x,y))
     #know we should find triengle in each image
     for j in range(0,len(triengle)):
         x=int(triengle[j][0])
         y=int(triengle[j][1])
         z=int(triengle[j][2])
         tringle1=[points1[x],points1[y],points1[z]]
         tringle2=[points2[x],points2[y],points2[z]]
         tringle3=[point[x],point[y],point[z]]
         morphed_frame=affine_triengle(tringle1,tringle2,tringle3,img1,img2,alpha,size,morphed_image_final)
         res = Image.fromarray(cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
         
     filename ="gif2/file-%d.png"%(i)
     res.save(filename)
     print( i," -th image")
        
         
         
    


img1 = cv2.imread("img.jpg")
img2 = cv2.imread("img_dst.jpg")

#rescale image 
imgf1,imgf2 =rescale_image(img1,img2)
#cv2.imwrite("out1.jpg", imgf1)
#cv2.imwrite("out2.jpg", imgf2)
#find key points & add 8 point to image

points1=Keypoint(imgf1)
points2=Keypoint(imgf2)
'''
for n in range(0, 76):
    x = points1[n][0]
    y = points1[n][1]
    cv2.circle(imgf1, (x, y), 5, (255, 0, 0), -1)
    
    cv2.imwrite("Output.jpg", imgf1)
'''
points3=np.zeros_like(points1)
for i in range(len(points1)):
    points3[i][0]=1/2*(points1[i][0]+points2[i][0])
    points3[i][1]=1/2*(points1[i][1]+points2[i][1])

tringle=Delaunay(points3)
tringle1=tringle.simplices

size=imgf2.shape
res=morphing(imgf1,imgf2,50,points1,points2,tringle1,size)

os.system('ffmpeg -i gif2/file-%d.png -r 10 -vcodec mpeg4 res2.MP4')
