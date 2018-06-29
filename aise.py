import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import models


image_path=r'D:\Deep_Learning\codes\depth\a (4).jpg'
model_data_path=r'D:\Deep_Learning\codes\depth\NYU_FCRN.ckpt'




def predictor():

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

   
    # Read image
    img = Image.open(image_path)
    actualwidth,actualheight = img.size
    
    img = img.resize([width,height], Image.ANTIALIAS)
    imgsaved = img.resize([160,128], Image.ANTIALIAS)
    
    imgsaved = np.array(imgsaved).astype('float32')

    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})

        #mycode
        outimage=pred[0,:,:,0]
        xmax, xmin = outimage.max(), outimage.min()
        outnormimage = (outimage - xmin)/(xmax - xmin)
        outnorminvimage = np.ones((np.shape(outnormimage)))- outnormimage
        #outnorminvimage = (outnorminvimage - 0.7)
        #outnorminvimage[outnorminvimage<0] = 0
        #outnorminvimage = outnorminvimage * 3
        outnorminvimage = np.power(outnorminvimage,2)
        print(outnorminvimage[60,60])
        outnorminvimage = outnorminvimage[:,:,np.newaxis]
        imgsaved1 = np.multiply(imgsaved[:,:,:],outnorminvimage[:,:,:])

        #Image.fromarray(np.uint8(imgsaved1)).show()

        #fig = plt.figure()
        #jj=plt.imshow(outnorminvimage[:,:,0])
        #fig.colorbar(jj)
        #plt.show()
        print(np.shape(outnorminvimage))
        fig = plt.figure("basic image into filter in low res")
        kk=plt.imshow(Image.fromarray(np.uint8(imgsaved1)))
        

        
        #extras
        reimagefilter=Image.fromarray(np.uint8(outnorminvimage[:,:,0]*255))
        resizedfilter = reimagefilter.resize([actualwidth,actualheight], Image.ANTIALIAS)
        resizedfilter = np.array(resizedfilter).astype('float32')
        resizedfilter = resizedfilter / 255
        resizedfilter = resizedfilter[:,:,np.newaxis]
        for x in range(3,actualheight-3):
         for y in range(3,actualwidth-3):
              resizedfilter[x,y,0]=np.mean(resizedfilter[x-2:x+2,y-2:y+2,0])
              
        #mutiplied darkening    
        img2 = Image.open(image_path)
        img2 = img2.resize([actualwidth,actualheight], Image.ANTIALIAS)
        img2 = np.array(img2).astype('float32')
        imgsaved2 = np.multiply(img2[:,:,:],resizedfilter[:,:,:])
        #Image.fromarray(np.uint8(imgsaved2)).show()
        fig = plt.figure("darken filter into image in high res")
        kkiss=plt.imshow(Image.fromarray(np.uint8(imgsaved2)))
        

        #hard darkening
        bbc=img2.copy()
        abc=img2.copy()
        for x in range(10,actualheight-11):
         for y in range(10,actualwidth-10):      
          g=0
          if resizedfilter[x,y,0]<0.3:
              g=5
          elif resizedfilter[x,y,0]<0.5:
              g=3
          if g>0:    
              bbc[x,y,0]=abc[x,y,0]-g
              bbc[x,y,1]=abc[x,y,1]-g
              bbc[x,y,2]=abc[x,y,2]-g
          if g==0:
              bbc[x,y,0]=abc[x,y,0]+5
              bbc[x,y,1]=abc[x,y,1]+5
              bbc[x,y,2]=abc[x,y,2]+5
        fig = plt.figure("hard darken image high res")
        kbkiss=plt.imshow(Image.fromarray(np.uint8(bbc)))
        
              


        
        #extra2
        img3 = Image.open(image_path)
        img3 = img3.resize([actualwidth,actualheight], Image.ANTIALIAS)
        img3 = np.array(img3).astype('float32')
        bb=img3.copy()
        print(resizedfilter[10,10,0])
        
        
        for x in range(10,actualheight-11):
         for y in range(10,actualwidth-10):      
          f=0
          if resizedfilter[x,y,0]<0.3:
              f=3
          elif resizedfilter[x,y,0]<0.5:
              f=2
          if f>0:    
              bb[x,y,0]=np.mean(img3[x-f:x+f,y-f:y+f,0])
              bb[x,y,1]=np.mean(img3[x-f:x+f,y-f:y+f,1])
              bb[x,y,2]=np.mean(img3[x-f:x+f,y-f:y+f,2])
        #bb=np.multiply(bb[:,:,:],resizedfilter[:,:,:])  
        #Image.fromarray(np.uint8(bb)).show()
        fig = plt.figure("blurred image high res")
        bkkiss=plt.imshow(Image.fromarray(np.uint8(bb)))

        fig = plt.figure("actual image high res")
        bkkiss=plt.imshow(Image.fromarray(np.uint8(img3)))
        




        #dark and blur both
        
        cc=imgsaved2.copy()       ###take the already darkened photo
        dd=imgsaved2.copy()
        print(resizedfilter[10,10,0])
        
        #for x in range(10,actualheight-11):
         #for y in range(10,actualwidth-10):
              #resizedfilter[x,y,0]=np.mean(resizedfilter[x-1:x+1,y-1:y+1,0])
        for x in range(10,actualheight-11):
         for y in range(10,actualwidth-10):      
          f=0
          if resizedfilter[x,y,0]<0.3:
              f=2
          elif resizedfilter[x,y,0]<0.5:
              f=1
          if f>0:    
              dd[x,y,0]=np.mean(cc[x-f:x+f,y-f:y+f,0])
              dd[x,y,1]=np.mean(cc[x-f:x+f,y-f:y+f,1])
              dd[x,y,2]=np.mean(cc[x-f:x+f,y-f:y+f,2])
          if f==0:
              dd[x,y,0]=cc[x,y,0]+5
              dd[x,y,1]=cc[x,y,1]+5
              dd[x,y,2]=cc[x,y,2]+5
        #bb=np.multiply(bb[:,:,:],resizedfilter[:,:,:])  
        #Image.fromarray(np.uint8(bb)).show()
        fig = plt.figure("blurred and dark image high res")
        pussy=plt.imshow(Image.fromarray(np.uint8(dd)))

        #fig = plt.figure("actual image high res")
        #bkkiss=plt.imshow(Image.fromarray(np.uint8(img3)))
        


        
        # Plot result
        fig = plt.figure("actual depth estimate")
        iih = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(iih)
        plt.show()

        
        
predictor()

