# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random

def main(filename):
    if(filename == ""): 
        cam = cv2.VideoCapture(0)
        b, image = cam.read()
        cv2.imwrite("./image.jpg", image)
        fname = "./image.jpg"
        if b:    
            cv2.imshow("Taken by camera",image)
            k = cv2.waitKey(0) 
            if k == 27:
                cv2.destroyAllWindows()
            elif k == ord('w'):
                output_image(image)
            elif k == ord('g'):
                openCV_gray_scale(fname)
            elif k == ord('G'):
                algo_gray_scale(fname)
            elif k == ord('c'):
                cycle_image(fname)
            elif k == ord('s'):
                smooth_image(fname)            
            elif k == ord('x'):
                 convolution_x_derivative(fname)
            elif k == ord('y'):
                 convolution_y_derivative(fname)
            elif k == ord('m'):
                magnitude(fname)
            elif k == ord('r'):
                rotate_Q_angle(fname)
            elif k == ord('h'):
                help(filename)
            elif k == ord('S'):
                algo_smooth_image(filename)
            
         
    else:
       image = cv2.imread(filename)
       cv2.imshow('window',image)
       k = cv2.waitKey(0) 
       if k == 27:
            cv2.destroyAllWindows()
       elif k == ord('w'):
           output_image(image)
       elif k == ord('g'):
           openCV_gray_scale(filename)
       elif k == ord('G'):
           algo_gray_scale(filename)
       elif k == ord('c'):
           cycle_image(filename)
       elif k == ord('s'):
           smooth_image(filename)        
       elif k == ord('x'):
            convolution_x_derivative(filename)
       elif k == ord('y'):
            convolution_y_derivative(filename)
       elif k == ord('m'):
           magnitude(filename)
       elif k == ord('r'):
           rotate_Q_angle(filename)
       elif k == ord('h'):
           help(filename)
       elif k == ord('S'):
            algo_smooth_image(filename)

def load_image(filename): 
    cv2.destroyAllWindows()
    main(filename)

def output_image(image): 
    cv2.imwrite('./output.jpg', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def openCV_gray_scale(filename):
    image = cv2.imread(filename)
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray scaled Image with OpenCV', grayScale)
    k = cv2.waitKey(0)
    if k == ord('w'):
        output_image(grayScale)
    if k == ord('i'):
        load_image(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def algo_gray_scale(filename):
    image = cv2.imread(filename)
    blue,green,red = cv2.split(image)
    grayScale = (blue+green+red/((blue+green+red)/3))
    cv2.imshow('Gray scaled Image using Algorithm',grayScale)
    k = cv2.waitKey(0)
    if k == ord('w'):
        output_image(grayScale)
    elif k == ord('i'):
        load_image(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cycle_image(filename):
    image = cv2.imread(filename)
    blue,green,red = cv2.split(image) 
    image[:,:,random.randrange(0, 3, 1)] = 0 
    cv2.imshow('Color channel',image)
    k = cv2.waitKey(0)
    if k == ord('w'):
        output_image(image)
    elif k == ord('i'):
        load_image(filename)
    elif k == ord('c'):
        cycle_image(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()  

def nothing(x):
    pass
    
def smooth_image(filename): 
    image = cv2.imread(filename)
    cv2.namedWindow('Image')
    low_k = 1  
    high_k = 100 
    cv2.createTrackbar('Blur', 'Image', low_k, high_k, nothing)
    while(True):
        ksize = cv2.getTrackbarPos('Blur', 'Image')
        ksize = 2*ksize+1  
        gaussian = cv2.GaussianBlur(image,(ksize,ksize),cv2.BORDER_DEFAULT)
        cv2.imshow('Image', gaussian) 
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('w'):
            output_image(gaussian)
            break
        if k == ord('i'):
            load_image(filename)
            break
    cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()

def algo_smooth_image(filename):
    image = cv2.imread(filename)
    cv2.namedWindow('Image')
    low_k = 1  
    high_k = 100
    cv2.createTrackbar('Blur', 'Image', low_k, high_k, nothing)
    while(True):
        ksize = cv2.getTrackbarPos('Blur', 'Image')
        ksize = 2*ksize+1
        median = cv2.medianBlur(image, ksize)
        cv2.imshow('Image', median) 
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('w'):
            output_image(median)
            break
        if k == ord('i'):
            load_image(filename)
            break
    cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()


def  convolution_x_derivative(filename):
    image = cv2.imread(filename)    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    xconvul = cv2.flip(grayscale, 0)
    cv2.imshow("Convolution X", xconvul)
    k = cv2.waitKey(0)
    if k == ord('w'):
        output_image(xconvul)
    if k == ord('i'):
        load_image(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def  convolution_y_derivative(filename):
    image = cv2.imread(filename)    
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yconvul = cv2.flip(gray_scale, 1)
    cv2.imshow("Convolution Y", yconvul)
    k = cv2.waitKey(0)
    if k == ord('w'):
        output_image(yconvul)
    if k == ord('i'):
        load_image(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def rotate_Q_angle(filename): 
    image = cv2.imread(filename)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row,col = gray_scale.shape[:2]
    center = (row / 2, col / 2)
    low_k = 0  
    high_k = 360
    cv2.namedWindow('Rotated')
    cv2.createTrackbar('Rotated', 'Rotated', low_k, high_k, nothing)
    while(True):
        angle = cv2.getTrackbarPos('Rotated', 'Rotated')
        rotateMat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray_scale, rotateMat, (row , col))
        cv2.imshow("Rotated", rotated)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('w'):
            output_image(rotated)
            break
        if k == ord('i'):
            load_image(filename)
            break
    cv2.waitKey()
    cv2.destroyAllWindows()

def magnitude(filename):
    image = cv2.imread(filename)
    x_gradient = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=7) 
    y_gradient = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    cv2.convertScaleAbs(x_gradient,x_gradient)
    cv2.convertScaleAbs(y_gradient,y_gradient)    
    magnitude = cv2.magnitude(x_gradient,y_gradient)
    cv2.imshow('Magnitude',magnitude)
    k = cv2.waitKey(0)
    if k == ord('w'):
        output_image(mag)
    elif k == ord('i'):
        load_image(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

def help(filename): 
    print """
    This program performs different key operations on image. the image can be from a specified path or a camera capture.
    
    Key Operations: 
    
    i - reload the original image (i.e. cancel any previous processing)
    w - save the current (possibly processed) image into the file ’out.jpg’
    g - convert the image to grayscale using the openCV conversion function.
    G - convert the image to grayscale using your implementation of conversion function.
    c - cycle through the color channels of the image showing a different channel every time the key is pressed.
    s - convert the image to grayscale and smooth it using the openCV function. Use a track bar to control the amount of smoothing.
    S - convert the image to grayscale and smooth it using your function which should perform convolution with a suitable filter. Use a track bar to control the amount of smoothing.
    x - convert the image to grayscale and perform convolution with an x derivative filter. Normalize the obtained values to the range [0,255].
    y - convert the image to grayscale and perform convolution with a y derivative filter. Normalize the obtained values to the range [0,255].
    m - show the magnitude of the gradient normalized to the range [0,255]. The gradient is computed based on the x and y derivatives of the image.
    p - convert the image to grayscale and plot the gradient vectors of the image every N pixels and let the plotted gradient vectors have a length of K. Use a track bar to control N. Plot the vectors as short line segments of length K.
    r - convert the image to grayscale and rotate it using an angle of Q degrees. Use a track bar to control the rotation angle. The rotation of the image should be performed using an inverse map so there are no holes in it.
    For further imformation on executing the program, please refer to the report under the folder Report."""
    k = cv2.waitKey(0) 
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('w'):
       output_image(image)
    elif k == ord('g'):
       openCV_gray_scale(filename)
    elif k == ord('G'):
       algo_gray_scale(filename)
    elif k == ord('c'):
       cycle_image(filename)
    elif k == ord('s'):
       smooth_image(filename)        
    elif k == ord('x'):
        convolution_x_derivative(filename)
    elif k == ord('y'):
        convolution_y_derivative(filename)
    elif k == ord('m'):
       magnitude(filename)
    elif k == ord('r'):
        rotate_Q_angle(filename)
    elif k == ord('h'):
        help(filename)
    elif k == ord('S'):
        algo_smooth_image(filename)

main("./logo.jpg")
