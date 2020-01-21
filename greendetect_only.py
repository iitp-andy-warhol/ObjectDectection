import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from keras.models import load_model 

def make_cropped(img): 
    '''
    lower_orange = (0, 200, 200)
    upper_orange = (30, 255, 255)
    '''
    
    lower_green = (40, 100, 0)
    upper_green = (70, 255, 200)
    '''
    lower_blue = (0, 180, 55)
    upper_blue = (20, 255, 200)
    '''

    #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #cv2.imwrite("img_original.jpeg", img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    img_green_only = cv2.bitwise_and(img, img, mask =img_mask)
    
    #cv2.imwrite("result_green_only.jpeg", img_green_only)
    img_gray = cv2.cvtColor(img_green_only, cv2.COLOR_BGR2GRAY)

    #blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    #cv2.imwrite("result_blurred.jpeg", img_gray)
    ret, img_binary = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    #cv2.imwrite("result_binary.jpeg", img_binary)
    

    #_,contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    _,contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  
    tmp_con = [] 
    for c in contours: 
        area = cv2.contourArea(c) 
        #print(area)
        x,y,w,h = cv2.boundingRect(c)
        '''
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.imwrite('rrrrr.jpeg', img)
        '''
        if(area<2000) : 
            if(area>100): #200
                if(w/h>1.25):
                    tmp_con.append(c)
       

    contours = tmp_con   
    max_area = -1 
    max_con = 0 
    #print(len(tmp_con))

    if(len(tmp_con)!=0 ) :
        for c in contours: 
            area = cv2.contourArea(c)
            #print(area)

            x,y,w,h = cv2.boundingRect(c)
            #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            
            if(area> max_area):
                max_area = area 
                max_con = c 

        x,y,w,h = cv2.boundingRect(max_con)

        if(w/h >1.25):
            #cropped = img[y:y+h, x:x+w]
            #cv2.imwrite('rrrrr.jpeg', cropped)
            #print("Hello")
            return 1

        else: 
            return 0 
    
    else: 
        return 0

vidcap = cv2.VideoCapture('linetracing2.mp4')
count = 0 
 
f = open("./result_green_detect.txt", 'w') 

while(vidcap.isOpened()):
    
    ret, img = vidcap.read()
    if ret == True: 

        result = make_cropped(img)
        #make_cropped("rotated.jpeg")
    
        if(result ==1) : 
            f.write(" # %d : Yes! Detected! \n" %count)
    
        elif(result ==0) : 
            f.write(" # %d : NO!\n" %count)
    
    count+=1 
    
vidcap.release()
f.close()
