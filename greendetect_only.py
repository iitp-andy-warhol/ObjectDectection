import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from keras.models import load_model 


def CalConti(current_num, prev_num, prev_conti_0, prev_conti_1): 

    if current_num == 0 :  
        if prev_num== 0: 
            current_conti_0 = 1+ prev_conti_0 
            current_conti_1 = 0 

        elif prev_num == 1: 
            current_conti_0 = 0 
            current_conti_1 = 1 + prev_conti_1 

    elif current_num == 1: 
        if prev_num == 0 : 
            current_conti_0 = 1+ prev_conti_0 
            current_conti_1 = 0 

        elif prev_num ==1: 
            current_conti_0 = 0 
            current_conti_1 = 1 + prev_conti_1

    return  current_conti_0, current_conti_1 



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

    height = img.shape[0] 
    height = int(height)
    width = img.shape[1] 

    img[int(height/4):height, :, :] = 0
    img[:int(height/4), :int(width/3), :] = 0 
    cv2.imwrite("half_black.jpeg", img)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    img_green_only = cv2.bitwise_and(img, img, mask =img_mask)
    
    cv2.imwrite("result_green_only.jpeg", img_green_only)
    img_gray = cv2.cvtColor(img_green_only, cv2.COLOR_BGR2GRAY)

    ret, img_binary = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
    cv2.imwrite("result_binary.jpeg", img_binary)
    

    #_,contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    _,contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  
    tmp_con = [] 
    for c in contours: 
        area = cv2.contourArea(c) 
        x,y,w,h = cv2.boundingRect(c)
        #print(area)
        '''
        print("\n")
        print(area)
        x,y,w,h = cv2.boundingRect(c)
        print(w/h)
        print('\n')
        '''
        
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.imwrite('rrrrr.jpeg', img)
        
        if(area<2000) : 
            if(area>100): #200
                print(w/h)
                if(w/h>1.2):
                    if(w/h<3.5): 
                        tmp_con.append(c)
       

    contours = tmp_con   
    max_area = -1 
    max_con = 0 
    #print(len(tmp_con))

    if(len(tmp_con)!=0 ) :

        for c in contours: 
            area = cv2.contourArea(c)
            print(area)

            x,y,w,h = cv2.boundingRect(c)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.imwrite('rrrrr.jpeg', img)
            
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

count = 0 

f = open("./result_green_detect.txt", 'w') 

prev_num = 0 
prev_conti_0 = 0
prev_conti_1 = 0


for i in range(270): 
    frame_num =str(i)
    print(frame_num)
    img_path = "./images/frame"+ frame_num+ ".jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ret = make_cropped(img)
    #make_cropped("rotated.jpeg")
    frame_num = i 
    
    if(ret==1): 
        f.write(str(frame_num) +' '+ "Detected!\n" )

    '''
    current_conti_0, current_conti_1 = CalConti(ret, prev_num, prev_conti_0, prev_conti_1)

    if current_conti_1 >=9:
        f.write(str(frame_num) +' '+ "Detected!\n" )

    prev_num = ret
    prev_conti_0 = current_conti_0
    prev_conti_1 = current_conti_1
    
    ''



frame_num =str(82)
#print(frame_num)
img_path = "./images/frame"+ frame_num+ ".jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
ret = make_cropped(img)
#make_cropped("rotated.jpeg")
if(ret ==1) : 
    #f.write("%d 1\n" %frame_num)
    print("detected")

else:   
    #f.write("%d 0\n" %frame_num) 
    print("not")

'''
