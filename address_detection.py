import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model 

def make_cropped(img_path): 
    '''
    lower_orange = (0, 200, 200)
    upper_orange = (30, 255, 255)
    '''
    
    lower_green = (40, 150, 0)
    upper_green = (60, 255, 100)
    '''
    lower_blue = (0, 180, 55)
    upper_blue = (20, 255, 200)
    '''

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.imwrite("img_original.jpeg", img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    img_green_only = cv2.bitwise_and(img, img, mask =img_mask)

    cv2.imwrite("result_green_only.jpeg", img_green_only)
    img_gray = cv2.cvtColor(img_green_only, cv2.COLOR_BGR2GRAY)

    #blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    cv2.imwrite("result_blurred.jpeg", img_gray)
    ret, img_binary = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)

    cv2.imwrite("result_binary.jpeg", img_binary)
    

    #_,contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    _,contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  
    tmp_con = [] 
    for c in contours: 
        area = cv2.contourArea(c) 
  
        x,y,w,h = cv2.boundingRect(c)
        
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        if(area<1900) : 
            if(area>200): #200
                tmp_con.append(c)
       

    
    contours = tmp_con   
    max_area = -1 
    max_con = 0 
    print(len(tmp_con))
    if(len(tmp_con)!=0 ) :
        for c in contours: 
            area = cv2.contourArea(c)

            if(area> max_area):
                max_area = area 
                max_con = c 

        x,y,w,h = cv2.boundingRect(max_con)

        if(w/h >1.4):
            cropped = img[y:y+h, x:x+w]
            print(cv2.contourArea(max_con))
            #print("Thisis")
            cv2.imwrite('cropped.jpeg', cropped)
            #print("Hello")
            return 1  
        else: 
            print("Only 2 digit number is recognized. can not recognize the address number")
    
    else: 
        print("The sign is too small. can not recognize the address number") 



def make_2_contours(contours): 
        cont_info = {} 

        for i, c in enumerate(contours): 
            x,y,w,h = cv2.boundingRect(c) 
            cont_info[i] = x 

        sorted_cont_info = sorted(cont_info.items(), key=lambda x: x[1])

        _1_digit_cont_ind = sorted_cont_info[0][0] 
        _3_digit_cont_ind = sorted_cont_info[-1][0]

        ret_list =[] 
        ret_list.append(contours[_1_digit_cont_ind]) 
        ret_list.append(contours[_3_digit_cont_ind])

        return ret_list



def make_digit_contour(img_path): 

    
    lower_white = (30, 0, 20)
    upper_white = (70, 255, 255)  #masking white  
  

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #print(img.shape)
    #img = cv2.resize(img, (300,150))
    cv2.imwrite("cropped_orginal.jpeg",img)


    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_white, upper_white)
    img_green_only = cv2.bitwise_and(img, img, mask =img_mask)

    #img_blurred = cv2.blur(img_green_only,(3,3) )
    cv2.imwrite("result_small_green_only.jpeg", img_green_only)

    img_gray = cv2.cvtColor(img_green_only, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("result_small_gray.jpeg", img_gray)

    img_blurred = cv2.blur(img_gray,(3,3))

    cv2.imwrite("result_small_blurred.jpeg", img_blurred)


    ret, img_binary = cv2.threshold(img_blurred, 50, 255, cv2.THRESH_BINARY)

    cv2.imwrite("result_small_binary.jpeg", img_binary)


    kernel = np.ones((1,1), np.uint8) 

    #img_erosion = cv2.erode(img, kernel, iterations=1) 
    img_dilation = cv2.dilate(img_binary, kernel, iterations=1)

    cv2.imwrite("dilation.jpeg", img_dilation)


    _,contours, _ = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 


    for c in contours: 
        x,y,w,h = cv2.boundingRect(c) 
        s = (x,y) 
        e = (x+w, y+h)
        img = cv2.rectangle(img, s, e, (0,255,0), 1)
        #print(cv2.contourArea(c))
        cv2.imwrite("ccccccc.jpeg", img)


    num_cont = len(contours)
   
    #answer_digit = 999 
    if(num_cont >=3) : 
        contours = make_2_contours(contours) 

        for i, c in enumerate(contours): 
            x,y,w,h = cv2.boundingRect(c) 
            #s = (x,y) 
            #e = (x+w, y+h)
            #img = cv2.rectangle(img, s, e, (0,0,255), 1)
            #cv2.imwrite("only2.jpeg", img)
            #print(cv2.contourArea(c))
            cropped = img_dilation[y:y+h, x:x+w]
            win_name = "cropped"+str(i)+".jpeg" 

            cv2.imwrite(win_name, cropped)
        _1_digit_num =input_to_model("cropped0.jpeg")
        _3_digit_num = input_to_model("cropped1.jpeg")
       

        address_num = 100*_1_digit_num + _3_digit_num
    
        print("The address number is "+str(address_num)) 
        

    else: 
      
        print("can not recognize the address number") 
    
def model_predict(test_num):
    answer = 9
    
    #test_num = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
    test_num = test_num.reshape((1, 28, 28, 1))
    answer = model.predict_classes(test_num)
    #print('The Answer is ', model.predict_classes(answer))
    return answer


def input_to_model(img_path): 
    test_num = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    w, h = test_num.shape
    #print("This is area")
    #print(w+h)
    #print(w/h) 
    answer = 9
    #print(w/h)

    if((w/h) >2 or w*h<100) :
        answer = 1
    
    elif((w/h) <=2 ) : 
        
        test_num = cv2.resize(test_num, dsize=(28, 28))

        answer = model_predict(test_num)
        
        
    return answer

model = load_model('MNIST_CNN_model.h5')
'''
for i in range(20): 
    frame_num =str(370+i)
    print(frame_num)
    img_path = "./images/frame"+ frame_num+ ".jpg"

    ret = make_cropped(img_path)
    #make_cropped("rotated.jpeg")
    
    if(ret ==1) : 
        make_digit_contour("cropped.jpeg")

'''


img_path = "./images/frame0.jpg"
ret = make_cropped(img_path)
#print(ret)
if(ret ==1) : 
   
    make_digit_contour("cropped.jpeg")

