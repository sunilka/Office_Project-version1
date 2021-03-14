import cv2
import easyocr
import matplotlib
import numpy as np

# func that uses easyocr for text extraction
def text_extraction(img):
    reader = easyocr.Reader(['en'],False)
    text = reader.readtext(img,detail = 0)
    print(text)
    
# func showing the image on screen
def show_image_on_screen(screen_name,img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('C:/Users/Shruthi/Downloads/0001.jpg')
show_image_on_screen("hello",img)

# Steps for binarization
# 1) converting the image to gray scale image first
# 2) Applying the Ostu binarization on the gray scale image
# Note: The ostu binarization works on gray images only


# converting the image to gray scale for the binarization process
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blurring the image 
blur = cv2.GaussianBlur(img_grey,(5,5),0)

# performing the Ostu's binarization on the blurred image 
img_thresh_Gaussian = cv2.adaptiveThreshold(blur,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# displaying the image after binarization
# half = cv2.resize(img_thresh_Gaussian, (0, 0), fx = 0.2, fy = 0.2)
show_image_on_screen("binarized",img_thresh_Gaussian)


text_extraction(img_thresh_Gaussian)