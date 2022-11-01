import cv2
import numpy as np
from matplotlib import pyplot as plt

cam = cv2.VideoCapture(0)

def getcontours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        cv2.drawContours(imgcontours,cnt,-1,(255,0,0),4)




def everythingvideo(img0):
    # converting to gray scale
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(1,1),0)

    # convolute with proper kernels
    imgcanny = cv2.Canny(img,30,30)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    laplacianbeta = np.uint8(np.clip(laplacian, 0, 255))
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    sobel = sobely +sobelx
    sobelbeta = np.uint8(np.clip(sobel, 0, 255))
    th3 = cv2.adaptiveThreshold(sobelbeta,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #cv2.convertScaleAbs()
    getcontours(sobelbeta)
    cv2.imshow("laplace",laplacian)
    cv2.imshow("laplacebeta",laplacianbeta)
    cv2.imshow("sobelbeta",sobelbeta)
    cv2.imshow("X",sobelx)
    cv2.imshow("Y",sobely)
    cv2.imshow("TH3",th3)
    cv2.imshow("sobel",sobel)



while True:
    success, img = cam.read()
    imgcontours = img.copy()
    everythingvideo(img)
    cv2.imshow("contours?",imgcontours)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cam.release()
        cv2.destroyAllWindows()
        break