import cv2
import math
import numpy as np
from PIL import Image


cnt =0
def braille_exist(img, braille_TF):

    global cnt

    # img_visualize = cv2.rectangle(img, (280, 250), (360, 360), (255,255,255), 2)
    # cv2.imshow('original', img_visualize)
    # cv2.waitKey(0)

    section_ratio = np.zeros(3,dtype=np.float)

    for i in range(3):

        Temp = img[:,int((img.shape[1]/3)*i):int((img.shape[1]/3)*(i+1))]
        Temp_nonzero = np.count_nonzero(Temp)
        section_ratio[i] = 3*Temp_nonzero/ np.size(Temp)

    print(np.argmax(section_ratio),np.max(section_ratio))

    if np.argmax(section_ratio)==1 and np.max(section_ratio) > 0.7 :
        cnt +=1
        cv2.imwrite('test_center%d.png'%cnt,img)
    # img_mid = img[250:360, 280:360]
    # row, col = np.where(img_mid != 0)
    #
    # braille = np.size(col)
    # background = np.size(img_mid)
    # ratio = braille / background

    # cv2.imshow('ww', img_mid)
    # cv2.waitKey(0)
    # if ratio >= 0.7:
    #     braille_TF = True
    # print("braille block ratio = %f"%ratio)
    # print("Is User on the braille block?")
    # if braille_TF == True:
    #     print("YES")
    # else:
    #     print("NO")
    # print("\n")

def main():

    braille_TF = False
    for i in range(1, 64):
        img = cv2.imread('test_image%d.png'%i, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('original', img)
        # cv2.waitKey(0)
        print(i,"st frame")
        braille_exist(img, braille_TF)
    for j in range(1, 98):
        img = cv2.imread('frame_concat%d.png' % j, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('original', img)
        # cv2.waitKey(0)
        print(j, "st frame")
        braille_exist(img, braille_TF)
if __name__ == '__main__':
    main()
