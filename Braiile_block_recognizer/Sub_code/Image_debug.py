import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from numpy.linalg import norm
import math


def arrow_show(img, Principle_M_component):
    for component in Principle_M_component:
        cv2.arrowedLine(img, (component[1], component[0]), (component[3], component[2]), 255, 1)

    for component in Principle_M_component:
        cv2.arrowedLine(img, (component[1], component[0]), (component[3], component[2]), 255, 1)
    cv2.imshow('d', img)
    cv2.waitKey(0)

def cos_similarity(vector1, vector2):
    return int(math.degrees(math.cos(np.inner(vector1, vector2) / (norm(vector1) * norm(vector2)))))

def main():

    contours = np.array([[50, 50], [50, 150], [150, 150], [150, 50]])
    img = np.zeros((200, 200))  # create a single channel 200x200 pixel black image
    cv2.fillPoly(img, pts=[contours], color=(255, 255, 255))
    cv2.imshow(" ", img)
    cv2.waitKey()

    img = cv2.imread('Crossline_image.png', cv2.COLOR_BGR2GRAY)
    img_show = img

    vector1 = [-1,1]
    vector2 = [1,1]
    angle = np.degrees(np.arccos(np.dot((vector1 / np.linalg.norm(vector1)), (vector2 / np.linalg.norm(vector2)))))
    ####  AFTER Centering ###

    test = np.array([[-115,34],[-103,61],[-41,-116]])
    a = np.sum((test > [-121,0]).astype(np.int), axis=0)[1]
    b= np.mean(test,axis=0)

    T_verify = np.delete(test, np.argmin(test, axis=0)[0], 0)
    T_verify = [[359,356,345,475],[359,356,352,474]]
    print(T_verify[0][3])
    k = np.absolute((T_verify[0][3] - T_verify[1][3]) / (T_verify[0][2] - T_verify[1][2]))
    ## STEP1 ##

    row,col = np.where(img ==100)

    max_col= np.argmax(np.bincount(col))
    max_row= np.argmax(np.bincount(row))
    b= np.bincount(col)
    col_info= np.bincount(col)

    row_1 = row[np.where(row>=240)]
    col_1 = col[np.argwhere(row>=240)].squeeze()

    center_row = np.max(row)
    center_col = col[np.argmax(row)]

    # center_row = np.max(row)
    # center_col = max_col

    ##Assuming Center Line##

    distance_buffer = np.zeros_like(row_1)

    center_point_buffer = np.zeros((1,2),dtype = np.uint16)
    Principle_vector = np.zeros((1, 2), dtype=np.uint16)
    Principle_component = np.zeros((1, 4), dtype=np.uint16)

    Principle_M_vector = np.zeros((1, 2), dtype=np.uint16)
    Principle_M_component = np.zeros((1, 4), dtype=np.uint16)

    Vec_count_1= Vec_count_2 = Vec_count_3 = 0


    Vector_buffer = np.zeros((1,4),dtype=np.uint16)
    center_point_buffer = np.append(center_point_buffer,np.reshape(np.array([center_row,center_col]),(1,2)),axis=0)
    center_point_buffer = center_point_buffer[1:]

    for index, row_val in enumerate(row_1):
        distance = (row_val - center_row) ** 2 + (col_1[index] - center_col) ** 2
        if distance <= 14400:
            distance_buffer[index] = distance

    max_index = np.where(distance_buffer > (np.max(distance_buffer)*0.95))
    max_index = np.reshape(np.asarray(max_index),np.asarray(max_index).size)

    for num_index, max_index_value in enumerate(max_index):

        endpoint_row = row_1[max_index_value]
        endpoint_col = col_1[max_index_value]
        Vector_buffer = np.append(Vector_buffer,np.reshape(np.array([center_row,center_col,endpoint_row,endpoint_col]),(1,4)),axis=0)
        #cv2.arrowedLine(img, (center_col,center_row),(endpoint_col,endpoint_row),255,1)

    Vector_buffer = Vector_buffer[1:]

    ## Vector Selecting##

    Temp = Vector_buffer[:,2:] - Vector_buffer[:,:2]

    Similarity_buffer = np.diag(np.arange(Temp.shape[0])+1)
    for j in range(Temp.shape[0]):
        Initial_vector = Temp[j]
        for i in range(Temp.shape[0]):
            if i > j:
                cos_sim = np.inner(Initial_vector, Temp[i]) / (norm(Initial_vector) * norm(Temp[i]))  ## cosine similarity
                if int(math.degrees(math.acos(cos_sim)))<3:
                    Similarity_buffer[j,i] = j+1

    for k in range(Temp.shape[0]):
        if np.count_nonzero(Similarity_buffer[:,k]) == 1:
              Principle_vector = np.append(Principle_vector,np.reshape(Temp[k],(1,2)),axis=0)
              Principle_component =np.append(Principle_component,np.reshape(Vector_buffer[k],(1,4)),axis=0)

    Principle_vector = Principle_vector[1:]
    Principle_component = Principle_component[1:]

    ## Vector  Verification In Frist step ##
    for i, vector_component in enumerate(Principle_component) :
        if Principle_vector[i][1] <=0 :
            Verify_mat = img[vector_component[2]:vector_component[0]+1,vector_component[3]:vector_component[1]+1]/100
        else :
            Verify_mat = img[vector_component[2]:vector_component[0]+1,vector_component[1]:vector_component[3]+1]/100
        Count_pixel = np.sum(np.multiply(Verify_mat,np.ones_like(Verify_mat)),dtype=np.uint16)
        Diag_size  = int(math.sqrt(Verify_mat.shape[0]**2+Verify_mat.shape[1]**2))
        #print(Count_pixel,Diag_size,Count_pixel/Diag_size)
        if Count_pixel/Diag_size > 0.6: ##  Need to debug
            Vec_count_1 +=1
            print(Vec_count_1,Count_pixel/Diag_size)
            Principle_M_vector = np.append(Principle_M_vector,np.reshape(Principle_vector[i],(1,2)),axis=0)
            Principle_M_component = np.append(Principle_M_component,np.reshape(vector_component,(1,4)),axis=0)

    Principle_vector = np.zeros((1, 2), dtype=np.uint16)
    Principle_component = np.zeros((1, 4), dtype=np.uint16)

    Principle_M_component= Principle_M_component[1:]
    Principle_M_vector= Principle_M_vector[1:]

    ## Classify by case ##
    ## On the Straight Line##
    if Vec_count_1 == 1 :

        center_row = Principle_M_component[0,2]
        center_col = Principle_M_component[0,3]

        ##Stacking center point##
        center_point_buffer = np.append(center_point_buffer,np.reshape(np.array([center_row,center_col]),(1,2)),axis=0)

        ## Step 2 ##
        row_2 = row[np.where((row<240) & (row>=120))]
        col_2 = col[np.argwhere((row<240) & (row>=120))].squeeze()

        distance_buffer = np.zeros_like(row_2)
        Vector_buffer = np.zeros((1, 4), dtype=np.uint16)

        for index, row_val in enumerate(row_2):
            distance = (row_val - center_row) ** 2 + (col_2[index] - center_col) ** 2
            if distance <= 14400:
                distance_buffer[index] = distance

        max_index = np.where(distance_buffer > np.max(distance_buffer)*0.99)
        max_index = np.reshape(np.asarray(max_index),np.asarray(max_index).size)

        for num_index, max_index_value in enumerate(max_index):
            endpoint_row = row_2[max_index_value]
            endpoint_col = col_2[max_index_value]
            Vector_buffer = np.append(Vector_buffer,np.reshape(np.array([center_row,center_col,endpoint_row,endpoint_col]),(1,4)),axis=0)
            #cv2.arrowedLine(img, (center_col,center_row),(endpoint_col,endpoint_row),255,1)

        Vector_buffer = Vector_buffer[1:]

        Temp = Vector_buffer[:, 2:] - Vector_buffer[:, :2]
        ## Vector Selecting##
        Similarity_buffer = np.diag(np.arange(Temp.shape[0]) + 1)

        for j in range(Temp.shape[0]):
            Initial_vector = Temp[j]
            for i in range(Temp.shape[0]):
                if i > j:
                    cos_sim = np.inner(Initial_vector, Temp[i]) / (norm(Initial_vector) * norm(Temp[i]))  ## cosine similarity
                    if int(math.degrees(math.acos(cos_sim))) < 3:
                        Similarity_buffer[j, i] = j + 1

        for k in range(Temp.shape[0]):
            if np.count_nonzero(Similarity_buffer[:, k]) == 1:
                Principle_vector = np.append(Principle_vector, np.reshape(Temp[k], (1, 2)), axis=0)
                Principle_component = np.append(Principle_component, np.reshape(Vector_buffer[k], (1, 4)), axis=0)

        Principle_vector = Principle_vector[1:]
        Principle_component = Principle_component[1:]

        ## Vector  Verification In Frist step ##
        for i, vector_component in enumerate(Principle_component):
            if Principle_vector[i][1] <= 0:
                Verify_mat = img[vector_component[2]:vector_component[0] + 1,vector_component[3]:vector_component[1] + 1] / 100
            else:
                Verify_mat = img[vector_component[2]:vector_component[0] + 1,vector_component[1]:vector_component[3] + 1] / 100
            Count_pixel = np.sum(np.multiply(Verify_mat, np.ones_like(Verify_mat)), dtype=np.uint16)
            Diag_size = int(math.sqrt(Verify_mat.shape[0] ** 2 + Verify_mat.shape[1] ** 2))
            #print(Count_pixel, Diag_size, Count_pixel / Diag_size)
            if Count_pixel / Diag_size > 0.25:  ##  Need to debug
                Vec_count_2 += 1
                Principle_M_vector = np.append(Principle_M_vector, np.reshape(Principle_vector[i], (1, 2)), axis=0)
                Principle_M_component = np.append(Principle_M_component, np.reshape(vector_component, (1, 4)), axis=0)

        ## Connected Line ##
        if Vec_count_2 == 1: ## More condition required

            Principle_vector = np.zeros((1, 2), dtype=np.uint16)
            Principle_component = np.zeros((1, 4), dtype=np.uint16)

            center_row = Principle_M_component[1,2]
            center_col = Principle_M_component[1,3]

            ##Center_point_stacking
            center_point_buffer = np.append(center_point_buffer,np.reshape(np.array([center_row,center_col]),(1,2)),axis=0)

            row_3 = row[np.where(row<120)]
            col_3 = col[np.argwhere(row<120)].squeeze()

            Vector_buffer = np.zeros((1, 4), dtype=np.uint16)
            distance_buffer = np.zeros_like(row_3)

            for index, row_val in enumerate(row_3):
                distance = (row_val - center_row) ** 2 + (col_3[index] - center_col) ** 2
                if distance <= 14400:
                    distance_buffer[index] = distance

            max_index = np.where(distance_buffer > np.max(distance_buffer)*0.99)
            max_index = np.reshape(np.asarray(max_index),np.asarray(max_index).size)

            for num_index, max_index_value in enumerate(max_index):

                endpoint_row = row_3[max_index_value]
                endpoint_col = col_3[max_index_value]
                Vector_buffer = np.append(Vector_buffer,np.reshape(np.array([center_row,center_col,endpoint_row,endpoint_col]),(1,4)),axis=0)

            Vector_buffer = Vector_buffer[1:]

            Temp = Vector_buffer[:, 2:] - Vector_buffer[:, :2]
            ## Vector Selecting##
            Similarity_buffer = np.diag(np.arange(Temp.shape[0]) + 1)

            for j in range(Temp.shape[0]):
                Initial_vector = Temp[j]
                for i in range(Temp.shape[0]):
                    if i > j:
                        cos_sim = np.inner(Initial_vector, Temp[i]) / (norm(Initial_vector) * norm(Temp[i]))  ## cosine similarity
                        if int(math.degrees(math.acos(cos_sim))) < 3:
                            Similarity_buffer[j, i] = j + 1

            for k in range(Temp.shape[0]):
                if np.count_nonzero(Similarity_buffer[:, k]) == 1:
                    Principle_vector = np.append(Principle_vector, np.reshape(Temp[k], (1, 2)), axis=0)
                    Principle_component = np.append(Principle_component, np.reshape(Vector_buffer[k], (1, 4)), axis=0)

            Principle_vector = Principle_vector[1:]
            Principle_component = Principle_component[1:]

            ## Vector  Verification In Frist step ##
            for i, vector_component in enumerate(Principle_component):
                if Principle_vector[i][1] <= 0:
                    Verify_mat = img[vector_component[2]:vector_component[0] + 1,
                                 vector_component[3]:vector_component[1] + 1] / 100
                else:
                    Verify_mat = img[vector_component[2]:vector_component[0] + 1,
                                 vector_component[1]:vector_component[3] + 1] / 100
                Count_pixel = np.sum(np.multiply(Verify_mat, np.ones_like(Verify_mat)), dtype=np.uint16)
                Diag_size = int(math.sqrt(Verify_mat.shape[0] ** 2 + Verify_mat.shape[1] ** 2))
                #print(Count_pixel, Diag_size, Count_pixel / Diag_size)
                if Count_pixel / Diag_size > 0.25:  ##  Need to debug
                    Vec_count_3+=1
                    Principle_M_vector = np.append(Principle_M_vector, np.reshape(Principle_vector[i], (1, 2)), axis=0)
                    Principle_M_component = np.append(Principle_M_component, np.reshape(vector_component, (1, 4)), axis=0)
            ## Step 3 ##
            if Vec_count_3 == 1: ## More condition required

                print('Straight Line')
                #####Detecting Curve####
                Cos_sim_buffer = np.zeros(2, dtype=np.float)

                Cos_sim_buffer[0] = cos_similarity(Principle_M_vector[0],Principle_M_vector[1])
                Cos_sim_buffer[1] = cos_similarity(Principle_M_vector[1],Principle_M_vector[2])

                # Angle Thresholding (Below 5 degrees)
                Cos_sim_buffer =  np.where(Cos_sim_buffer <= 5,0,Cos_sim_buffer)

                if np.count_nonzero(Cos_sim_buffer) == 0:
                    print('Straight way forward')

                else :
                    Dominant_curve = np.argmax(Cos_sim_buffer)
                    Dominant_curve_degree = np.max(Cos_sim_buffer)

                    if Dominant_curve == 0 : ## Distance

                        if Principle_M_component[1,3] - Principle_M_component[1,3] >=0:  ## Side
                            if Dominant_curve_degree > 60:
                                print('Soon will meet right way ',Dominant_curve_degree)
                            else :
                                print('Soon will meet right curve with %d degree'%Dominant_curve_degree)
                                arrow_show(img_show,Principle_M_component)
                        else :
                            if Dominant_curve_degree > 60:
                                print('Soon will meet left way')
                            else :
                                print('Soon will meet left curve with %d degree'%Dominant_curve_degree)

                    else :
                        if Principle_M_component[1, 3] - Principle_M_component[1, 3] >= 0:  ## Side
                            if Dominant_curve_degree > 60:
                                print('Right way appeared', Dominant_curve_degree)
                            else:
                                print('Right curve with %d degree appeared' % Dominant_curve_degree)
                        else:
                            if Dominant_curve_degree > 60:
                                print('Left way appeared')
                            else:
                                print('Left curve with %d degree appeared' % Dominant_curve_degree)
            else:
                arrow_show(img_show, Principle_M_component)
                print('Cross section far from here')
                ################################Vector Debug##########################################

        ## Middle point cross line
        else :
            Cos_sim_buffer = np.zeros(Vec_count_2,dtype=np.float)

            for i in range(1, Principle_M_vector.shape[0]):

                Cos_sim_buffer[i-1] = cos_similarity(Principle_M_vector[0],Principle_M_vector[i])

            Condition_check = np.count_nonzero(Cos_sim_buffer)

            if Condition_check < Vec_count_2 :
                print('Will soon meet straight way and new way')
                ##Left, right Side detection required#3
            else :
                print("Will soon meet T cross section")


    ## From Step 1 ##

    ## Several point detected in high confidence region ##
    elif Vec_count_1 == 2: ##

        print('Arrived to cross line ')

    elif Vec_count_1 == 3: ##

        arrow_show(img_show,Principle_M_component)
        print('Arrived to T cross line')

    ## Too many noise in high confidence region ##
    else:
        print('Noise abort or Arrived to cross line ')


    #
    # elif Principle_vector.shape[0] != 1:
    #
    #         print('Uncertain route')


    # img = mpimg.imread('T_crossline.png')
    #
    # img = np.where(img<20,0,img)
    # img = np.where(img!=0,100,img)
    #
    # cv2.imshow('d',img)
    # cv2.waitKey(0)

    # row,col = np.where(img==100)
    # row_unique = np.unique(row)
    # col_unique = np.unique(col)

    #
    # for i in range(row_unique.size):
    #     temp_row = row_unique[i]
    #     row_index = col[np.where(row == temp_row)]
    #     temp_last = 0
    #     for j in range(row_index.size-1):
    #         if row_index[j+1] != (row_index[j]+1) :
    #             mean = np.round(np.mean(row_index[temp_last:j+1])).astype(np.uint16)
    #             img[temp_row,mean] = 255
    #             temp_last = j+1
    #             #print('d')
    #         elif j == row_index.size-2 :
    #             mean = np.round(np.mean(row_index[temp_last:j+1])).astype(np.uint16)
    #             img[temp_row,mean] = 255
    #
    # for i in range(col_unique.size):
    #     temp_col = col_unique[i]
    #     col_index = row[np.where(col == temp_col)]
    #     temp_last = 0
    #     for j in range(col_index.size - 1):
    #         if col_index[j + 1] != (col_index[j] + 1):
    #             mean = np.round(np.mean(col_index[temp_last:j + 1])).astype(np.uint16)
    #             img[mean, temp_col] = 0
    #             temp_last = j + 1
    #             # print('d')
    #         elif j == col_index.size - 2:
    #             mean = np.round(np.mean(col_index[temp_last:j + 1])).astype(np.uint16)
    #             img[mean, temp_col] = 0
    #
    # cv2.imshow('d',img)
    # cv2.waitKey(0)
    # implot = plt.imshow(img)
    # plt.show()
main()