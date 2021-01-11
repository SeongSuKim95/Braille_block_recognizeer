# predict on jpg files or mp4 video
import cv2
import torch
from glob import glob
import os
import os.path as osp
from pathlib import Path
from torchvision import transforms
from modules.dataloaders.utils import decode_segmap
from modules.models.deeplab_xception import DeepLabv3_plus
from modules.models.sync_batchnorm.replicate import patch_replication_callback
import numpy as np
from PIL import Image
from numpy.linalg import norm
import math
import timeit
import scipy.spatial.distance as ssd
import pygame

pygame.mixer.init()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
### RUN OPTIONS ###
MODEL_PATH = './run/surface/deeplab/experiment/checkpoint.pth.tar'

ORIGINAL_HEIGHT = 360
ORIGINAL_WIDTH = 640
ORIGINAL_WIDTH = 640

MODEL_HEIGHT = 360
MODEL_WIDTH = 640
NUM_CLASSES = 5  # including background
CUDA = True if torch.cuda.is_available() else False

MODE = 'mp4'
DATA_PATH = './test/mp4/TEST4.mp4'
OUTPUT_PATH = './output/mp4/4.mp4'

# MODE = 'jpg'  # 'mp4' or 'jpg'
# DATA_PATH = './test/test_image16.png'  # .mp4 path or folder containing jpg images
# OUTPUT_PATH = 'output/jpgs/temp'  # where video file or jpg frames folder should be saved.

SHOW_OUTPUT = True if 'DISPLAY' in os.environ else False  # whether to cv2.show()

OVERLAPPING = False  # whether to mix segmentation map and original image
FPS_OVERRIDE = 30  # None to use original video fps

# class = 7
CUSTOM_COLOR_MAP = [
    [0, 0, 0],  # background
    # [255, 128, 0], #bike_lane
    # [255, 0 , 0 ], #caution_zone
    [255, 0, 255],  # crosswalk #pink
    [255, 255, 0],  # guide_block #yellow
    [0, 0, 255],  # roadway # blue
    [0, 255, 0],  # sidewalk # green
]  # To ignore unused classes while predicting

CUSTOM_N_CLASSES = len(CUSTOM_COLOR_MAP)

arrow = np.zeros((360, 640), dtype=np.uint8)

frame_concat = np.zeros((360, 640), dtype=np.float)
start_time = 0
cnt = 0
concat_cnt = 0
Info_cnt = 0
Stop_Info_cnt = 0
non_detect_cnt = 0
inform_txt = ""

distance_valid_ratio = 0.98
cross_detect_ratio = 20
vector_valid_angle = 10
concat_frame_num = 3
state = 0
state_before = 0
state_txt = ""
curve_degree = 0

## Case classifier
case_buffer = np.ones(10, dtype=np.uint8) * 9
stop_buffer = np.ones(10, dtype=np.uint8)
obstacle_buffer = np.zeros(10, dtype=np.float)
non_detect_buffer = np.ones(10, dtype=np.uint8)

obstacle_detector = np.zeros(1, dtype=np.float)
obstacle_flag = 0

straight_case = np.array([32, 33, 35, 37, 39, 317, 28])
T_case = np.array([311, 316, 24, 27, 14, 17, 318, 29, 19, 110])
Right_case = np.array([34, 38, 312, 314, 22, 25, 12, 15])
Left_case = np.array([36, 310, 313, 315, 23, 26, 13, 16])
Non_detected_case = np.array([11])
Noise_case = np.array([18])
End_case = np.array([31, 21])

# Obstacle_case = np.array([4])

def arrow_show(img_show, Principle_M_component, concat_cnt, case_num):
    for component in Principle_M_component:
        cv2.arrowedLine(img_show, (component[1], component[0]), (component[3], component[2]), (255, 255, 255), 1)

def arrow_write(img_show, Principle_M_component, concat_cnt, case_num):
    for component in Principle_M_component:
        cv2.arrowedLine(img_show, (component[1], component[0]), (component[3], component[2]), (0, 0, 255), 3)
    return img_show

def cos_similarity(vector1, vector2):
    return np.degrees(np.arccos(np.dot((vector1 / np.linalg.norm(vector1)), (vector2 / np.linalg.norm(vector2)))))
    # return int(math.degrees(math.cos(np.inner(vector1, vector2) / (norm(vector1) * norm(vector2)))))

def case_num_classifier(case_num):
    global straight_case
    global Right_case
    global Left_case
    global Non_detected_case
    global Noise_case
    global End_case
    # global Obstacle_case

    if np.any(straight_case == case_num):
        case_num = 2
    if np.any(Right_case == case_num):
        case_num = 4
    if np.any(Left_case == case_num):
        case_num = 5
    if np.any(T_case == case_num):
        case_num = 3
    if np.any(Non_detected_case == case_num):
        case_num = 0
    if np.any(Noise_case == case_num):
        case_num = 1
    if np.any(End_case == case_num):
        case_num = 6
    # if np.any(Obstacle_case == case_num):
    #     case_num = 7
    return case_num

def case_buffering(concat_cnt, case_num, case_buffer):
    if concat_cnt <= case_buffer.shape[0]:
        case_buffer[concat_cnt - 1] = case_num_classifier(case_num)
    else:
        case_buffer = np.delete(case_buffer, 0, 0)
        case_buffer = np.append(case_buffer, np.array([case_num_classifier(case_num)]), axis=0)

    return case_buffer

def Buffering(buffer):
    buffer = np.delete(buffer, 0, 0)
    buffer = np.append(buffer, np.array([0]), axis=0)
    return buffer
def Obstacle_buffering(Dominant_label_map, case_num):
    global obstacle_buffer
    global concat_cnt
    global state
    global state_before
    global obstacle_flag
    Label_sum = (Dominant_label_map == 1).sum()

    if concat_cnt <= obstacle_buffer.shape[0]:
        obstacle_buffer[concat_cnt - 1] = Label_sum

    else:
        obstacle_buffer = np.delete(obstacle_buffer, 0, 0)
        obstacle_buffer = np.append(obstacle_buffer, np.array([Label_sum]), axis=0)

    if np.any(np.array([21, 31, 18]) == case_num):
        if state_before == 2:
            print(obstacle_buffer)
            Obstacle_detecting(obstacle_buffer)
        else:
            obstacle_flag = 0
    else:
        obstacle_flag = 0


def Obstacle_detecting(Obstacle_buffer):
    global concat_cnt
    global obstacle_buffer
    print(state_before, state)

    obstacle_detector = np.zeros(1, dtype=np.float)

    if concat_cnt >= 2 and concat_cnt <= Obstacle_buffer.shape[0]:
        for i in range(1, concat_cnt):
            obstacle_detector = np.append(obstacle_detector, np.array([Obstacle_buffer[i] - Obstacle_buffer[i - 1]]), axis=0)
        obstacle_detector = np.delete(obstacle_detector, 0, 0)

    elif concat_cnt > Obstacle_buffer.shape[0]:
        for i in range(1, Obstacle_buffer.shape[0]):
            obstacle_detector = np.append(obstacle_detector, np.array([Obstacle_buffer[i] - Obstacle_buffer[i - 1]]), axis=0)
        obstacle_detector = np.delete(obstacle_detector, 0, 0)
        print(obstacle_detector)

    if obstacle_detector[-1] < 0:
        Temp_diff = np.zeros(obstacle_detector.shape[0] - 1, dtype=np.float)
        obstacle_detector = np.abs(obstacle_detector)
        for i in range(Temp_diff.shape[0]):
            Temp_diff[i] = obstacle_detector[i + 1] - obstacle_detector[i]
        Temp_diff = np.abs(Temp_diff)
        mean = np.mean(Temp_diff[:-1])
        var = np.std(Temp_diff[:-1])
        print((Temp_diff[-1] - mean) / var)
        Obstacle_Criterion = abs((Temp_diff[-1] - mean) / var)
        Obstacle_Informing(Obstacle_Criterion)

def Obstacle_Informing(Criterion):
    global obstacle_buffer
    global obstacle_flag
    global inform_txt

    if Criterion > 0.05:
        # obstacle_flag = 1
        print("Obstacle detected")
        inform_txt = "Obstacle detected"
        obstacle_buffer = np.delete(obstacle_buffer, -1, 0)

    elif Criterion > 20:
        # obstacle_flag = 1
        print("Obstacle in front of you")
        inform_txt = "Obstacle in front of you"
        obstacle_buffer = np.delete(obstacle_buffer, -1, 0)
        pygame.mixer.music.load('./sound/obstacle.wav')
        pygame.mixer.music.play()
    else:
        obstacle_flag = 0

def weighted_case(case_buffer):
    weighted_case = np.zeros(np.max(case_buffer) + 1)
    weight = np.arange(10) + 1

    for i in range(np.max(case_buffer) + 1):
        weighted_case[i] = np.sum(weight[np.where(case_buffer == i)])

    return weighted_case

def Stop_Informing(Dominant_state):
    global obstacle_flag
    global Info_cnt
    global inform_txt
    global state_txt
    print("%%%%%%%%%%%%%%%%%%%%%%%")

    if Dominant_state != 0:
        if Dominant_state == 2:
            print("End of the guide line")

        elif Dominant_state == 3:
            print("Arrived to T cross section")
            inform_txt = "Arrived to T cross section"
            state_txt = "T"
            if pygame.mixer.music.get_busy() == 0:
                pygame.mixer.music.load('./sound/arrive T.wav')
                pygame.mixer.music.play()

        elif Dominant_state == 4:
            print("Turn to Right")
            inform_txt = "Turn to Right"
            state_txt = "R"

            if pygame.mixer.music.get_busy() == 0:
                pygame.mixer.music.load('./sound/arrive right.wav')
                pygame.mixer.music.play()

        elif Dominant_state == 5:
            print("Turn to Left")
            inform_txt = "Turn to Left"
            state_txt = "L"

            if pygame.mixer.music.get_busy() == 0:
                pygame.mixer.music.load('./sound/arrive left.wav')
                pygame.mixer.music.play()
    else:
        print("Can't detect guide line")
        state_txt = ""

        inform_txt = "Can't detect guide line"
        if pygame.mixer.music.get_busy() == 0:
            pygame.mixer.music.load('./sound/not find.wav')
            pygame.mixer.music.play()

            # Info_cnt = 0
    print("%%%%%%%%%%%%%%%%%%%%%%%")


def Informing(state, label_map):
    global curve_degree
    global inform_txt
    global state_txt
    print("######################################")
    if state == 0:
        if state_before != 0:
            if state_before == 3:
                print("Arrived to T cross section")
                inform_txt = "Arrived to T cross section"
                state_txt = "T"

            elif state_before == 4:
                print("Arrived to right way")
                inform_txt = "Arrived to right way"
                state_txt = "R"

            elif state_before == 5:
                print("Arrived to left way")
                inform_txt = "Arrived to left way"
                state_txt = "L"

        else:
            print("Can't detect guide line")
            inform_txt = "Can't detect guide line"
            state_txt = ""

    elif state == 1:
        print("Too much Noise detected")
        inform_txt = "Too much Noise detected"

    elif state == 2:
        print("Straight line detected")
        state_txt = "S"
        if curve_degree != 0:
            if Direction_flag == 1:
                print("Slight curve for right side degree:%1f" % curve_degree)
                inform_txt = "Slight curve for right side"
                if pygame.mixer.music.get_busy() == 0:
                    pygame.mixer.music.load('./sound/right curve.wav')
                    pygame.mixer.music.play()
            elif Direction_flag == 2:
                print("Slight curve for left side degree:%1f" % curve_degree)
                inform_txt = "Slight curve for left side"
                if pygame.mixer.music.get_busy() == 0:
                    pygame.mixer.music.load('./sound/left_curve.wav')
                    pygame.mixer.music.play()
        else:
            inform_txt = "Straight line detected"
            if pygame.mixer.music.get_busy() == 0:
                pygame.mixer.music.load('./sound/straight road.wav')
                pygame.mixer.music.play()

    elif state == 3:
        print("T cross section detected")
        inform_txt = "T cross section detected"
        state_txt = "T"

        if pygame.mixer.music.get_busy() == 0:
            pygame.mixer.music.load('./sound/T road.wav')
            pygame.mixer.music.play()
    elif state == 4:
        print("Right way detected")
        inform_txt = "Right way detected"
        state_txt = "R"

        if pygame.mixer.music.get_busy() == 0:
            pygame.mixer.music.load('./sound/right road.wav')
            pygame.mixer.music.play()
    elif state == 5:
        print("Left way detected")
        inform_txt = "Left way detected"
        state_txt = "L"

        if pygame.mixer.music.get_busy() == 0:
            pygame.mixer.music.load('./sound/left road.wav')
            pygame.mixer.music.play()
    elif state == 6:
        print("End of the guide line")
        state_txt = ""
        inform_txt = "End of the guide line"

    inform_flag = 1
    if (label_map == 1).sum() / (label_map.size / 9) > 0.3:
        print("Additional Information : Cross walk detected")
        pygame.mixer.music.load('./sound/detect cross.wav')
        pygame.mixer.music.play()

    print("######################################")


def Make_Skeleton(label_map):
    Mophology_structure = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    Skeleton = np.zeros_like(label_map, dtype=np.uint8)
    label_map = cv2.blur(label_map, (5, 5))
    label_map = np.where(label_map != 0, 100, label_map)  # For debug, for skeletonize

    # cv2.imwrite('Label_map_test.png',cv2.normalize(label_map,label_map,0,255,cv2.NORM_MINMAX))

    label_map = label_map.astype(np.uint8)

    while True:
        Opening_mask = cv2.morphologyEx(label_map, cv2.MORPH_OPEN, Mophology_structure)
        temp = cv2.subtract(label_map, Opening_mask)
        Eroded = cv2.erode(label_map, Mophology_structure)
        Skeleton = cv2.bitwise_or(Skeleton, temp)
        label_map = Eroded.copy()

        if cv2.countNonZero(label_map) == 0:
            break

    return Skeleton


def Distance_measuring(row, col, center_row_point, center_col_point):
    distance_buffer = np.zeros_like(row)

    for index, row_val in enumerate(row):
        distance = (row_val - center_row_point) ** 2 + (col[index] - center_col_point) ** 2
        if distance <= 14400:
            distance_buffer[index] = distance

    return distance_buffer


def Vector_buffer_creating(distance_buffer, row, col, center_row_point, center_col_point):
    global distance_valid_ratio

    Vector_buffer = np.zeros((1, 4), dtype=np.uint16)
    max_index = np.where(distance_buffer > (np.max(distance_buffer) * distance_valid_ratio))
    max_index = np.reshape(np.asarray(max_index), np.asarray(max_index).size)
    for num_index, max_index_value in enumerate(max_index):
        endpoint_row = row[max_index_value]
        endpoint_col = col[max_index_value]
        Vector_buffer = np.append(Vector_buffer, np.reshape(np.array([center_row_point, center_col_point, endpoint_row, endpoint_col]),
                                                            (1, 4)), axis=0)
    Vector_buffer = Vector_buffer[1:]

    return Vector_buffer


def Vector_selecting(Vector_buffer):
    Principle_vector = np.zeros((1, 2), dtype=np.uint16)
    Principle_component = np.zeros((1, 4), dtype=np.uint16)

    Temp = Vector_buffer[:, 2:] - Vector_buffer[:, :2]
    Similarity_buffer = np.diag(np.arange(Temp.shape[0]) + 1)
    for j in range(Temp.shape[0]):
        Initial_vector = Temp[j]
        for i in range(Temp.shape[0]):
            if i > j:
                if cos_similarity(Initial_vector, Temp[i]) < vector_valid_angle:
                    Similarity_buffer[j, i] = j + 1

    for k in range(Temp.shape[0]):
        if np.count_nonzero(Similarity_buffer[:, k]) == 1:
            Principle_vector = np.append(Principle_vector, np.reshape(Temp[k], (1, 2)), axis=0)
            Principle_component = np.append(Principle_component, np.reshape(Vector_buffer[k], (1, 4)), axis=0)

    Principle_vector = Principle_vector[1:]
    Principle_component = Principle_component[1:]

    return Principle_vector, Principle_component


def Vector_verification(Principle_vector, Principle_component, Skeleton, Principle_M_vector, Principle_M_component):
    Vec_count = 0
    for i, vector_component in enumerate(Principle_component):
        if Principle_vector[i][1] <= 0:
            Verify_mat = Skeleton[vector_component[2]:vector_component[0] + 1,
                         vector_component[3]:vector_component[1] + 1] / 100
        else:
            Verify_mat = Skeleton[vector_component[2]:vector_component[0] + 1,
                         vector_component[1]:vector_component[3] + 1] / 100
        Count_pixel = np.sum(np.multiply(Verify_mat, np.ones_like(Verify_mat)), dtype=np.uint16)
        Diag_size = int(math.sqrt(Verify_mat.shape[0] ** 2 + Verify_mat.shape[1] ** 2))
        # print("1st region Vector verification--> %dth vector:" %(i+1), Count_pixel/Diag_size)
        if Count_pixel / Diag_size > 0.5:  ##  Need to debug
            Vec_count += 1
            # print(Vec_count_1, Count_pixel / Diag_size)
            Principle_M_vector = np.append(Principle_M_vector, np.reshape(Principle_vector[i], (1, 2)), axis=0)
            Principle_M_component = np.append(Principle_M_component, np.reshape(vector_component, (1, 4)),
                                              axis=0)

    return Principle_M_vector, Principle_M_component, Vec_count


def Multiple_vector_case_classifier(Principle_M_vector, Principle_M_component, Order, Vec_count):
    global cross_detect_ratio
    print(cross_detect_ratio)
    Principle_M_component_end = Principle_M_component[Order - 1:]
    Principle_M_vector_end = Principle_M_vector[Order - 1:]
    if Vec_count == 2:
        ##Debug
        if np.absolute((Principle_M_component_end[0][3] - Principle_M_component_end[1][3]) / (Principle_M_component_end[0][2] - Principle_M_component_end[1][2])) > cross_detect_ratio:
            if Order == 3:
                # print("T crossline appeared(end point)")
                case_num = 311
            elif Order == 2:
                # print("Soon will meet T crossline")
                case_num = 24
            elif Order == 1:
                # print("T cross line arrived")
                case_num = 14
        else:
            # Angle = cos_similarity(Principle_M_vector_end[0], Principle_M_vector_end[1])
            # if Angle > 20:
            if np.mean(Principle_M_vector_end, axis=0)[1] > 0:
                if Order == 3:
                    # print("Right side way(end point)")
                    case_num = 312
                elif Order == 2:
                    # print("Soon will meet right side way ")
                    case_num = 22
                elif Order == 1:
                    # print("Turn to right side way")
                    case_num = 12
            else:
                if Order == 3:
                    # print("Left side way(end point) ")
                    case_num = 313
                elif Order == 2:
                    # print("Soon will meet left side way ")
                    case_num = 23
                elif Order == 1:
                    # print("Turn to left side way ")
                    case_num = 13
    elif Vec_count == 3:
        if np.sum((Principle_M_vector_end > [-121, 0]).astype(np.int), axis=0)[1] == 3:
            if np.mean(Principle_M_vector_end, axis=0)[1] > 40:
                if Order == 3:
                    # print("Right side way with straight way(end point)")
                    case_num = 314
                elif Order == 2:
                    # print("Soon will meet right side way with straight way")
                    case_num = 25
                elif Order == 1:
                    # print("Arrived to right side way (with straight way)")
                    case_num = 15
            else:
                case_num = 19
        elif np.sum((Principle_M_vector_end < [1, 0]).astype(np.int), axis=0)[1] == 3:
            if np.mean(Principle_M_vector_end, axis=0)[1] < -40:
                if Order == 3:
                    # print("Left side way with straight way (end point)")
                    case_num = 315
                elif Order == 2:
                    # print("Soon will meet left side way (with straight way)")
                    case_num = 26
                elif Order == 1:
                    # print("Arrived to left side way (with straight way)")
                    case_num = 16
            else:
                case_num = 19
        else:
            T_verify = np.delete(Principle_M_component_end, np.argmin(Principle_M_vector_end, axis=0)[0], 0)
            if np.absolute((T_verify[0][3] - T_verify[1][3]) / (T_verify[0][2] - T_verify[1][2])) > cross_detect_ratio:
                if Order == 3:
                    # print("T cross line with straight way (end point)")
                    case_num = 316
                elif Order == 2:
                    # print("Soon will meet T cross line (with straight way)")
                    case_num = 27
                elif Order == 1:
                    # print("T cross line arrived (with straight way)")
                    case_num = 17
            else:
                # print("Noise,,,,")
                case_num = 19
    elif Vec_count == 4:
        Ratio_1 = np.absolute((Principle_M_component_end[0][3] - Principle_M_component_end[1][3]) / (Principle_M_component_end[0][2] - Principle_M_component_end[1][2]))
        Ratio_2 = np.absolute((Principle_M_component_end[2][3] - Principle_M_component_end[3][3]) / (Principle_M_component_end[2][2] - Principle_M_component_end[3][2]))
        if Ratio_1 > 20 and Ratio_2 > 20:
            if Order == 3:
                # print("T cross_section far from here")
                case_num = 318
            elif Order == 2:
                # print("T cross_section far from here")
                case_num = 29
            elif Order == 1:
                # print("Arrived to T cross_section")
                case_num = 110
        else:
            # print("Noise,,,,")
            case_num = 19
    else:
        if Order == 3:
            # print("Straight way forward") ## For further information
            case_num = 317
        elif Order == 2:
            # print("Straight way forward") ## For further information
            case_num = 28
        elif Order == 1:
            # print('Noise abort ')
            case_num = 18

    return case_num


def Curve_detector(Principle_M_vector, Principle_M_component):
    global curve_degree
    global Direction_flag

    #####Detecting Curve####
    Cos_sim_buffer = np.zeros(2, dtype=np.float)
    Cos_sim_buffer[0] = cos_similarity(Principle_M_vector[0], Principle_M_vector[1])
    Cos_sim_buffer[1] = cos_similarity(Principle_M_vector[1], Principle_M_vector[2])
    # Angle Thresholding (Below 5 degrees)
    Cos_sim_buffer = np.where(Cos_sim_buffer <= 5, 0, Cos_sim_buffer)
    if np.count_nonzero(Cos_sim_buffer) == 0:
        # print('Straight way forward')
        case_num = 32
        curve_degree = 0
        Direction_flag = 0
    else:
        Dominant_curve = np.argmax(Cos_sim_buffer)
        Dominant_curve_degree = np.max(Cos_sim_buffer)
        if Dominant_curve == 0:  ## Distance
            if Principle_M_component[1, 3] - Principle_M_component[0, 3] >= 0:  ## Right Side
                if Dominant_curve_degree > 60:
                    # print('Soon will meet right way,Degree:%f'%Dominant_curve_degree)
                    curve_degree = 0
                    case_num = 34
                else:
                    # print('Soon will meet right curve with %d degree' % Dominant_curve_degree)
                    case_num = 33
                    Direction_flag = 1
                    curve_degree = Dominant_curve_degree
            else:
                if Dominant_curve_degree > 60:
                    # print('Soon will meet left way,Degree:%f'%Dominant_curve_degree)
                    case_num = 36
                    curve_degree = 0
                else:
                    # print('Soon will meet left curve with %d degree' % Dominant_curve_degree)
                    case_num = 35
                    Direction_flag = 2
                    curve_degree = Dominant_curve_degree
        else:
            if Principle_M_component[2, 3] - Principle_M_component[1, 3] >= 0:  ## Right Side

                if Dominant_curve_degree > 40:
                    # print('Right way appeared,Degree:%f'%Dominant_curve_degree)
                    case_num = 38
                    curve_degree = 0
                else:
                    # print('Right curve with %d degree appeared' % Dominant_curve_degree)
                    curve_degree = Dominant_curve_degree
                    Direction_flag = 1
                    case_num = 37
            else:
                if Dominant_curve_degree > 40:
                    # print('Left way appeared ,Degree:%f'%Dominant_curve_degree)
                    case_num = 310
                    curve_degree = 0
                else:
                    # print('Left curve with %d degree appeared' % Dominant_curve_degree)
                    case_num = 39
                    Direction_flag = 2
                    curve_degree = Dominant_curve_degree
    return case_num


def Put_text(string, info_cnt):
    global arrow
    # Alarm visualizing
    # alarm_contours = np.array([[10, 280], [10, 310], [100, 310], [100, 280]])
    # cv2.fillPoly(arrow, pts=[alarm_contours], color=(255, 255, 255))
    # cv2.rectangle(arrow, (10, 280), (100, 310), (0, 0, 255), 2)
    # cv2.putText(arrow, "Sound : ", (15, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # print(info_cnt, pygame.mixer.music.get_busy())
    # if info_cnt == 0 and pygame.mixer.music.get_busy() == 0:
    #     cv2.circle(arrow, (87, 296), 5, (0, 255, 0), -1)

    # State visualizing
    state_contours = np.array([[10, 280], [10, 310], [100, 310], [100, 280]])
    cv2.fillPoly(arrow, pts=[state_contours], color=(255, 255, 255))
    cv2.rectangle(arrow, (10, 280), (100, 310), (0, 0, 255), 2)
    cv2.putText(arrow, "State : ", (15, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(arrow, state_txt, (80, 303), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    txt2_contours = np.array([[540, 260], [540, 350], [638, 350], [638, 260]])
    cv2.fillPoly(arrow, pts=[txt2_contours], color=(255, 255, 255))
    cv2.rectangle(arrow, (540, 260), (638, 350), (0, 0, 255), 2)
    cv2.putText(arrow, "S : straight", (545, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(arrow, "T : cross", (545, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(arrow, "R : Right", (545, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(arrow, "L : Left", (545, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    txt_contours = np.array([[10, 310], [10, 340], [225, 340], [225, 310]])
    cv2.fillPoly(arrow, pts=[txt_contours], color=(255, 255, 255))
    cv2.rectangle(arrow, (10, 310), (225, 340), (0, 0, 255), 2)

    cv2.putText(arrow, string, (15, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class FrameGeneratorMP4:
    def __init__(self, mp4_file: str, output_path=None, show=True):
        assert osp.isfile(mp4_file), "DATA_PATH should be existing mp4 file path."
        self.vidcap = cv2.VideoCapture(mp4_file)
        self.fps = int(self.vidcap.get(cv2.CAP_PROP_FPS))
        self.total = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.show = show
        self.output_path = output_path

        if self.output_path is not None:
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')

            if FPS_OVERRIDE is not None:
                self.fps = int(FPS_OVERRIDE)
            # self.out = cv2.VideoWriter(OUTPUT_PATH, self.fourcc, self.fps, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))## output file name, codec, fps, outputfile size

            self.out = cv2.VideoWriter(OUTPUT_PATH, self.fourcc, self.fps, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))

    def __iter__(self):
        success, image = self.vidcap.read()
        for i in range(0, self.total):
            if success:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield np.array(img)

            success, image = self.vidcap.read()

    def __len__(self):
        return self.total

    def write(self, rgb_img):

        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        if self.show:
            cv2.imshow('output', bgr)
            # cv2.imshow('output', rgb_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User Interrupted')
                self.close()
                exit(1)

        if self.output_path is not None:
            self.out.write(bgr)
            # self.out.write(rgb_img)

    def close(self):
        cv2.destroyAllWindows()
        self.vidcap.release()
        if self.output_path is not None:
            self.out.release()


class FrameGeneratorJpg:
    def __init__(self, jpg_folder: str, output_folder=None, show=True):
        assert osp.isdir(jpg_folder), "DATA_PATH should be directory including jpg files."
        self.files = sorted(glob(osp.join(jpg_folder, '*.jpg'), recursive=False))
        self.show = show
        self.output_folder = output_folder
        self.last_file_name = ""

        if self.output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)

    def __iter__(self):
        for file in self.files:
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.last_file_name = str(Path(file).name)
            yield np.array(img)

    def __len__(self):
        return len(self.files)

    def write(self, rgb_img):
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        if self.show:
            cv2.imshow('output', bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User Interrupted')
                self.close()
                exit(1)

        if self.output_folder is not None:
            path = osp.join(self.output_folder, f'{self.last_file_name}')
            cv2.imwrite(path, bgr)

    def close(self):
        cv2.destroyAllWindows()


class ModelWrapper:

    def __init__(self):
        self.composed_transform = transforms.Compose([
            transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.model = self.load_model(MODEL_PATH)

    @staticmethod
    def load_model(model_path):
        model = DeepLabv3_plus(nInputChannels=3, n_classes=NUM_CLASSES, os=16)
        if CUDA:
            model = torch.nn.DataParallel(model, device_ids=[0])
            patch_replication_callback(model)
            model = model.cuda()
        if not osp.isfile(MODEL_PATH):
            raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        if CUDA:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch: {}, best_pred: {})"
              .format(model_path, checkpoint['epoch'], checkpoint['best_pred']))
        model.eval()
        return model

    def predict(self, rgb_img: np.array):
        global frame_concat
        global cnt
        global concat_cnt
        global Info_cnt
        global non_detect_cnt

        global start_time
        global case_buffer
        global stop_buffer
        global non_detect_buffer
        global obstacle_buffer
        global obstacle_detector
        global concat_frame_num
        global state
        global state_before
        global arrow
        global inform_txt
        x = self.composed_transform(Image.fromarray(rgb_img))
        x = x.unsqueeze(0)

        if CUDA:
            x = x.cuda()
        with torch.no_grad():
            output = self.model(x)

        cnt += 1  # Frame counting
        pred = output.data.detach().cpu().numpy()
        pred_value = pred
        pred = np.argmax(pred, axis=1).squeeze(0)  ## Pred = Label map

        softmax_value = np.max(pred_value, axis=1).squeeze(0)  # softmax_value = prediction value map

        braille_label = pred
        braille_search = np.zeros_like(braille_label, dtype=np.uint8)

        softmax_value = np.where(braille_label != 2, 0, softmax_value)  ## Non-block label eliminating

        ##Predict value visualizing
        ##################################################################################################
        # softmax_value = cv2.normalize(softmax_value,softmax_value,0,255,cv2.NORM_MINMAX)
        # softmax_value = np.where(braille_label != 2, 0, softmax_value)
        # softmax_value = softmax_value.astype(np.uint8)
        ##################################################################################################
        ##Predict value thresholded visualizing
        ##################################################################################################
        # softmax_value = cv2.normalize(softmax_value,softmax_value,0,255,cv2.NORM_MINMAX)
        # softmax_value = np.where(braille_label != 2, 0, softmax_value)
        # softmax_value = softmax_value.astype(np.uint8)
        # Mean_predict = np.mean(softmax_value)
        #
        # #softmax_value = np.where(braille_label != 2, 0, softmax_value)
        # label_num, connected_img = cv2.connectedComponents(softmax_value,connectivity=4)
        # #cv2.imwrite('d.png',connected_img*40)
        #
        # if label_num > 1: # if there is at least one object
        #     label_num_buffer = np.zeros(label_num,dtype= np.uint16)
        #     Dominant_label_map = np.zeros_like(connected_img)
        #     for label_index in range(1, label_num):
        #             label_num_buffer[label_index] = np.sum(connected_img == label_index)
        #             temp = np.mean(softmax_value[np.where(connected_img == label_index)])
        #             #Mean Prediction value Thresholding
        #             if temp < Mean_predict * 0.3:
        #                 softmax_value[np.where(connected_img == label_index)] = 0
        ##Skeleton visualize
        ####################################################################################################
        # skel_visualize = Make_Skeleton(softmax_value)
        # skel_visualize = (skel_visualize).astype(np.uint8)
        ####################################################################################################

        Mean_predict = np.mean(softmax_value)
        frame_concat += softmax_value  ## Concat for 10 value map
        section_ratio = np.zeros(3, dtype=np.float)

        if cnt == concat_frame_num:  ## Per 10 frames
            case_num = 11
            start_time = timeit.default_timer()
            concat_cnt += 1
            Dominant_label_map = np.zeros_like(frame_concat)
            # if concat_cnt >= 16 and concat_cnt <= 20:
            #     im_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            #     cv2.imwrite("test_image%d.png"%concat_cnt,im_bgr)
            # cv2.imwrite('Arrow_img%d_%d.png'%(concat_cnt,case_num),img_show)
            print("##")
            print("Concat_count: %d" % concat_cnt)
            if np.any(frame_concat != 0):  ## If no guide_block detected for 10 frame

                # softmax_value_norm = np.round(255*(frame_concat - np.min(frame_concat)) / (np.max(frame_concat)-np.min(frame_concat))).astype(np.uint8)
                # softmax_value_norm = np.where(braille_label != 2, 0 , softmax_value_norm)
                #
                # # Global Thresholding for noise reduction
                # softmax_value_norm = np.where(softmax_value_norm < 20, 0, softmax_value_norm)
                #
                # if np.any(softmax_value_norm!=0):
                #########Connected Component Labeling for Thresholded value#####
                braille_search[np.where(softmax_value != 0)] = 255  # Just for binary image
                label_num, connected_img = cv2.connectedComponents(braille_search, connectivity=4)
                # cv2.imwrite('d.png',connected_img*40)

                if label_num > 1:  # if there is at least one object
                    label_num_buffer = np.zeros(label_num, dtype=np.uint16)

                    for label_index in range(1, label_num):
                        label_num_buffer[label_index] = np.sum(connected_img == label_index)
                        temp = np.mean(softmax_value[np.where(connected_img == label_index)])
                        # Mean Prediction value Thresholding
                        if temp < Mean_predict * 0.3:
                            softmax_value[np.where(connected_img == label_index)] = 0

                    Dominant_label_map = np.where(connected_img == np.argmax(label_num_buffer), 1, Dominant_label_map)

                if np.any(softmax_value != 0):

                    for i in range(3):
                        Temp = softmax_value[:, int((softmax_value.shape[1] / 3) * i):int((softmax_value.shape[1] / 3) * (i + 1))]
                        Temp_nonzero = np.count_nonzero(Temp)
                        section_ratio[i] = Temp_nonzero / np.size(Temp)

                    if np.argmax(section_ratio) == 1 and np.max(section_ratio) > 0.2:  ##Center Valid ratio

                        Skeleton = Make_Skeleton(softmax_value)
                        img_show = Skeleton
                        # --------------------------------------------------------------------------------------------------------------------------#
                        if np.any(Skeleton != 0):
                            ## STEP1 ##
                            Valid_braille_map = np.multiply(Skeleton, Dominant_label_map)
                            row, col = np.where(Valid_braille_map != 0)

                            center_row = np.max(row)
                            center_col = col[np.argmax(row)]

                            row, col = np.where(Skeleton == 100)

                            row_1 = row[np.where(row >= 240)]
                            col_1 = col[np.argwhere(row >= 240)].squeeze()
                            # center_row = np.max(row)
                            # center_col = max_col
                            center_point_buffer = np.zeros((1, 2), dtype=np.uint16)
                            Principle_M_vector = np.zeros((1, 2), dtype=np.uint16)
                            Principle_M_component = np.zeros((1, 4), dtype=np.uint16)

                            # center_point_buffer = np.append(center_point_buffer, np.reshape(np.array([center_row, center_col]), (1, 2)),axis=0)
                            # center_point_buffer = center_point_buffer[1:]

                            if row_1.size:

                                distance_buffer = Distance_measuring(row_1, col_1, center_row, center_col)
                                Vector_buffer = Vector_buffer_creating(distance_buffer, row_1, col_1, center_row, center_col)
                                ## Vector Selecting##
                                Principle_vector, Principle_component = Vector_selecting(Vector_buffer)
                                ## Vector  Verification In Frist step ##

                                Principle_M_vector, Principle_M_component, Vec_count_1 = Vector_verification(Principle_vector, Principle_component, Skeleton, Principle_M_vector, Principle_M_component)
                                Principle_M_vector = Principle_M_vector[1:]
                                Principle_M_component = Principle_M_component[1:]
                                ## Classify by case ##
                                ## On the Straight Line##
                                if Vec_count_1 == 1:

                                    center_row = Principle_M_component[0, 2]
                                    center_col = Principle_M_component[0, 3]
                                    ##Stacking center point##
                                    # center_point_buffer = np.append(center_point_buffer,np.reshape(np.array([center_row, center_col]), (1, 2)), axis=0)
                                    ## Step 2 ##
                                    row_2 = row[np.where((row < 240) & (row >= 120))]
                                    col_2 = col[np.argwhere((row < 240) & (row >= 120))].squeeze()

                                    if row_2.size:

                                        distance_buffer = Distance_measuring(row_2, col_2, center_row, center_col)
                                        Vector_buffer = Vector_buffer_creating(distance_buffer, row_2, col_2, center_row, center_col)
                                        Principle_vector, Principle_component = Vector_selecting(Vector_buffer)
                                        Principle_M_vector, Principle_M_component, Vec_count_2 = Vector_verification(Principle_vector, Principle_component, Skeleton, Principle_M_vector,
                                                                                                                     Principle_M_component)
                                        ## Connected Line ##
                                        if Vec_count_2 == 1:  ## More condition required

                                            center_row = Principle_M_component[1, 2]
                                            center_col = Principle_M_component[1, 3]
                                            ##Center_point_stacking
                                            # center_point_buffer = np.append(center_point_buffer,np.reshape(np.array([center_row, center_col]), (1, 2)), axis=0)
                                            row_3 = row[np.where(row < 120)]
                                            col_3 = col[np.argwhere(row < 120)].squeeze()

                                            if row_3.size:
                                                distance_buffer = Distance_measuring(row_3, col_3, center_row, center_col)
                                                Vector_buffer = Vector_buffer_creating(distance_buffer, row_3, col_3, center_row, center_col)
                                                Principle_vector, Principle_component = Vector_selecting(Vector_buffer)
                                                Principle_M_vector, Principle_M_component, Vec_count_3 = Vector_verification(Principle_vector, Principle_component, Skeleton, Principle_M_vector,
                                                                                                                             Principle_M_component)
                                                if concat_cnt >= 16 and concat_cnt <= 30:
                                                    arrow_show(img_show, Principle_M_component, concat_cnt, concat_cnt)
                                                ## Step 3 ##
                                                if Vec_count_3 == 1:  ## More condition required
                                                    case_num = Curve_detector(Principle_M_vector, Principle_M_component)
                                                elif Vec_count_3 == 0:
                                                    # print("Guide way will soon end")
                                                    case_num = 31
                                                else:
                                                    case_num = Multiple_vector_case_classifier(Principle_M_vector, Principle_M_component, 3, Vec_count_3)
                                        elif Vec_count_2 == 0:
                                            # print('Straight way will soon end')
                                            case_num = 21
                                        ## Middle point cross line
                                        else:
                                            case_num = Multiple_vector_case_classifier(Principle_M_vector, Principle_M_component, 2, Vec_count_2)
                                elif Vec_count_1 == 0:
                                    # print("Can't find guide block")
                                    case_num = 11
                                else:
                                    case_num = Multiple_vector_case_classifier(Principle_M_vector, Principle_M_component, 1, Vec_count_1)
                                arrow = arrow_write(cv2.resize(rgb_img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT), interpolation=cv2.INTER_NEAREST), Principle_M_component, concat_cnt, case_num)

                        else:
                            arrow = arrow_write(cv2.resize(rgb_img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT), interpolation=cv2.INTER_NEAREST), np.array([[0, 0, 0, 0]]), concat_cnt, case_num)
                    else:
                        arrow = arrow_write(cv2.resize(rgb_img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT), interpolation=cv2.INTER_NEAREST), np.array([[0, 0, 0, 0]]), concat_cnt, case_num)
                else:
                    arrow = arrow_write(cv2.resize(rgb_img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT), interpolation=cv2.INTER_NEAREST), np.array([[0, 0, 0, 0]]), concat_cnt, case_num)
            else:
                arrow = arrow_write(cv2.resize(rgb_img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT), interpolation=cv2.INTER_NEAREST), np.array([[0, 0, 0, 0]]), concat_cnt, case_num)
            # --------------------------------------------------------------------------------------------------------------------------#
            if np.any(np.array([11, 21, 31, 18]) == case_num):
                Info_cnt = 0
                if case_num == 11:
                    non_detect_cnt += 1
                    if non_detect_cnt == 10:
                        Stop_Informing(0)
                        stop_buffer = np.ones(10, dtype=np.uint8)
                        case_buffer = np.ones(10, dtype=np.uint8) * 9
                        non_detect_cnt = 0
                stop_buffer = Buffering(stop_buffer)
                # print(np.bincount(stop_buffer))
                if np.bincount(stop_buffer)[0] >= 3:
                    Temp_buffer = np.bincount(case_buffer)[2:]
                    Dominant_state = np.argmax(Temp_buffer) + 2
                    if Dominant_state == 2:
                        if Temp_buffer[0] > 7:
                            Stop_Informing(Dominant_state)
                            non_detect_buffer = np.ones(10, dtype=np.uint8)
                        else:
                            Temp_buffer_RLT = np.bincount(case_buffer)[3:]
                            Dominant_state = np.argmax(Temp_buffer_RLT) + 3
                            Stop_Informing(Dominant_state)
                            non_detect_buffer = np.ones(10, dtype=np.uint8)
                    else:
                        Stop_Informing(Dominant_state)
            else:
                # Stop, non_detect case clearing
                non_detect_cnt = 0
                stop_buffer = np.ones(10, dtype=np.uint8)
                case_buffer = case_buffering(concat_cnt, case_num, case_buffer)
                state_before = state

                if concat_cnt <= 10:
                    state_temp = np.bincount(case_buffer)[:case_buffer.size]
                    if np.max(state_temp) >= 3:
                        state = np.argmax(state_temp)
                    if state != state_before:
                        Informing(state, braille_label)
                        Info_cnt = 0
                    elif state == state_before:
                        Info_cnt += 1
                        if Info_cnt == 7:
                            Informing(state, braille_label)
                            Info_cnt = 0
                else:
                    state = np.argmax(weighted_case(case_buffer))
                    if state != state_before:
                        if np.max(weighted_case(case_buffer)) >= 25:
                            Informing(state, braille_label)
                            Info_cnt = 0

                    elif state == state_before:
                        Info_cnt += 1
                        if Info_cnt == 5:
                            Informing(state, braille_label)
                            Info_cnt = 0

            Put_text(inform_txt, Info_cnt)
            print("case: ", case_num)

            print(case_buffer)
            print(stop_buffer)
            print(non_detect_buffer)
            cnt = 0
            frame_concat = np.zeros((360, 640), dtype=np.float)
            terminate_time = timeit.default_timer()
            # print("%f sec consumed." % (terminate_time - start_time))
            # print("##")
        # cv2.imwrite('softmax_norm_threshold.jpg',softmax_value_norm)

        segmap = decode_segmap(pred, dataset='custom', label_colors=CUSTOM_COLOR_MAP, n_classes=CUSTOM_N_CLASSES)
        segmap = np.array(segmap * 255).astype(np.uint8)
        resized = cv2.resize(segmap, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT), interpolation=cv2.INTER_NEAREST)

        return arrow


#       return cv2.resize(rgb_img, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),interpolation=cv2.INTER_NEAREST)
#       return resized
#       return softmax_value

def main():
    global arrow

    print('Loading model...')
    model_wrapper = ModelWrapper()

    if MODE == 'mp4':
        generator = FrameGeneratorMP4(DATA_PATH, OUTPUT_PATH, show=SHOW_OUTPUT)
        # generator = FrameGeneratorMP4(OUTPUT_PATH,show=SHOW_OUTPUT)
    elif MODE == 'jpg':
        generator = FrameGeneratorJpg(DATA_PATH, OUTPUT_PATH, show=SHOW_OUTPUT)

    elif MODE == 'live':
        cap = cv2.VideoCapture(0)
        livefps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        liveout = cv2.VideoWriter(OUTPUT_PATH, fourcc, livefps, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
        while (True):

            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame)
            segmap = model_wrapper.predict(frame)
            if OVERLAPPING:
                h, w, _ = np.array(segmap).shape
                img_resized = cv2.resize(frame, (w, h))
                result = (img_resized * 0.5 + segmap * 0.5).astype(np.uint8)
            else:
                result = segmap

            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            liveout.write(result)
        liveout.release()
        cap.release()
        cv2.destroyAllWindows()
    else:
        raise NotImplementedError('MODE should be "mp4" or "jpg".')

    for index, img in enumerate(generator):

        segmap = model_wrapper.predict(img)
        if OVERLAPPING:
            h, w, _ = np.array(segmap).shape
            img_resized = cv2.resize(img, (w, h))
            result = (img_resized * 0.5 + segmap * 0.5).astype(np.uint8)
        else:
            result = segmap
        generator.write(result)

    generator.close()
    print('Done.')


if __name__ == '__main__':
    main()
