import cv2
import numpy as np

def draw_forehead_rectangle(image, shape):
    # Burada OpenCV'nin dikdörtgen çizme işlevini kullanacağız
    left_forehead_index = 19
    right_forehead_index = 24
    left_forehead = shape[left_forehead_index]
    right_forehead = shape[right_forehead_index]
    forehead_width = int(np.linalg.norm(left_forehead - right_forehead))
    forehead_center = ((left_forehead[0] + right_forehead[0]) // 2, (left_forehead[1] + right_forehead[1]) // 2)
    rectangle_height = forehead_width // 4
    upper_left = (forehead_center[0] - forehead_width // 2, forehead_center[1] - rectangle_height - 20)
    bottom_right = (forehead_center[0] + forehead_width // 2, forehead_center[1] + rectangle_height - 20)
    cv2.rectangle(image, upper_left, bottom_right, (0, 255, 255), 2)
    return (upper_left[0], upper_left[1], bottom_right[0], bottom_right[1])

def draw_cheek_rectangles(image, shape):
    # Burada da OpenCV'nin dikdörtgen çizme işlevini kullanacağız
    left_cheek_index = 31
    right_cheek_index = 35
    left_cheek = shape[left_cheek_index]
    right_cheek = shape[right_cheek_index]
    rectangle_side_length = int(np.linalg.norm(left_cheek - right_cheek))
    left_cheek_center = (left_cheek[0] - 30, left_cheek[1] - 25)
    right_cheek_center = (right_cheek[0] + 30, right_cheek[1] - 25)
    cv2.rectangle(image, (left_cheek_center[0] - rectangle_side_length // 2, left_cheek_center[1] - rectangle_side_length // 2),
                  (left_cheek_center[0] + rectangle_side_length // 2, left_cheek_center[1] + rectangle_side_length // 2), (0, 0, 255), 2)
    cv2.rectangle(image, (right_cheek_center[0] - rectangle_side_length // 2, right_cheek_center[1] - rectangle_side_length // 2),
                  (right_cheek_center[0] + rectangle_side_length // 2, right_cheek_center[1] + rectangle_side_length // 2), (0, 0, 255), 2)
    return ((left_cheek_center[0] - rectangle_side_length // 2, left_cheek_center[1] - rectangle_side_length // 2,
             left_cheek_center[0] + rectangle_side_length // 2, left_cheek_center[1] + rectangle_side_length // 2),
            (right_cheek_center[0] - rectangle_side_length // 2, right_cheek_center[1] - rectangle_side_length // 2,
             right_cheek_center[0] + rectangle_side_length // 2, right_cheek_center[1] + rectangle_side_length // 2))

def calculate_red_mean(image, rectangle):
    x1, y1, x2, y2 = rectangle
    roi = image[y1:y2, x1:x2]
    mean_red = np.mean(roi[:, :, 2])  # Kırmızı kanal (BGR formatında 2. kanal)
    return mean_red

def calculate_red_change(current_mean, previous_mean):
    if previous_mean is None:
        return 0
    return current_mean - previous_mean
