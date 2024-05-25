import numpy as np
import cv2

def draw_forehead_rectangle(frame, shape):
    # Alın dikdörtgeni oluşturma işlemleri
    # Örnek bir hesaplama:
    x = int((shape[21][0] + shape[22][0]) / 2)
    y = int((shape[21][1] + shape[22][1]) / 2) - 50
    return (x-30, y-30, x+30, y+30)

def draw_cheek_rectangles(frame, shape):
    # Yanak dikdörtgenleri oluşturma işlemleri
    # Örnek bir hesaplama:
    left_cheek = (shape[1][0], shape[29][1], shape[3][0], shape[33][1])
    right_cheek = (shape[15][0], shape[29][1], shape[13][0], shape[33][1])
    return left_cheek, right_cheek

def calculate_red_mean(frame, rectangle):
    x1, y1, x2, y2 = rectangle
    region = frame[y1:y2, x1:x2]
    red_mean = np.mean(region[:, :, 0])  # Kırmızı kanalın ortalamasını hesapla
    return red_mean

def calculate_red_change(current_mean, previous_mean):
    if previous_mean is None:
        return 0
    return current_mean - previous_mean
