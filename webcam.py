import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils
from collections import OrderedDict
import time

from main import draw_forehead_rectangle, draw_cheek_rectangles, vs, detector, predictor
from main import fps,fps_counter,start_time


def calculate_color_signal(image, rectangle):
    # Dikdörtgenin koordinatlarını al
    x1, y1, x2, y2 = rectangle

    # Dikdörtgenin içindeki bölgenin alanını al
    roi = image[y1:y2, x1:x2]

    # ROI'daki renk ortalamasını hesapla
    mean_color = np.mean(roi, axis=(0, 1))

    # Örneğin, yeşil rengin ortalaması 100'den büyükse, yeşil bir sinyal gönderelim
    if mean_color[1] > 100:
        return "Green Signal"
    else:
        return "No Signal"



# Dikdörtgenlerin içindeki renk sinyallerini hesaplayarak ekranda gösterme
def show_color_signals(image, forehead_rectangle, left_cheek_rectangle, right_cheek_rectangle):
    # Sarı dikdörtgenin içindeki renk sinyalini hesapla
    forehead_signal = calculate_color_signal(image, forehead_rectangle)

    # Sol yanak dikdörtgeninin içindeki renk sinyalini hesapla
    left_cheek_signal = calculate_color_signal(image, left_cheek_rectangle)

    # Sağ yanak dikdörtgeninin içindeki renk sinyalini hesapla
    right_cheek_signal = calculate_color_signal(image, right_cheek_rectangle)

    # Renk sinyallerini ekranda göster
    cv2.putText(image, f"Forehead Signal: {forehead_signal}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Left Cheek Signal: {left_cheek_signal}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Right Cheek Signal: {right_cheek_signal}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# Renk sinyallerini hesapla ve gösterme işlevini mevcut kodunuzun döngüsüne ekleyin:
while True:
    # Video akışından bir frame al
    ret, frame = vs.read()
    # Frame'i yatay olarak ters çevir
    frame = cv2.flip(frame, 1)

    # Görüntüyü griye dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti yap
    rects = detector(gray, 0)

    # Yüz bulunduğunda
    for rect in rects:
        # Yüz landmarklarını tespit et
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Alın bölgesini gösteren sarı dikdörtgeni çiz
        draw_forehead_rectangle(frame, shape)

        # Sağ ve sol yanağı temsil eden kare dikdörtgenleri çiz
        draw_cheek_rectangles(frame, shape)

        # Renk sinyallerini hesapla ve ekranda göster
        show_color_signals(frame, forehead_rectangle=(100, 100, 200, 300),  # Örnek koordinatlar
                           left_cheek_rectangle=(150, 150, 250, 250),     # Örnek koordinatlar
                           right_cheek_rectangle=(350, 150, 450, 250))    # Örnek koordinatlar

    # FPS sayacını güncelle
    fps_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = fps_counter / elapsed_time
        fps_counter = 0
        start_time = time.time()

    # FPS'yi ekrana yazdır
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Frame'i göster
    cv2.imshow("Frame", frame)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
