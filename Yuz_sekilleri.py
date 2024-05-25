import cv2
import dlib
import numpy as np
from imutils import face_utils
import time

# Dlib'ten yüz tespiti yapacak nesneyi oluştur
detector = dlib.get_frontal_face_detector()

# Dlib'ten yüz landmarklarını tespit edecek nesneyi oluştur
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Alın bölgesini gösteren sarı dikdörtgenin çizilmesi için fonksiyon
def draw_forehead_rectangle(image, shape):
    # Alın bölgesinin sol ve sağ kenarlarını belirleyen landmarkların indekslerini al
    left_forehead_index = 19
    right_forehead_index = 24

    # Sol ve sağ alın kenarlarının koordinatlarını al
    left_forehead = shape[left_forehead_index]
    right_forehead = shape[right_forehead_index]

    # Alt kenar uzunluğunu hesapla
    forehead_width = int(np.linalg.norm(left_forehead - right_forehead))

    # Alın bölgesinin merkezini hesapla
    forehead_center = ((left_forehead[0] + right_forehead[0]) // 2, (left_forehead[1] + right_forehead[1]) // 2)

    # Dikdörtgenin yüksekliğini belirle (genişliğin 1/4'ü olarak ayarlanmıştı)
    rectangle_height = forehead_width // 4

    # Dikdörtgenin üst ve alt kenarlarını 2 cm yukarı kaydır
    upper_left = (forehead_center[0] - forehead_width // 2, forehead_center[1] - rectangle_height - 20)
    bottom_right = (forehead_center[0] + forehead_width // 2, forehead_center[1] + rectangle_height - 20)

    # Alın bölgesine sarı dikdörtgen çiz
    cv2.rectangle(image, upper_left, bottom_right, (0, 255, 255), 2)


# Sağ ve sol yanağı seçen ve kare dikdörtgen çizen fonksiyon
def draw_cheek_rectangles(image, shape):
    # Sol ve sağ yanağı temsil eden landmarkların indekslerini al
    left_cheek_index = 31
    right_cheek_index = 35

    # Sol ve sağ yanağın koordinatlarını al
    left_cheek = shape[left_cheek_index]
    right_cheek = shape[right_cheek_index]

    # Kare dikdörtgenlerinin kenar uzunluğunu sol ve sağ yanağın arasındaki mesafe olarak ayarla
    rectangle_side_length = int(np.linalg.norm(left_cheek - right_cheek))

    # Sol ve sağ yanağın merkezini hesapla
    left_cheek_center = (left_cheek[0] - 30, left_cheek[1] - 25)
    right_cheek_center = (right_cheek[0] + 30, right_cheek[1] - 25)

    # Kare dikdörtgenleri çiz
    cv2.rectangle(image, (left_cheek_center[0] - rectangle_side_length // 2, left_cheek_center[1] - rectangle_side_length // 2),
                  (left_cheek_center[0] + rectangle_side_length // 2, left_cheek_center[1] + rectangle_side_length // 2), (0, 0, 255), 2)
    cv2.rectangle(image, (right_cheek_center[0] - rectangle_side_length // 2, right_cheek_center[1] - rectangle_side_length // 2),
                  (right_cheek_center[0] + rectangle_side_length // 2, right_cheek_center[1] + rectangle_side_length // 2), (0, 0, 255), 2)


# Renk sinyalini hesaplayacak fonksiyon
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


# Renk sinyallerini hesaplayacak ve gösterecek fonksiyon
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


# Video akışını aç
vs = cv2.VideoCapture(0)

# FPS sayacını başlat
fps_counter = 0
start_time = time.time()
fps = 0

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

# Video akışını serbest bırak ve pencereleri kapat
vs.release()
cv2.destroyAllWindows()
