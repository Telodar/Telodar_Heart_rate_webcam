import cv2
import dlib
import numpy as np
from imutils import face_utils
import imutils
from collections import OrderedDict
import time

# Dlib'ten yüz tespiti yapacak nesneyi oluştur
detector = dlib.get_frontal_face_detector()

# Dlib'ten yüz landmarklarını tespit edecek nesneyi oluştur
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Video akışını aç
vs = cv2.VideoCapture(0)

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

        # Her bir indeksi yüzün belirli bir noktasının yanına yaz
        for (i, (x, y)) in enumerate(shape):
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)



    # Frame'i göster
    cv2.imshow("Frame", frame)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Video akışını serbest bırak ve pencereleri kapat
vs.release()
cv2.destroyAllWindows()
