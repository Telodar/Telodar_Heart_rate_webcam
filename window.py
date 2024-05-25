import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import dlib
from imutils import face_utils
import signal_processing

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel()
        self.forehead_plot = PlotWidget("Forehead Red Change")
        self.left_cheek_plot = PlotWidget("Left Cheek Red Change")
        self.right_cheek_plot = PlotWidget("Right Cheek Red Change")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)

        plots_layout = QHBoxLayout()
        plots_layout.addWidget(self.forehead_plot)
        plots_layout.addWidget(self.left_cheek_plot)
        plots_layout.addWidget(self.right_cheek_plot)
        layout.addLayout(plots_layout)

        self.central_widget.setLayout(layout)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.vs = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 milliseconds per frame

        self.previous_forehead_mean = None
        self.previous_left_cheek_mean = None
        self.previous_right_cheek_mean = None

        # Pencere boyutlarını ayarlıyoruz
        self.resize(1920, 1080)  # Genişlik: 1920, Yükseklik: 1080

    def update_frame(self):
        ret, frame = self.vs.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        painter = QPainter(pixmap)
        pen = QPen(Qt.yellow)
        pen.setWidth(2)
        painter.setPen(pen)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            forehead_rectangle = signal_processing.draw_forehead_rectangle(frame, shape)
            left_cheek_rectangle, right_cheek_rectangle = signal_processing.draw_cheek_rectangles(frame, shape)

            # Drawing rectangles on the video frame using PyQt5
            self.draw_rectangle(painter, forehead_rectangle)
            self.draw_rectangle(painter, left_cheek_rectangle)
            self.draw_rectangle(painter, right_cheek_rectangle)

            current_forehead_mean = signal_processing.calculate_red_mean(frame, forehead_rectangle)
            current_left_cheek_mean = signal_processing.calculate_red_mean(frame, left_cheek_rectangle)
            current_right_cheek_mean = signal_processing.calculate_red_mean(frame, right_cheek_rectangle)

            forehead_change = signal_processing.calculate_red_change(current_forehead_mean, self.previous_forehead_mean)
            left_cheek_change = signal_processing.calculate_red_change(current_left_cheek_mean,
                                                                       self.previous_left_cheek_mean)
            right_cheek_change = signal_processing.calculate_red_change(current_right_cheek_mean,
                                                                        self.previous_right_cheek_mean)

            self.forehead_plot.update_plot(forehead_change)
            self.left_cheek_plot.update_plot(left_cheek_change)
            self.right_cheek_plot.update_plot(right_cheek_change)

            self.previous_forehead_mean = current_forehead_mean
            self.previous_left_cheek_mean = current_left_cheek_mean
            self.previous_right_cheek_mean = current_right_cheek_mean

        painter.end()

        self.video_label.setPixmap(pixmap)

    def draw_rectangle(self, painter, rectangle):
        x1, y1, x2, y2 = rectangle
        width = x2 - x1
        height = y2 - y1
        painter.drawRect(x1, y1, width, height)

class PlotWidget(QWidget):
    def __init__(self, title):
        super().__init__()

        self.figure = plt.figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title)
        self.ax.set_xlabel('Frames')
        self.ax.set_ylabel('Red Change')
        self.lines, = self.ax.plot([], [], 'b-')
        self.x_data = []
        self.y_data = []

        layout = QHBoxLayout()  # Yatay düzen kullanılıyor
        layout.addWidget(self.canvas)

        # Yeni bir widget oluşturup, diğer widget'ları içine yerleştiriyoruz
        widget = QWidget()
        widget.setLayout(layout)

        # Ana düzeneğe bu widget'ı ekliyoruz
        main_layout = QVBoxLayout()
        main_layout.addWidget(widget)
        self.setLayout(main_layout)

    def update_plot(self, y):
        self.y_data.append(y)
        self.x_data.append(len(self.y_data))
        self.lines.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
