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
import time


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel()
        self.average_red_change_plot = PlotWidget("Average Red Change")
        self.pulse_estimation_plot = PlotWidget("Pulse Estimation")
        self.pulse_label = QLabel("Pulse: -- BPM")
        self.pulse_label.setAlignment(Qt.AlignCenter)
        self.pulse_label.setStyleSheet("font-size: 24px; color: red;")

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setAlignment(Qt.AlignCenter)
        self.fps_label.setStyleSheet("font-size: 18px; color: blue;")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.average_red_change_plot)
        layout.addWidget(self.pulse_estimation_plot)
        layout.addWidget(self.pulse_label)
        layout.addWidget(self.fps_label)

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
        self.fft_window_size = 128  # Size of the FFT window
        self.pulse_estimation_buffer = []  # Buffer to store average red change values
        self.pulse_estimation_frequency = None  # Estimated pulse frequency
        self.frame_rate = 15  # Assuming a frame rate of 30 FPS
        self.fps = None  # Initialize fps variable

        # Variables for FPS calculation
        self.frame_count = 0
        self.start_time = time.time()

        # Pencere boyutlarını ayarlıyoruz
        self.resize(1920, 1080)  # Genişlik: 1920, Yükseklik: 1080

    def update_frame(self):
        ret, frame = self.vs.read()
        if not ret:
            return  # VideoCapture'den frame alınamadıysa çıkış yap

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

        # FPS calculation
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.fps_label.setText(f"FPS: {self.fps:.2f}")
            self.frame_count = 0
            self.start_time = time.time()

        if rects:  # Check if a face is detected
            rect = rects[0]
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            forehead_rectangle = signal_processing.draw_forehead_rectangle(frame, shape)
            left_cheek_rectangle, right_cheek_rectangle = signal_processing.draw_cheek_rectangles(frame, shape)

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

            # Calculate the average red change
            average_red_change = (forehead_change + left_cheek_change + right_cheek_change) / 3

            # Update average red change buffer
            self.pulse_estimation_buffer.append(average_red_change)
            if len(self.pulse_estimation_buffer) > self.fft_window_size:
                self.pulse_estimation_buffer.pop(0)  # Remove oldest value

            # Perform FFT and pulse estimation
            if len(self.pulse_estimation_buffer) == self.fft_window_size:
                try:
                    fft_data = np.fft.fft(self.pulse_estimation_buffer)
                    fft_frequencies = np.fft.fftfreq(len(fft_data), d=3 / self.fps)  # Convert to Hz

                    # Identify the dominant frequency in the expected pulse range (0.75-3 Hz, 45-180 BPM)
                    indices = np.where((fft_frequencies > 0.75) & (fft_frequencies < 2))
                    fft_data_filtered = np.abs(fft_data[indices])
                    fft_frequencies_filtered = fft_frequencies[indices]

                    if len(fft_data_filtered) > 0:
                        dominant_frequency_index = np.argmax(fft_data_filtered)
                        self.pulse_estimation_frequency = fft_frequencies_filtered[dominant_frequency_index]

                        # Update pulse label
                        pulse_bpm = self.pulse_estimation_frequency * 60  # Convert to BPM
                        if 45 <= pulse_bpm <= 120:  # Valid BPM range check
                            self.pulse_label.setText(f"Pulse: {pulse_bpm:.2f} BPM")
                        else:
                            self.pulse_label.setText("Pulse: -- BPM")

                        # Update pulse estimation plot
                        self.pulse_estimation_plot.update_plot(pulse_bpm)  # Convert to BPM
                    else:
                        self.pulse_label.setText("Pulse: -- BPM")
                except Exception as e:
                    print(f"Error in pulse estimation: {e}")
                    self.pulse_label.setText("Pulse: -- BPM")

            self.average_red_change_plot.update_plot(average_red_change)

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