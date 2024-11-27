import sys
import cv2
import mediapipe as mp
import time
import os
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette
from PyQt5.QtCore import QTimer

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class WaterIntakeApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up window with dark_freeze theme
        self.setWindowTitle("Water Intake Tracker")
        self.setGeometry(100, 100, 800, 600)

        # Apply dark theme colors
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(33, 33, 33))         # Dark background
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))  # White text
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(44, 44, 44))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Highlight, QColor(142, 45, 197))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(dark_palette)

        # Set up layout and widgets
        self.label = QLabel(self)
        self.sips_label = QLabel("Sips: 0", self)
        self.cups_label = QLabel("Cups: 0", self)
        self.close_button = QPushButton("Close", self)

        # Initialize counters
        self.sip_count = 0
        self.cup_count = 0
        self.sips_per_cup = 10  # Set average sips per cup

        # Set up main layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.sips_label)
        layout.addWidget(self.cups_label)
        layout.addWidget(self.close_button)

        # Set up central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize video capture and timers
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        # MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Connect close button
        self.close_button.clicked.connect(self.close_app)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # Process hand tracking
        hand_results = self.hands.process(rgb_frame)

        # Initialize flag for hand near face
        hand_near_face = False

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_center = (x + w // 2, y + h // 2)

            # Check if hand is detected
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Get the coordinates of the wrist
                    wrist_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1])
                    wrist_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame.shape[0])

                    # Check distance between wrist and face center
                    distance = ((wrist_x - face_center[0]) ** 2 + (wrist_y - face_center[1]) ** 2) ** 0.5
                    if distance < 50:  # Adjust this distance for accuracy
                        hand_near_face = True

        # If hand is near face, increment sip count
        if hand_near_face:
            self.sip_count += 1
            time.sleep(0.5)  # Delay to avoid multiple counts for the same sip action

        # Calculate cup count
        self.cup_count = self.sip_count // self.sips_per_cup

        # Update GUI labels
        self.sips_label.setText(f"Sips: {self.sip_count}")
        self.cups_label.setText(f"Cups: {self.cup_count}")

        # Display the video feed in GUI
        qt_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def close_app(self):
        # Release resources and close the app
        self.cap.release()
        self.close()  # This will end the application without needing to call self.hands.close()

# Run the app with dark_freeze styling
app = QApplication(sys.argv)
app.setStyle('Fusion')  # Set Fusion style for uniform dark theme support
window = WaterIntakeApp()
window.show()
sys.exit(app.exec_())
