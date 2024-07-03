from PyQt5 import QtWidgets, uic, QtCore
import cv2
import numpy as np

class FishDetectionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(FishDetectionApp, self).__init__()
        uic.loadUi('gui.ui', self)
        self.detector = FishDetector('yolo/custom-yolov4-tiny-detector_2000.weights', 'yolo/yolov4-obj.cfg', 'yolo/fish.names')
        self.Browsebtn.clicked.connect(self.browse_file)
        self.Playbtn.clicked.connect(self.play_video)
        self.Exportbtn.clicked.connect(self.export_video)
        self.Pausebtn.clicked.connect(self.pause_video)
        self.bermudabtn.clicked.connect(lambda: self.set_class_filter('bermuda'))
        self.hogfishbtn.clicked.connect(lambda: self.set_class_filter('hogfish'))
        self.sergeantbtn.clicked.connect(lambda: self.set_class_filter('sergeant'))
        self.stripedbtn.clicked.connect(lambda: self.set_class_filter('striped'))
        self.stingraybtn.clicked.connect(lambda: self.set_class_filter('stingray'))
        self.stop = False
        self.pause = False
        self.output_path = 'output.avi'  # Default output path
        self.class_filter = None

    def set_class_filter(self, fish_class):
        self.class_filter = fish_class

    def browse_file(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video", "", "All Files (*);;MP4 Files (*.mp4);;AVI Files (*.avi)", options=options)
        if file_name:
            self.lineEdit_path.setText(file_name)

    def play_video(self):
        video_path = self.lineEdit_path.text()
        if video_path:
            self.process_video(video_path)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
        while cap.isOpened():
            if self.stop:
                break
            if not self.pause:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = self.detector.detect(frame)
                # Draw bounding boxes
                for detection in detections:
                    for box in detection:
                        x, y, w, h = box[0:4]
                        class_id = int(box[5])
                        label = str(self.detector.classes[class_id])
                        if self.class_filter is None or label == self.class_filter:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                out.write(frame)
                cv2.imshow('video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def export_video(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Video", "", "AVI Files (*.avi);;All Files (*)", options=options)
        if file_name:
            # Ensure the output video writer is released properly
            if hasattr(self, 'out'):
                self.out.release()
            # Update the output file path
            self.output_path = file_name
            self.Exportbtn.setText('Exported')
            self.Exportbtn.setEnabled(False)

    def pause_video(self):
        self.pause = not self.pause

class FishDetector:
    def __init__(self, weights_path, config_path, classes_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        with open(classes_path, 'r') as f:
            self.classes = f.read().strip().split("\n")

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        detections = self.net.forward(output_layers)
        return detections

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = FishDetectionApp()
    mainWindow.show()
    sys.exit(app.exec_())

