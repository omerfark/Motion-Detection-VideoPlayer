import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import threading
from playsound import playsound
import pygame
import cv2
import numpy as np

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")

        self.btn_select_video = tk.Button(self.root, text="Select Video", command=self.select_video)
        self.btn_select_video.pack()

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Control Buttons
        self.btn_play = tk.Button(self.root, text="Start", command=self.play_video)
        self.btn_play.pack(side=tk.LEFT)

        self.btn_pause = tk.Button(self.root, text="Stop", command=self.pause_video)
        self.btn_pause.pack(side=tk.LEFT)

        # Coordinates of the selected area
        self.coordinates = []

        # Coordinate entries
        self.entries = []
        for i in range(4):
            frame = tk.Frame(self.root)
            frame.pack()

            tk.Label(frame, text=f"Point {i+1} X:").pack(side=tk.LEFT)
            x_entry = tk.Entry(frame)
            x_entry.pack(side=tk.LEFT)
            x_entry.insert(0, '0')
            x_entry.bind("<KeyRelease>", self.update_coordinates)

            tk.Label(frame, text=f"Point {i+1} Y:").pack(side=tk.LEFT)
            y_entry = tk.Entry(frame)
            y_entry.pack(side=tk.LEFT)
            y_entry.insert(0, '0')
            y_entry.bind("<KeyRelease>", self.update_coordinates)

            self.entries.append((x_entry, y_entry))

        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, width=640, height=360, bg='black')
        self.canvas.pack()

        # Movement detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2() # Background subtractor

        self.cap = None 
        self.playing = False
        self.current_frame = 0 
        self.previous_frame = None 
        self.alert_playing = False
        self.dragging_point = None

        # Canvas click and drag events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag) 

        # Pygame sound initialization
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("alert2.wav") 

        #----- Yovo V3 tiny ------
        model_cfg = 'yolov3-tiny.cfg'
        model_weights = 'yolov3-tiny.weights'
        class_file = 'coco.names'

        # Load YOLO
        self.net = cv2.dnn.readNet(model_weights, model_cfg)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]

        #Coco classes
        with open(class_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect_and_classify_objects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob) 
        outputs = self.net.forward(self.output_layers)

        for output in outputs: 
            for detection in output: 
                scores = detection[5:] 
                class_id = np.argmax(scores) 
                confidence = scores[class_id] 
                if confidence > 0.5: 
                    center_x, center_y, w, h = map(int, detection[:4] * np.array([width, height, width, height])) 
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    label = self.classes[class_id]

                    # Check if the object's center is inside the shaded area
                    if cv2.pointPolygonTest(np.array(self.coordinates, np.int32), (center_x, center_y), False) >= 0:
                        if label == 'person':
                            print('Human Detected!')
                        elif label == 'car':
                            print('Car Detected!')
                        elif label in ['dog', 'cat']:
                            print('Animal Detected!')

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def select_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Folder", "*.mp4 *.avi *.mov *.mkv")])
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.playing = False
            self.update_frame()

    def on_click(self, event):
        if len(self.coordinates) < 4:
            self.coordinates.append((event.x, event.y))
            self.update_labels()
        else:
            # control if the click is on a point
            for i, (x, y) in enumerate(self.coordinates):
                if abs(event.x - x) < 10 and abs(event.y - y) < 10:
                    self.dragging_point = i
                    break

    # Dragging the points
    def on_drag(self, event):
        if self.dragging_point is not None:
            x, y = event.x, event.y
            self.coordinates[self.dragging_point] = (event.x, event.y)
            self.update_labels()
            self.update_frame()

    def update_labels(self): 
        for i, (x, y) in enumerate(self.coordinates):
            self.entries[i][0].delete(0, tk.END)
            self.entries[i][0].insert(0, str(x))
            self.entries[i][1].delete(0, tk.END)
            self.entries[i][1].insert(0, str(y))
        self.draw_coordinates()

    def draw_coordinates(self):
        self.canvas.delete("all")

        for i, (x, y) in enumerate(self.coordinates):
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red')
            self.canvas.create_text(x, y-15, text=f'Point {i+1}', fill='white')

        # Combine the coordinates
        if len(self.coordinates) == 4:
            self.canvas.create_line(self.coordinates[0], self.coordinates[1], fill='green', width=2)
            self.canvas.create_line(self.coordinates[1], self.coordinates[2], fill='green', width=2)
            self.canvas.create_line(self.coordinates[2], self.coordinates[3], fill='green', width=2)
            self.canvas.create_line(self.coordinates[3], self.coordinates[0], fill='green', width=2)

    def update_frame(self):
        if self.cap is not None and self.playing:
            ret, frame = self.cap.read()
            if not ret:
                return

            # Canvas siyah arka plan
            canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
            frame_height, frame_width = frame.shape[:2]

            scaled_coordinates = []
            for (x, y) in self.coordinates:
                scaled_x = int(x * frame_width / canvas_width)
                scaled_y = int(y * frame_height / canvas_height)
                scaled_coordinates.append((scaled_x, scaled_y))

            # Videodaki gölgeli alan
            if len(scaled_coordinates) == 4:
                mask = np.zeros_like(frame)
                points = np.array(scaled_coordinates, np.int32)
                cv2.fillPoly(mask, [points], (255, 255, 255)) 
                masked_frame = cv2.bitwise_and(frame, mask)
            else:
                masked_frame = frame

            # Hareket algılama (Motion Detection)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            fg_mask = self.bg_subtractor.apply(gray)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # minimum alan
                    x, y, w, h = cv2.boundingRect(contour) 
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
                    motion_detected = True

            # Hareket algılama sonucuna göre alarm çalma
            if motion_detected and not self.alert_playing:
                self.alert_playing = True
                threading.Thread(target=self.play_alert).start()
            elif not motion_detected:
                self.alert_playing = False

            # Yolo nesne tespiti ve sınıflandırma
            frame = self.detect_and_classify_objects(frame)

            # Orijinal kare boyutuna koordinatları ölçekleme
            if len(scaled_coordinates) == 4:
                pts = np.array(scaled_coordinates, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Video boyutlandırma 
            frame = cv2.resize(frame, (640, 360))
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.config(image=photo)
            self.video_label.image = photo

            self.root.after(int(1000 / self.cap.get(cv2.CAP_PROP_FPS)), self.update_frame)



    def play_alert(self):
        self.alert_sound.play()  

    def play_video(self):
        if len(self.coordinates) == 4 and not self.playing:
            self.playing = True
            self.update_frame()

    def pause_video(self):
        self.playing = False

    def update_coordinates(self, event=None):
        # Update the coordinates from the entries
        self.coordinates = []
        for (x_entry, y_entry) in self.entries:
            try:
                x = int(x_entry.get())
                y = int(y_entry.get())
                self.coordinates.append((x, y))
            except ValueError:
                self.coordinates.append((0, 0))
        self.draw_coordinates()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()
