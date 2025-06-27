import os
import cv2
import numpy as np
from tensorflow.keras import models, layers, optimizers, losses, metrics, datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import RPi.GPIO as GPIO

BUZZER_PIN = 17
MOTOR_PINS = [18, 23, 24, 25]

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
for pin in MOTOR_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

def stop_motors():
    for pin in MOTOR_PINS:
        GPIO.output(pin, GPIO.LOW)


def escape_behavior(duration=5):
    GPIO.output(MOTOR_PINS[0], GPIO.HIGH)
    GPIO.output(MOTOR_PINS[1], GPIO.LOW)
    GPIO.output(MOTOR_PINS[2], GPIO.HIGH)
    GPIO.output(MOTOR_PINS[3], GPIO.LOW)
    
    time.sleep(duration)
    stop_motors()


motion_detector = cv2.createBackgroundSubtractorMOG2(historyLength=500, varThreshold=20, detectShadows=True)

def has_motion(frame, areas=5000):
    fgmask = motion_detector.apply(frame)
    _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > areas:
            return True
    return False


class ImageClassifier:
    def __init__(self,
                 input_shape=(32, 32, 3),
                 num_classes=10,
                 class_names=None,
                 model_path=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = class_names or ['deer','dog', 'frog', 'horse', 'ship', 'truck']
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)


    def build_model(self):
        m = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        m.compile(
            optimizer=optimizers.Adam(),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[metrics.SparseCategoricalAccuracy()]
        )
        self.model = m
        return m

    def train(self,
              x_train, y_train,
              x_val=None, y_val=None,
              epochs=10,
              batch_size=64,
              augment=False):
        if self.model is None:
            self.build_model()

        if augment:
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True
            )
            datagen.fit(x_train)
            history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                validation_data=(x_val, y_val) if x_val is not None else None,
                epochs=epochs,
                steps_per_epoch=len(x_train) // batch_size
            )
        else:
            history = self.model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val) if x_val is not None else None,
                epochs=epochs,
                batch_size=batch_size
            )
        return history

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model

    def preprocess_frame(self, frame):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        img = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, frame):
        img_array = self.preprocess_frame(frame)
        preds = self.model.predict(img_array, verbose=0)[0]
        confidence = np.max(preds)
        class_idx = np.argmax(preds)
        
        if confidence < 0.7:
            return "unknown", 0.0, False
        return self.class_names[class_idx], confidence, class_idx

class SafetyMonitor:
    def __init__(self, classifier, dangerous_animals):
        self.classifier = classifier
        self.dangerous_animals = dangerous_animals
        self.last_detection_time = 0
        self.alarm_active = False
        
    def process_frame(self, frame):
        animal, confidence, _ = self.classifier.predict(frame)
        is_dangerous = animal in self.dangerous_animals
        
        if animal != "unknown":
            print(f"Detected: {animal} ({confidence:.2f}) | Dangerous: {is_dangerous}")
            
        return animal, is_dangerous    

def shuffle_split(x, y, split=0.2):
    indices = np.random.permutation(len(x))
    split_idx = int(len(x) * split)
    return (x[indices[split_idx:]], y[indices[split_idx:]], 
            x[indices[:split_idx]], y[indices[:split_idx]])

def main():
    DANGEROUS_ANIMALS = ['snake', 'bear', 'wolf', 'lion']
    MODEL_PATH = 'animal_detector.h5'
    CLASS_NAMES = ['cat', 'dog', 'deer', 'bird', 'snake', 'bear', 'background']
    INPUT_SHAPE = (128, 128, 3)
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    classifier = ImageClassifier(input_shape=INPUT_SHAPE, class_names=CLASS_NAMES, model_path=MODEL_PATH)
    safety_system = SafetyMonitor(classifier, DANGEROUS_ANIMALS)
    frame_counter = 0
    SKIP_FRAMES = 3
    
    last_capture_time = time.time()
    CAPTURE_INTERVAL = 10
    
    ALARM_DURATION = 5
    alarm_start_time = 0
    
    try:
        print("Starting animal detection system. Press Ctrl+C to exit.")
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Error reading camera frame")
                break
                
            frame_counter = (frame_counter + 1) % SKIP_FRAMES
            if frame_counter != 0:
                time.sleep(0.05)
                continue
                
            motion = has_motion(frame)
            current_time = time.time()
            
            if motion or (current_time - last_capture_time >= CAPTURE_INTERVAL):
                last_capture_time = current_time
                
                animal, is_dangerous = safety_system.process_frame(frame)
                
                if is_dangerous:
                    print("DANGER! Triggering alarm and escape behavior")
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    escape_behavior()
                    alarm_start_time = time.time()
                    safety_system.alarm_active = True
            
            if safety_system.alarm_active and (time.time() - alarm_start_time > ALARM_DURATION):
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                safety_system.alarm_active = False
                
            cv2.imshow('Animal Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        stop_motors()
        GPIO.cleanup()
        print("System shutdown complete")