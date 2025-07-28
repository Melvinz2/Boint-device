from ultralytics import YOLO
import cv2
import os
from datetime import datetime

class YOLODetector:
    """
    Class untuk YOLOv8 Detection yang dapat digunakan untuk berbagai source
    """
    
    def __init__(self, model_path="hasil_traning_botol_kaleng/botol_kaleng_model2/weights/best.pt"):
        """
        Initialize YOLO detector
        
        Args:
            model_path (str): Path ke model weights
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file tidak ditemukan: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            print(f"[{datetime.now()}] Model YOLOv8 berhasil dimuat dari {self.model_path}")
            return True
        except Exception as e:
            print(f"[{datetime.now()}] Error loading model: {e}")
            self.model = None
            return False
    
    def detect_image(self, image_path, conf_threshold=0.5, imgsz=640):
        """
        Detect objects dalam gambar
        
        Args:
            image_path (str): Path ke file gambar
            conf_threshold (float): Confidence threshold
            imgsz (int): Image size untuk inference
            
        Returns:
            dict: Hasil deteksi dengan format:
                {
                    'labels': ['botol', 'kaleng'],
                    'confidences': [0.85, 0.92],
                    'boxes': [box1, box2],
                    'total_objects': 2,
                    'unique_labels': ['botol', 'kaleng']
                }
        """
        if self.model is None:
            raise Exception("Model belum dimuat atau gagal dimuat")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
        
        try:
            # Load gambar
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Gagal membaca gambar")
            
            # Inference
            results = self.model(image, imgsz=imgsz, conf=conf_threshold)
            
            # Extract informasi deteksi
            detected_labels = []
            confidences = []
            boxes = []
            
            for result in results:
                for box in result.boxes:
                    if box.conf >= conf_threshold:
                        class_id = int(box.cls)
                        label = self.model.names[class_id]
                        confidence = float(box.conf)
                        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        
                        detected_labels.append(label)
                        confidences.append(confidence)
                        boxes.append(bbox)
            
            # Get unique labels
            unique_labels = list(set(detected_labels))
            
            return {
                'labels': detected_labels,
                'confidences': confidences,
                'boxes': boxes,
                'total_objects': len(detected_labels),
                'unique_labels': unique_labels,
                'model_names': dict(self.model.names)
            }
            
        except Exception as e:
            raise Exception(f"Error dalam deteksi: {str(e)}")
    
    def detect_frame(self, frame, conf_threshold=0.5, imgsz=640):
        """
        Detect objects dalam frame (untuk webcam/video)
        
        Args:
            frame: OpenCV frame
            conf_threshold (float): Confidence threshold
            imgsz (int): Image size untuk inference
            
        Returns:
            tuple: (annotated_frame, detection_results)
        """
        if self.model is None:
            raise Exception("Model belum dimuat atau gagal dimuat")
        
        try:
            # Inference
            results = self.model(frame, imgsz=imgsz, conf=conf_threshold)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Extract informasi deteksi
            detected_labels = []
            confidences = []
            boxes = []
            
            for result in results:
                for box in result.boxes:
                    if box.conf >= conf_threshold:
                        class_id = int(box.cls)
                        label = self.model.names[class_id]
                        confidence = float(box.conf)
                        bbox = box.xyxy[0].tolist()
                        
                        detected_labels.append(label)
                        confidences.append(confidence)
                        boxes.append(bbox)
            
            unique_labels = list(set(detected_labels))
            
            detection_results = {
                'labels': detected_labels,
                'confidences': confidences,
                'boxes': boxes,
                'total_objects': len(detected_labels),
                'unique_labels': unique_labels
            }
            
            return annotated_frame, detection_results
            
        except Exception as e:
            raise Exception(f"Error dalam deteksi frame: {str(e)}")
    
    def webcam_detection(self, source=0, window_name="YOLOv8 Detection"):
        """
        Real-time detection dari webcam (original functionality)
        
        Args:
            source: Camera source (0, 1, atau path ke video)
            window_name (str): Nama window untuk display
        """
        if self.model is None:
            print("Error: Model belum dimuat")
            return
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Tidak dapat membuka source {source}")
            return
        
        print(f"[{datetime.now()}] Memulai deteksi webcam dari source {source}")
        print("Tekan 'q' untuk keluar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame")
                break
            
            try:
                # Inference dan visualisasi
                annotated_frame, detection_results = self.detect_frame(frame)
                
                # Tampilkan informasi deteksi di terminal
                if detection_results['unique_labels']:
                    labels_str = ', '.join(detection_results['unique_labels'])
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Terdeteksi: {labels_str} ({detection_results['total_objects']} objek)", end='')
                
                # Tampilkan frame
                cv2.imshow(window_name, annotated_frame)
                
            except Exception as e:
                print(f"\nError dalam deteksi: {e}")
                cv2.imshow(window_name, frame)
            
            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[{datetime.now()}] Deteksi webcam dihentikan")
    
    def get_model_info(self):
        """Get informasi model"""
        if self.model is None:
            return None
        
        return {
            'model_path': self.model_path,
            'model_names': dict(self.model.names),
            'num_classes': len(self.model.names)
        }

# Untuk backward compatibility dengan kode lama
if __name__ == "__main__":
    # Original functionality
    detector = YOLODetector()
    detector.webcam_detection(source=1)
