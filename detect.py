from ultralytics import YOLO
import cv2

# Load model hasil training
model = YOLO("hasil_traning_botol_kaleng/botol_kaleng_model2/weights/best.pt")

# Pilihan sumber:
# source = 0            # Webcam
# source = "path.jpg"   # Gambar
# source = "video.mp4"  # Video
source = 1  # pakai webcam

cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, imgsz=640)

    # Visualisasi hasil
    annotated_frame = results[0].plot()

    # Tampilkan
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
