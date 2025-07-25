if __name__ == '__main__':
    from ultralytics import YOLO
    import torch

    print("CUDA Available:", torch.cuda.is_available())
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    model = YOLO("yolov8n.pt")

    model.train(
        data="botol_kaleng_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=24,
        project="hasil_traning_botol_kaleng",
        name="botol_kaleng_model",
        device=0  # paksa pakai GPU kalau ada
    )
