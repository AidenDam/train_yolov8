from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

results = model.train(data='custom-dataset.yaml', epochs=1, imgsz=640, pretrained='True')
