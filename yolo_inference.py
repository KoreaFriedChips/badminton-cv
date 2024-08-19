from ultralytics import YOLO

model = YOLO('models/best.pt')

model.predict('input_videos/LDvLCW Asia Championships.mov', save=True)