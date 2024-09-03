import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# class CourtLineDetector:
#     def __init__(self):
#         # No need to load a model for manual keypoints
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def get_manual_keypoints(self, original_w, original_h):
#         # Replace with your manually determined keypoints
#         manual_keypoints = np.array([
#             436, 492, 484, 493, 755, 492, 1025, 493, 1073, 493,
#             429, 506, 479, 508, 755, 508, 1032, 508, 1081, 508,

#         ], dtype=np.float32)

#         # Scale points to the original image size
#         manual_keypoints[::2] *= original_w / 224.0
#         manual_keypoints[1::2] *= original_h / 224.0
#         return manual_keypoints

#     def predict(self, image):
#         original_h, original_w = image.shape[:2]
#         keypoints = self.get_manual_keypoints(original_w, original_h)
#         print("Keypoints: ", keypoints)
#         return keypoints

class CourtLineDetector:
    def __init__(self):
        # # No need to load a model for manual keypoints
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.id = 0

    def get_manual_keypoints(self):
        # Replace with your actual values, using the coordinates you got from the image editor
        manual_keypoints = np.array([
            600, 575,
            1485, 575,
            330, 1190,
            1785, 1190,

        ], dtype=np.float32)

        # Since you're working with the original dimensions, no scaling is necessary
        return manual_keypoints

    def predict(self, image):
        keypoints = self.get_manual_keypoints()
        print("Keypoints: ", keypoints)
        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            point_number = i // 2 + 1  # Point number starts from 1
            # print(f"Drawing point {point_number} at ({x}, {y})")
            cv2.putText(image, str(point_number), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
