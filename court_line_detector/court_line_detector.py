import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self):
        self.id = 0

    def get_manual_keypoints(self):
        manual_keypoints = np.array([
            # row 0 (far court back line)
            604, 575, 
            675, 575,
            1048, 575,
            1420, 575,
            1485, 575,

            # row 1 (far court doubles service line)
            597, 600, 
            665, 600,
            1048, 600,
            1425, 600,
            1492, 600,

            # row 2 (far court service line)
            535, 725,
            618, 725,
            1048, 725,
            1472, 725,
            1552, 725,

            # row 3 (close court service line)
            455, 900,
            550, 900,
            1048, 900,
            1545, 900,
            1637, 900,

            # row 4 (close court doubles service line)
            350, 1137,
            455, 1137,
            1050, 1137,
            1645, 1137,
            1748, 1137,

            # row 5 (close court back line)
            330, 1187,
            434, 1187,
            1052, 1185,
            1666, 1187,
            1780, 1187,

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
