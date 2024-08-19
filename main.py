from utils import (read_video, save_video)
from trackers import (PlayerTracker, BallTracker)
import cv2 

def main():
    # read video
    input_video_path = 'input_videos/input_video.mov'
    video_frames = read_video(input_video_path)

    # detect players and ball
    player_tracker = PlayerTracker('yolov8x.pt')
    ball_tracker = BallTracker('models/last.pt')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # draw output

    ## draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## draw frame number on the top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()