from utils import (read_video, save_video)
from trackers import (PlayerTracker, BallTracker)
import cv2 
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

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

    # get key points
    key_point_extractor = CourtLineDetector()
    court_keypoints = key_point_extractor.predict(video_frames[0])

    # mini court
    mini_court = MiniCourt(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # draw output

    ## draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## draw court key points
    output_video_frames = key_point_extractor.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    # output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    # output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))

    ## draw frame number on the top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()