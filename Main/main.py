from utils.video_utils import read_video, save_video 
from trackers.trackers import Tracker
from Team_Assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_distance_estimator.speed_distance_estimator import SpeedDistanceEstimator
import numpy as np

import cv2

def main():
    # Read Video
    video_frames = read_video('E:\\supreet\\personal prjt\\ML prjts\\Football_analysis\\input_data.mp4')


    # Initialize Tracker
    tracker = Tracker('model\\best.pt')

    # Get Tracks
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True ,
                                       stub_path='stubs/track_stubs.pkl')
    
    #Get object positions
    tracker.add_position_to_tracks(tracks)

    # Get Camera Movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    
    # Get View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)





    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    #Speed and Distance Estimation
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    #Assign Player Teams
    team_assigner = TeamAssigner()

    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num,player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],track["bbox"], player_id)
            tracks['players'][frame_num][player_id]["team"] = team
            tracks['players'][frame_num][player_id]["team_color"] = team_assigner.team_color[team]
                
    # Ensure tracks have the same number of frames as video_frames
    while len(tracks["players"]) < len(video_frames):
        tracks["players"].append({})
    
    while len(tracks["referee"]) < len(video_frames):
        tracks["referee"].append({})
    
    while len(tracks["ball"]) < len(video_frames):
        tracks["ball"].append({})

    # Assign Player-Ball
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_player_ball(player_track,ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    #Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw speed and distance
    speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
