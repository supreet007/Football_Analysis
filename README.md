# Football_Analysis

## Overview
This project implements an automated football (soccer) video analysis system that detects, tracks, and analyzes players, referees, and the ball during a match. The system leverages computer vision and machine learning techniques to extract meaningful statistics such as player team assignment, speed, distance covered, ball possession, and camera movement.

## Features
- **Player and Object Detection:** Uses YOLOv5 trained on custom datasets to detect players, referees, goalkeepers, and the ball.
- **Tracking:** Tracks individual entities over time using OpenCV-based tracking and ellipse-based bounding shapes for better visualization.
- **Team Assignment:** Automatically clusters player jersey colors with KMeans to assign players to their respective teams.
- **Ball Possession:** Assigns ball possession to the nearest player based on distance thresholds.
- **Camera Movement Estimation:** Estimates camera motion using optical flow to adjust player positions and provide stable tracking.
- **Speed and Distance Calculation:** Calculates player speed (km/h) and total distance covered (meters) using frame-to-frame tracking data and field perspective transformation.
- **Visualization:** Annotates video frames with player speed, distance, team colors, and camera movement metrics.
- **Perspective Transformation:** Transforms camera view to a top-down field perspective for more accurate spatial measurements.

## Project Structure
```
football-video-analysis/
│
├── main.py                        # Main script to run the full pipeline
├── modules/                       # Custom modules
│   ├── team_assigner.py           # TeamAssigner class (jersey color clustering)
│   ├── speed_distance_estimator.py # SpeedDistanceEstimator class
│   ├── player_ball_assigner.py    # PlayerBallAssigner class (ball possession)
│   ├── camera_movement.py         # CameraMovementEstimator class
│   └── ...                       # Other utility scripts
├── utils/                        # Utility functions (bbox utilities, measurement helpers)
├── datasets/                     # Dataset for training YOLOv5
├── outputs/                      # Output annotated videos and intermediate results
├── requirements.txt              # Required Python packages
└── README.md                     # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/supreet007/football_analysis.git
   cd football_analysis
   ```

2. Download or prepare the YOLOv5 model and dataset:
   - The project uses a YOLOv5 model trained on a custom Roboflow dataset.
   - Place the trained weights in the appropriate directory.

## Usage

Run the main script on a football match video:

```bash
python main.py --input_video path/to/video.mp4 --output_video path/to/output.mp4
```

The script will:
- Detect and track players, referees, and the ball
- Assign players to teams based on jersey color clustering
- Estimate player speed and distance covered
- Track ball possession
- Estimate and visualize camera movement
- Output an annotated video with all relevant metrics

## Modules Summary

### TeamAssigner
- Uses KMeans clustering on player jersey colors extracted from player bounding boxes.
- Assigns each player a team ID based on dominant jersey color clusters.

### SpeedDistanceEstimator
- Calculates speed and total distance covered by players over a sliding window of frames.
- Annotates frames with speed (km/h) and distance (meters).

### PlayerBallAssigner
- Assigns ball possession to the closest player within a threshold distance.
- Uses bounding box and center point distance metrics.

### CameraMovementEstimator
- Estimates camera movement frame-to-frame using Lucas-Kanade optical flow on selected feature points.
- Adjusts player positions for stable tracking despite camera motion.
- Visualizes X and Y camera movement metrics on video frames.

## Notes
- Frame rate is assumed to be 24 FPS for speed and distance calculations.
- The perspective transformer module converts camera views to a top-down field view for more accurate distance estimation.
- The system uses elliptical bounding shapes to better approximate player positions.

## Future Improvements
- Integrate more advanced multi-object tracking algorithms.
- Incorporate event detection (passes, shots, fouls).
- Real-time processing optimization.
- Support for multiple camera angles and dynamic camera calibration.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

