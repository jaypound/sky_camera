# Satellite Detection and Video Processing

## Overview
This Python program processes sky camera videos to detect and track moving objects such as UFOs, UAPs, planes and satellites. Using frame differencing and clustering, the program identifies small, slow-moving objects, excludes static noise, and dynamically highlights detections with bounding boxes. The output is a filtered video with improved continuity for detected objects by including additional frames after detections.

## Features
- **Frame Differencing**: Identifies motion by calculating differences between consecutive video frames.
- **Noise Reduction**: Applies Gaussian blur and thresholding to minimize false positives.
- **Dynamic Clustering**: Groups nearby detections to form unified bounding boxes for large objects like planes or satellites.
- **Dynamic Bounding Boxes**: Automatically adjusts the size of bounding boxes based on video resolution.
- **Continuity Enhancement**: Includes additional frames after a detection to ensure smooth tracking in the output video.
- **Exclusion Zones**: Masks regions of the frame (e.g., timestamp areas) that should not be processed.
- **Real-Time Display**: Visualizes detections with bounding boxes during processing.
- **Configurable Parameters**: Easily adjust frame persistence, clustering tolerance, and detection frequency.

## Parameters
| Parameter                | Description                                                                                          | Default Value |
|--------------------------|------------------------------------------------------------------------------------------------------|---------------|
| `persistence_threshold`  | Minimum frames an object must persist before being considered valid.                                | `1`           |
| `movement_tolerance`     | Maximum pixel movement between frames to track the same object.                                     | `5`           |
| `draw_frequency`         | Number of times per second to draw bounding boxes.                                                  | `1`           |
| `clustering_tolerance`   | Maximum distance between detections to group them into a cluster.                                   | `10`          |
| `post_detection_frames`  | Number of frames to include in the output after a detection is made.                                | `2`           |
| `box_size`               | Percentage of the vertical frame size used to determine the size of bounding boxes.                 | `5%`          |

## Usage
1. **Input Folder**: Place your video files in the `videos_to_process` directory.
2. **Output Folder**: Processed videos with detections will be saved in the `videos_processed` directory.
3. **Run the Script**:
   ```bash
   python sky_camera.py
   


