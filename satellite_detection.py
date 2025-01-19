import cv2
import numpy as np
import os

# Directories for input and output videos
input_folder = "videos_to_process"
output_folder = "videos_processed"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Parameters for persistence and clustering
persistence_threshold = 1  # Minimum frames an object must persist
movement_tolerance = 5  # Maximum pixel movement between frames to track the same object
draw_frequency = 1  # Times per second to draw boxes
clustering_tolerance = 200  # Grid cell size for clustering
post_detection_frames = 24  # Number of additional frames to include after a detection

# Process each video file in the input folder
for video_file in os.listdir(input_folder):
    if not video_file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        continue

    input_path = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, video_file)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_path}. Skipping.")
        continue

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the dynamic EXCLUDE_REGION (bottom 6.5% of the screen)
    exclude_height = int(frame_height * 0.065)
    region_y_start = frame_height - exclude_height

    # Define left and right exclusion regions (1/3 of the width each)
    left_region = {"x_start": 0, "x_end": frame_width // 3}
    right_region = {"x_start": ((2 * frame_width) // 3) + (frame_width // 7), "x_end": frame_width}

    # Calculate dynamic box size (5% of vertical frame size)
    box_size = max(1, int(frame_height * 0.05))  # Ensure at least 1 pixel

    # Calculate frame interval for drawing boxes
    draw_interval = max(1, int(fps / draw_frequency))  # Ensure at least one frame

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {video_file}")
    print(f"Left exclude region: {left_region}, Right exclude region: {right_region}")
    print(f"Dynamic box size: {box_size}x{box_size}")
    print(f"Draw interval: {draw_interval} frames")
    print(f"Post-detection frames: {post_detection_frames}")

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Unable to read frames from video {video_file}. Skipping.")
        cap.release()
        continue

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    include_next_frames = 0  # Counter to track additional frames to include

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(curr_gray, prev_gray)

        # Apply a small amount of noise reduction
        frame_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)

        # Exclude the dynamically calculated regions
        cv2.rectangle(frame_diff, (left_region["x_start"], region_y_start), 
                      (left_region["x_end"], frame_height), 0, -1)  # Mask left region
        cv2.rectangle(frame_diff, (right_region["x_start"], region_y_start), 
                      (right_region["x_end"], frame_height), 0, -1)  # Mask right region

        # Apply thresholding to detect motion
        _, thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        # Get detected points
        non_black_pixels = np.argwhere(thresh > 0)
        detected_points = []

        for pixel in non_black_pixels:
            y, x = pixel
            detected_points.append((x, y))

        # Group detections using grid-based clustering
        clusters = {}
        for x, y in detected_points:
            grid_x = x // clustering_tolerance
            grid_y = y // clustering_tolerance
            grid_key = (grid_x, grid_y)
            if grid_key not in clusters:
                clusters[grid_key] = []
            clusters[grid_key].append((x, y))

        # Create bounding boxes for each cluster
        clustered_boxes = []
        for cluster_points in clusters.values():
            x_coords = [p[0] for p in cluster_points]
            y_coords = [p[1] for p in cluster_points]
            x_center = int(np.mean(x_coords))
            y_center = int(np.mean(y_coords))

            # Adjust bounding box size using box_size
            x_min = x_center - box_size // 2
            x_max = x_center + box_size // 2
            y_min = y_center - box_size // 2
            y_max = y_center + box_size // 2
            clustered_boxes.append((x_min, y_min, x_max, y_max))

        # Check for satellite detection
        satellites_detected = False
        if frame_count % draw_interval == 0:  # Only draw boxes every `draw_interval` frames
            for (x_min, y_min, x_max, y_max) in clustered_boxes:
                cv2.rectangle(curr_frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
                satellites_detected = True

        if satellites_detected:
            include_next_frames = post_detection_frames  # Trigger inclusion of additional frames

        if satellites_detected or include_next_frames > 0:
            out.write(curr_frame)
            include_next_frames -= 1  # Decrease counter for additional frames

        # Display the frame difference and output frame with the exclusion rectangles in real time
        display_frame = curr_frame.copy()
        cv2.rectangle(display_frame, (left_region["x_start"], region_y_start), 
                      (left_region["x_end"], frame_height), (0, 0, 255), 2)
        cv2.rectangle(display_frame, (right_region["x_start"], region_y_start), 
                      (right_region["x_end"], frame_height), (0, 0, 255), 2)
        cv2.imshow("Output Frame with Boxes and Exclusion Areas", display_frame)

        prev_gray = curr_gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    print(f"Finished processing video: {video_file}. Output saved to {output_path}")

cv2.destroyAllWindows()
print("All videos processed.")
