import cv2
import numpy as np
import os
import json
from datetime import datetime

# Directories for input and output videos
input_folder = "videos_to_process"
output_folder = "videos_processed"
completed_folder = "videos_completed"
live_folder = "live"

# Ensure folders exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(completed_folder, exist_ok=True)
os.makedirs(live_folder, exist_ok=True)

# Parameters for persistence and clustering
persistence_threshold = 2
movement_tolerance = 10
draw_frequency = 1
clustering_tolerance = 200
post_detection_frames = 90

camera_config_file = "camera_config.json"

def _create_video_writer(width: int, height: int, fps: float, camera_name: str) -> (cv2.VideoWriter, str):
    timestamp_str = datetime.now().strftime("%Y-%m-%d-%H")
    filename = f"{camera_name}_{timestamp_str}.mp4"
    output_path = os.path.join(live_folder, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"--> Starting new file: {filename}")
    return writer, timestamp_str

def process_capture(cap: cv2.VideoCapture, output_path: str, is_live: bool = False, camera_name: str = "LiveCamera"):
    if not cap.isOpened():
        print(f"Error: Unable to open capture source. Skipping.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    exclude_height = int(frame_height * 0.04)
    region_y_start = frame_height - exclude_height
    left_region = {"x_start": 0, "x_end": frame_width // 8}
    right_region = {"x_start": ((2 * frame_width) // 3) + (frame_width // 8), "x_end": frame_width}
    box_size = max(1, int(frame_height * 0.05))
    draw_interval = max(1, int(fps // draw_frequency))

    if not is_live:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        current_hour_str = ""
    else:
        out, current_hour_str = _create_video_writer(frame_width, frame_height, fps, camera_name)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read first frame. Skipping.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    include_next_frames = 0
    persistent_objects = {}
    next_object_id = 0
    
    # Initialize frame buffer
    frame_buffer = []
    buffer_size = int(fps * 5)  # 5 seconds buffer
    first_detection = True

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Add current frame to buffer
        frame_buffer.append(curr_frame.copy())
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        if is_live:
            new_hour_str = datetime.now().strftime("%Y-%m-%d-%H")
            if new_hour_str != current_hour_str:
                out.release()
                out, current_hour_str = _create_video_writer(frame_width, frame_height, fps, camera_name)

        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        """
        Gaussian Blur: Smaller kernel size preserves fainter signals
        Increasing sensitivity may also increase false positives. 
        Start by lowering the threshold value before adjusting blur.
        """
        frame_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0) # Default is (5, 5)

        cv2.rectangle(frame_diff, (left_region["x_start"], region_y_start),
                     (left_region["x_end"], frame_height), 0, -1)
        cv2.rectangle(frame_diff, (right_region["x_start"], region_y_start),
                     (right_region["x_end"], frame_height), 0, -1)

        # Lower values increase sensitivity to faint movements
        _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY) # Default is 20
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        non_black_pixels = np.argwhere(thresh > 0)
        detected_points = [(x, y) for (y, x) in non_black_pixels]

        clusters = {}
        for x, y in detected_points:
            grid_x = x // clustering_tolerance
            grid_y = y // clustering_tolerance
            grid_key = (grid_x, grid_y)
            if grid_key not in clusters:
                clusters[grid_key] = []
            clusters[grid_key].append((x, y))

        new_persistent_objects = {}
        for cluster_points in clusters.values():
            x_coords = [p[0] for p in cluster_points]
            y_coords = [p[1] for p in cluster_points]
            x_center = int(np.mean(x_coords))
            y_center = int(np.mean(y_coords))
            
            matched = False
            for obj_id, obj_data in persistent_objects.items():
                prev_x, prev_y = obj_data['pos']
                distance = np.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
                
                if distance <= movement_tolerance:
                    new_persistent_objects[obj_id] = {
                        'pos': (x_center, y_center),
                        'frames': obj_data['frames'] + 1
                    }
                    matched = True
                    break
            
            if not matched:
                new_persistent_objects[next_object_id] = {
                    'pos': (x_center, y_center),
                    'frames': 1
                }
                next_object_id += 1

        persistent_objects = new_persistent_objects

        clustered_boxes = []
        satellites_detected = False
        
        for obj_data in persistent_objects.values():
            if obj_data['frames'] >= persistence_threshold:
                x_center, y_center = obj_data['pos']
                x_min = x_center - box_size // 2
                x_max = x_center + box_size // 2
                y_min = y_center - box_size // 2
                y_max = y_center + box_size // 2
                clustered_boxes.append((x_min, y_min, x_max, y_max))
                satellites_detected = True

        if frame_count % draw_interval == 0:
            for (x_min, y_min, x_max, y_max) in clustered_boxes:
                cv2.rectangle(curr_frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)

        # Handle frame writing with buffer
        if satellites_detected and first_detection:
            # Write buffered frames on first detection
            for frame in frame_buffer:
                out.write(frame)
            first_detection = False
            include_next_frames = post_detection_frames

        if satellites_detected or include_next_frames > 0:
            out.write(curr_frame)
            include_next_frames = max(0, include_next_frames - 1)

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
    print(f"Finished processing capture: {output_path if not is_live else camera_name}")

def main():
    for video_file in os.listdir(input_folder):
        if not video_file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            continue

        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        cap = cv2.VideoCapture(input_path)
        process_capture(cap, output_path, is_live=False)
        cv2.destroyAllWindows()

        completed_path = os.path.join(completed_folder, video_file)
        os.rename(input_path, completed_path)
        print(f"Moved {video_file} to {completed_folder}")

    if os.path.isfile(camera_config_file):
        with open(camera_config_file, 'r') as f:
            cameras = json.load(f)

        for cam in cameras:
            name_key = [k for k in cam.keys() if "name" in k.lower()][0]
            address_key = [k for k in cam.keys() if "address" in k.lower()][0]
            protocol_key = [k for k in cam.keys() if "protocol" in k.lower()][0]

            camera_name = cam[name_key]
            camera_address = cam[address_key]
            camera_protocol = cam[protocol_key]

            username_key = [k for k in cam.keys() if "user" in k.lower()]
            password_key = [k for k in cam.keys() if "pass" in k.lower()]
            username = cam[username_key[0]] if username_key else ""
            password = cam[password_key[0]] if password_key else ""

            if username and password:
                stream_url = f"{camera_protocol.lower()}://{username}:{password}@{camera_address}"
            else:
                stream_url = f"{camera_protocol.lower()}://{camera_address}"

            cap = cv2.VideoCapture(stream_url)
            process_capture(cap, output_path="", is_live=True, camera_name=camera_name)
            cv2.destroyAllWindows()
    else:
        print(f"No camera configuration file found at '{camera_config_file}'. Skipping live streams.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()