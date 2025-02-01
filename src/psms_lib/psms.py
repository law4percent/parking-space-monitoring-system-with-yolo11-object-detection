import cv2
import numpy as np
import os
from ultralytics import YOLO
import re


def check_exist_file(file_path):
    """Check if a file exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")


def extract_data_from_file(file_path):
    check_exist_file(file_path)
    
    # Read the contents of the file
    with open(file_path, 'r') as file:
        text = file.read()

    # Initialize the variables
    zones = {}
    number_of_zones = 0
    frame_width = 0
    frame_height = 0

    # Regular expression to extract the zones and their points
    zones_pattern = re.compile(r"(\d+): \[(.*?)\]")
    matches = zones_pattern.findall(text)

    # Extracting zones data
    for match in matches:
        zone_id = int(match[0])  # Zone ID
        points_str = match[1]  # Points as string
        # Convert points to tuples of floats
        points = [
            tuple(map(float, point.strip('()').split(',')))
            for point in points_str.split('), (')
        ]
        zones[zone_id] = points  # Store the points in the dictionary with zone_id as key

    # Extracting the number of zones
    number_of_zone_pattern = re.compile(r"number_of_zone: (\d+)")
    number_of_zone_match = number_of_zone_pattern.search(text)
    if number_of_zone_match:
        number_of_zones = int(number_of_zone_match.group(1))

    # Extracting the frame dimensions
    frame_dimensions_pattern = re.compile(r"frame_width: (\d+)\nframe_height: (\d+)")
    frame_dimensions_match = frame_dimensions_pattern.search(text)
    if frame_dimensions_match:
        frame_width = int(frame_dimensions_match.group(1))
        frame_height = int(frame_dimensions_match.group(2))

    # Return the extracted data
    return zones, number_of_zones, frame_width, frame_height


def draw_polylines_zones(image, data, linesColor=(0, 255, 0), txtColor=(0, 0, 255), fontScale=0.65, thickness=2):
    """Draw polygons (zones) on an image."""
    for key, points in data.items():
        points_array = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points_array], isClosed=True, color=linesColor, thickness=2)
        centroid = np.mean(points_array, axis=0).astype(int)
        cv2.putText(image, f"{key}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, fontScale, txtColor, thickness)
    return image


def display_info(frame, number_of_zones, zones_list, color=(255, 255, 255), fontScale=0.75, thickness=2):
    """Display info for each zone on the frame."""
    for zone_indx in range(number_of_zones):
        cv2.putText(
            frame,
            f"zone: {zone_indx} | nv: {len(zones_list[zone_indx])}",
            (25, 25 + 28 * zone_indx),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            thickness,
        )
    return frame

def check_camera(cap):
    """Check if the camera is opened successfully."""
    if not cap.isOpened():
        raise TypeError("Cannot open camera.")

def track_objects_in_zones(frame, boxes, zones, zones_list, class_list):
    """Track detected objects in defined zones."""
    if frame is None:
        return [frame, zones_list]
    
    if len(zones) == 0:
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2, conf_score, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf_score = "%.2f" % conf_score
            cls_center_x = int(x1 + x2) // 2
            cls_center_y = int(y1 + y2) // 2
            cls_center_pnt = (cls_center_x, cls_center_y)
            frame = display_object_info(frame, x1, y1, x2, y2, cls, conf_score, class_list, cls_center_pnt)
        return [frame, zones_list]

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2, conf_score, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        conf_score = "%.2f" % conf_score
        cls_center_x = int(x1 + x2) // 2
        cls_center_y = int(y1 + y2) // 2
        cls_center_pnt = (cls_center_x, cls_center_y)
        for zone_indx, zone in enumerate(zones.values()):
            if cv2.pointPolygonTest(np.array(zone, dtype=np.int32), cls_center_pnt, False) == 1:
                frame = display_object_info(frame, x1, y1, x2, y2, cls, conf_score, class_list, cls_center_pnt)
                zones_list[zone_indx].append(class_list[int(cls)])
                break
    return [frame, zones_list]

def display_object_info(frame, x1, y1, x2, y2, cls, conf_score, class_list, cls_center_pnt):
    """Draw object information (bounding box, class, and confidence score) on the frame."""
    cv2.circle(img=frame, center=cls_center_pnt, radius=0, color=(255, 100, 255), thickness=5)
    cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(255, 100, 100), thickness=2)
    cv2.putText(img=frame, text=f"{class_list[int(cls)]} {conf_score}%", org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 255, 255), thickness=2)
    return frame

def init_zone_list(number_of_zones):
    """Initialize an empty list for each zone."""
    return [[] for _ in range(number_of_zones)]

def load_camera(video_source):
    """Load camera from video source."""
    captured = cv2.VideoCapture(video_source)
    check_camera(captured)
    return captured

def get_prediction_boxes(frame, yolo_model, confidence):
    """Get prediction boxes from the YOLO model."""
    pred = yolo_model.predict(source=[frame], save=False, conf=confidence)
    results = pred[0]
    boxes = results.boxes.data.numpy()
    return boxes

def show_frame(frame, frame_name, wait_key=1, ord_key='q'):
    """Display a frame using OpenCV."""
    cv2.imshow(frame_name, frame)
    if cv2.waitKey(wait_key) & 0xFF == ord(ord_key):
        return False
    return True

def display_zone_info(frame, number_of_zones, zones_list, color=(255, 255, 255), fontScale=0.75, thickness=2):
    
    count_occupied = 0
    for index in range(number_of_zones):
        if zones_list[index]:
            count_occupied += 1

    # for zone_indx in range(number_of_zones):
    cv2.putText(
                frame,
                f"vacant: {number_of_zones - count_occupied}",
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                thickness,
            )
    cv2.putText(
                frame,
                f"occupied: {count_occupied}",
                (25, 25 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                thickness,
            )