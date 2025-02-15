from ultralytics import YOLO
from src.psms_lib import psms
import cv2

def main():
    yolo_model = YOLO('src/model/yolo11n.pt')
    class_names = yolo_model.names
    zones, number_of_zones, frame_width, frame_height = psms.extract_data_from_file("src/data_file/data.txt")
    cap = psms.load_camera("inference/demo.mp4")
    success = True
    frame_name = "PSMS"
    wait_key = 1
    ord_key = 'q'


    while success:
        ret, frame = cap.read()

        if not ret:
            break

        zones_list = psms.init_zone_list(number_of_zones)
        frame = cv2.resize(frame, (frame_width, frame_height))
        boxes = psms.get_prediction_boxes(frame, yolo_model, 0.15)
        psms.draw_polylines_zones(frame, zones) # Optional
        frame, zones_list = psms.track_objects_in_zones(frame, boxes, zones, zones_list, class_names)

        psms.display_zone_info(frame, number_of_zones, zones_list) # Optional
        success = psms.show_frame(frame, frame_name, wait_key, ord_key) # Optional

    cap.release()
    cv2.destroyAllWindows()
    



if __name__ == "__main__":
    main()
