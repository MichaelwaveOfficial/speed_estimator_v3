import numpy as np
from .Settings import *

from .utils.ObjectDetection import ObjectDetection
from .utils.ObjectTracking import ObjectTracking
from .utils.SpeedEstimation import SpeedEstimation
from .utils.Captures import Captures
from .utils.Annotations import Annotations

# Intstantiate objects to mimic singleton pipeline.
annotations = Annotations()
object_detection = ObjectDetection(model=DETECTION_MODEL_PATH, confidence_threshold=BASE_YOLO_CONFIDENCE_THRESHOLD)
object_tracking = ObjectTracking()
speed_estimation = SpeedEstimation()
captures = Captures(annotations=annotations) # Dependency injection to access methods.

# Inform users whether hardware acceleration is being used or not. 
object_detection.check_for_hardware_acceleration()


def process_video(frame : np.ndarray, speed_limit : int = 0, frame_rate : int = 30, vision_type : str = 'object_detection', confidence_threshold :float = BASE_YOLO_CONFIDENCE_THRESHOLD) -> np.ndarray:
    
    '''
        Paramaters:
            * 

        Returns:
            * 
    '''

    # Update framerate variables once function is called from media being parsed to improve measurements accuracy.
    object_detection.confidence_threshold = confidence_threshold
    object_tracking.frame_rate = frame_rate
    speed_estimation.frame_rate = frame_rate
    captures.speed_limit = speed_limit


    try:

        ''' Object Detection. '''

        # Obtain detections data by running inference leveraging YOLOV11 model on input media. 
        detections : list[dict] = object_detection.run_inference(frame=frame) 

        ''' Object Tracking '''

        # Assign IDs to detections and update their center point values.
        tracked_detections : list[dict] = object_tracking.update_tracker(detections=detections)

        ''' Speed Estimation. '''

        # Estimate a detections speed by comparing current and previous center points. 
        speed_estimation_detections : list[dict] = speed_estimation.apply_estimations(detections=tracked_detections)

        ''' Violation Checks. '''

        captured_detections = captures.compare_speed(detections=speed_estimation_detections, frame=frame)

        ''' Frame Annotation. '''

        # Supply the final step of processed data to be annotated for traffic insights. 
        annotated_frame = annotations.annotate_frame(frame=frame, detections=captured_detections, vision_type=vision_type)
    
        # Return frame wether modified or not. 
        return annotated_frame

    except Exception as e:
        print(f'Error within the video processing pipeline.\n{e}.')
        return frame
