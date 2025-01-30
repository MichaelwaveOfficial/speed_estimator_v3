import numpy as np
from .Settings import *

from .utils.ObjectDetection import ObjectDetection
from .utils.ObjectTracking import ObjectTracking
from .utils.SpeedEstimation import SpeedEstimation
from .utils.BboxUtils import annotate_bbox_corners


object_detection = ObjectDetection(model=DETECTION_MODEL_PATH, confidence_threshold=0.8)
object_tracking = ObjectTracking()
speed_estimation = SpeedEstimation()

# Inform users whether hardware acceleration is being used or not. 
object_detection.check_for_hardware_acceleration()


def process_video(frame : np.ndarray, speed_limit : int = None, frame_rate : int = 30, vision_type : str = 'object_detection') -> np.ndarray:
    
    '''

        Paramaters:

            * 

        Returns:

            * 
    '''

    # Initialise detections list. 
    detections = []
    
    ''' Object Detection. '''

    # Obtain detections data by running inference leveraging YOLOV11 model on input media. 
    detections : list[dict] = object_detection.run_inference(frame=frame) 

    ''' Object Tracking '''

    # Assign IDs to detections and update their center point values.
    tracked_detections : list[dict] = object_tracking.update_tracker(detections=detections)

    ''' Speed Estimation. '''

    # Estimate a detections speed by comparing current and previous center points. 
    speed_estimation_detections : list[dict] = speed_estimation.apply_estimations(
        detections=tracked_detections,
        speed_limit=speed_limit,
        frame_rate=frame_rate
    )

    ''' Frame Annotation. '''
    
    # Supply the final step of processed data to be annotated for traffic insights. 
    annotated_frame = annotate_bbox_corners(
        frame=frame,
        detections=speed_estimation_detections,
        vision_type=vision_type
    )

    # Return frame wether modified or not. 
    return annotated_frame
