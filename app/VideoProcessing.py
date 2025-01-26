import numpy as np
from .Settings import *

from .utils.ObjectDetection import ObjectDetection
from .utils.ObjectTracking import ObjectTracking
from .utils.SpeedEstimation import SpeedEstimation


object_detection = ObjectDetection(DETECTION_MODEL_PATH)
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
    detections : list[dict] = object_tracking.update_tracker(detections=detections)

    ''' Speed Estimation. '''

    # Estimate a detections speed by comparing current and previous center points. 
    detections : list[dict] = speed_estimation.apply_estimations(
        detections=detections,
        frame_rate=frame_rate,
        speed_limit=speed_limit
    )

    ''' Frame Annotation. '''

    # Annotate accordingly depending on vision type.
    match vision_type:

        case 'object_detection':

            print(f'{vision_type} selected.')
        
        case 'object_tracking':

            print(f'{vision_type} selected.')
        
        case 'speed_estimation':

            print(f'{vision_type} selected.')

        case _:

            print('No vision mode selected.')

    # Return frame wether modified or not. 
    return frame
