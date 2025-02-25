from .BboxUtils import measure_euclidean_distance
from time import time
import numpy as np
from ..Settings import *


class SpeedEstimation(object):

    def __init__(self, frame_rate : int = 30, deregistration_time : int = 12):

        self.frame_rate = frame_rate
        self.deregistration_time = deregistration_time
        self.detection_speeds = {}

    
    def apply_estimations(self, detections):

        updated_at = time()

        for detection in detections:

            ID = detection.get('ID', None)

            if ID is None or \
                'center_points' not in detection or \
                    len(detection['center_points']) < 2:
                continue

            detection['ppm'] = self.calibrate_ppm(detection)

            prev_point, curr_point = detection['center_points'][-2:]

            frame_gap = 1.0 / self.frame_rate

            if frame_gap > 0:

                pixel_distance = measure_euclidean_distance(prev_point, curr_point)

                if ID not in self.detection_speeds:
                    self.detection_speeds[ID] = {'speeds' : [], 'last_updated': updated_at}

                self.detection_speeds[ID]['speeds'].append(self.calculate_speed(pixel_distance, detection.get('ppm', 1), frame_gap))

                detection['speed'] = round(np.median(self.detection_speeds[ID]['speeds']), 2)
                
        self.prune_outdated_objects(updated_at)

        return detections
    

    def unit_conversion(self, speed : float, measurement : str = 'mph') -> float:

        '''
            Simple helper function to convert speed measurement into a recognised unit of measurement determined by the user. 

            Parameters:
                * speed : float -> The detections calculated speed.
                * measurement : str -> The unit of measurement desired.

            Returns:
                * float -> speed multipled by the concered unit of measuremnt. 
        
        '''

        # Conversion values dictionary contianing key value pairs. 
        conversion_factors = {'mph': 2.23,  'kmh': 3.6}

        # If provided measurements not in conversions dictionary, let user know. 
        if measurement not in conversion_factors:
            raise ValueError(f"Unsupported measurement unit: {measurement}")

        # Return speed multiplied by specified conversion factor. 
        return speed * conversion_factors[measurement]
    

    def calibrate_ppm(self, detection : dict) -> float:

        '''
            Attempt to calibrate pixels per meter by obtaining the average real-world dimensions for a detection and 
                leveraging it against the detections bounding box width and height ro determine its scale.

            Parameters:
                * detection : dict -> single detection dictionary containing metadata. 

            Returns:
                * float -> pixels per meter values for that detection to later be used for speed estimation.
        '''

        # Accumulate the detections real world dimensions.
        real_width, real_height = detection['avg_class_dimensions']['width'], detection['avg_class_dimensions']['height']

        # Get the detections bbox width and height.
        detection_width = abs(detection['x2'] - detection['x1'])
        detection_height = abs(detection['y2'] - detection['y1'])

        # Calculate its ppm width and height. 
        ppm_width = detection_width / real_width
        ppm_height = detection_height / real_height

        # Return detections scale.
        return (ppm_width + ppm_height) / 2
    
    
    def calculate_speed(self, pixel_distance, ppm, elapsed_time):

        if elapsed_time <= 0 or ppm <= 0:
            return 0.0
        
        distance_meters = pixel_distance / ppm 

        speed_mps = distance_meters / elapsed_time

        return self.unit_conversion(speed=speed_mps, measurement='mph')
    

    def prune_outdated_objects(self, updated_at):

        '''
            Iterate over parameterised detections and prune those exceeding the set time limit threshold.

            Parameters:
                * parsed_detections : list[dict] -> list of detection data entries.
            Returns:
                * None. 
        '''

        # Initialise list to store ID values of detections to be pruned. 
        stale_detections = [ID for ID, detection in self.detection_speeds.items()
                            if (updated_at - detection['last_updated']) > self.deregistration_time]

        # Iterate over the IDs present. 
        for ID in stale_detections:
            # Use IDs to delete entries from tracked objects. 
            del self.detection_speeds[ID]
    