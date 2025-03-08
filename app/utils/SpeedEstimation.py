import numpy as np 
from collections import deque 
from time import time
from .BboxUtils import measure_euclidean_distance
from ..Settings import *


class SpeedEstimation(object):


    def __init__(
        self,
        frame_rate : int = 30, 
        deregistration_time : int = 12,
        rolling_window_size : int = 5,
        ppm_smoothing_factor : float = 0.7
    ):

        self.frame_rate = frame_rate
        self.deregistration_time = deregistration_time
        self.rolling_window_size = rolling_window_size
        self.ppm_smoothing_factor = ppm_smoothing_factor
        self.detection_speeds = {}

    
    def apply_estimations(self, detections):

        ''' '''

        updated_at = time()

        for detection in detections:

            if not self.validate_detection(detection):
                continue

            ID = detection['ID']
            current_center_point = detection['center_points'][-1]

            detection_ppm = self.calibrate_ppm(detection)

            if ID not in self.detection_speeds:

                self.initialise_detection_speeds(ID, current_center_point, detection_ppm, updated_at)
                continue

            previous_data = self.detection_speeds[ID]
            smoothed_detection_ppm = self.smooth_detection_ppm(previous_data['ppm'], detection_ppm)

            detection_speed = self.calculate_frame_speed(previous_data, current_center_point, smoothed_detection_ppm, updated_at)

            if detection_speed:

                self.update_detections_speed(ID, detection_speed, current_center_point, smoothed_detection_ppm, updated_at)

                detection['speed'] = round(float(np.median(self.detection_speeds[ID]['speeds'])), 2)

        self.prune_outdated_objects(updated_at)

        return detections
    

    def validate_detection(self, detection):

        ''' '''

        return (
            detection.get('ID') is not None and 
            'center_points' in detection and 
            len(detection['center_points']) >= 1
        )
    

    def initialise_detection_speeds(self, ID, center_point, ppm, updated_at):

        ''' '''

        self.detection_speeds[ID] = {
            'last_center' : center_point,
            'ppm' : ppm,
            'updated_at' : updated_at,
            'speeds' : deque(maxlen=self.rolling_window_size)
        }


    def smooth_detection_ppm(self, prev_ppm, curr_ppm):

        ''' '''

        return self.ppm_smoothing_factor * prev_ppm + (1 - self.ppm_smoothing_factor) * curr_ppm
    

    def calculate_frame_speed(self, prev_detection, current_center_point, current_ppm, updated_at):

        ''' '''

        prev_center = prev_detection['last_center']
        prev_ppm = prev_detection['ppm']
        prev_time = prev_detection['updated_at']

        pixel_distance = measure_euclidean_distance(prev_center, current_center_point)
        elapsed_time = updated_at - prev_time

        if elapsed_time <= 0 or pixel_distance < 2:
            return None 
        
        avg_ppm = prev_ppm + current_ppm / 2

        return self.calculate_speed(pixel_distance, avg_ppm, elapsed_time)
    

    def update_detections_speed(self, ID, speed, center_point, ppm, updated_at):

        ''' '''

        self.detection_speeds[ID]['speeds'].append(speed)

        self.detection_speeds[ID].update({
            'last_center' : center_point,
            'ppm' : ppm, 
            'updated_at' : updated_at
        })


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
        real_width, real_height = (
            detection['avg_class_dimensions']['width'],
            detection['avg_class_dimensions']['height']
        )

        # Get the detections bbox width and height.
        detection_width = max(abs(detection['x2'] - detection['x1']), 1)
        detection_height = max(abs(detection['y2'] - detection['y1']), 1)

        # Calculate its ppm width and height. 
        ppm_width = detection_width / real_width
        ppm_height = detection_height / real_height

        ppms = [ppm for ppm in [ppm_width, ppm_height] if ppm > 0]

        # Return detections scale.
        return sum(ppms) / len(ppms)
    
    
    def calculate_speed(self, pixel_distance, ppm, elapsed_time):

        ''' '''

        if elapsed_time <= 0 or ppm <= 0:
            return 0.0
        
        return self.unit_conversion(speed=(pixel_distance / ppm) / elapsed_time, measurement='mph')
    

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
                            if (updated_at - detection['updated_at']) > self.deregistration_time]

        # Iterate over the IDs present. 
        for ID in stale_detections:
            # Use IDs to delete entries from tracked objects. 
            del self.detection_speeds[ID]
    