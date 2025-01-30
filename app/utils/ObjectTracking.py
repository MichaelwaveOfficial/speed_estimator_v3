from .BboxUtils import calculate_center_point, measure_euclidean_distance
from time import perf_counter 
import numpy as np


class ObjectTracking(object):

    '''
        Module to parse detection data, track them by assigning IDs and pruning them when no longer required. 
    '''

    def __init__(self, euclidean_distance_threshold : float = 65, deregistration_time : int = 5, max_center_point_history_length : int = 25):

        # Dictionary to store a detections meta data. 
        self.tracked_objects : dict = {}

        # Increment counter to assign ID values to detections. 
        self.ID_increment_counter : int = 0

        # Maximum threshold before a detection is registered as a separate entity.
        self.euclidean_distance_threshold = euclidean_distance_threshold

        # Time in seconds before a detection is deregistered. 
        self.deregistration_time = deregistration_time

        # Maximum length threshold of center point rolling window size. 
        self.max_center_point_history_length = max_center_point_history_length


    def update_tracker(self, detections : list[dict]) -> list[dict]:

        '''
            Parse detections by updating their values with their velocity, assign ID values and prune those no 
                longer required to help resource management. 

            Parameters:
                * detections : list[dict] -> list containting detection dictionaries.

            Returns:
                * parsed_detections : list[dict] -> list containing the modified detection dictionaries. 
        '''

        # Ensure detections data being input is of correct type. 
        if not isinstance(detections, list):
                raise ValueError('Detections input must be a list of dictionaries.')
        
        # Initialise list to store and return the parsed detections dictionaries. 
        parsed_detections = []

        # Get current, intial time. 
        current_time = perf_counter()

        # Iterate over the current detections being parsed. 
        for current_detection in detections: 
             
            matched_detections_ID = self.match_detection_center_points(current_detection=current_detection)

            if matched_detections_ID:
                 
                self.update_tracked_objects(
                    matched_detections_ID,
                    current_detection,
                    current_time
                )

                parsed_detections.append(current_detection)
            
            else:
                 
                self.register_detection(
                    current_detection,
                    current_time
                )

                self.ID_increment_counter += 1

        # Iterate over the parsed detections list, prune the outdates detections surpassing the threshold.
        self.prune_outdated_detections(parsed_detections=parsed_detections)

        # Apply velocity estimations to detection values. 
        self.apply_velocity_estimations()

        # Return list of parsed detections. 
        return parsed_detections


    def register_detection(self, current_detection, current_time):
        
        center_point = calculate_center_point(current_detection)

        # Update tracked objects dictionary. 
        self.tracked_objects[self.ID_increment_counter] = {
            'center_point' : calculate_center_point(current_detection),
            'first_seen' : current_time,
            'last_seen' : current_time,
            'first_center' : center_point,
            'prev_center' : center_point,
            'velocity' : (0,0),
            'classname' : current_detection['classname'],
            'center_points_history' : [center_point]
        }

    
    def match_detection_center_points(self, current_detection):

        # Calculate current detections center point. 
        current_center_point = calculate_center_point(detection=current_detection)

        for ID, previous_detection in self.tracked_objects.items():
            
            # Calculate that previous detections center point. 
            previous_center_point = previous_detection['center_point']

            # Smooth current and previous center points data to provide greater consistency. 
            previous_smoothed_center_point = self.smooth_center_points( previous_center_point, current_center_point)
            current_smoothed_center_point = self.smooth_center_points(previous_smoothed_center_point,current_center_point)

            # Calculate the straight line distance between previous and current center points. 
            euclidean_distance_squared = measure_euclidean_distance(current_smoothed_center_point, previous_smoothed_center_point)

            # If euclidean distance within the threshold and the current classname matches the previous. 
            if current_detection['classname'] == previous_detection['classname'] and \
                euclidean_distance_squared <= (self.euclidean_distance_threshold **2):

                return ID
            
        return None 
    

    def update_tracked_objects(self, ID, current_detection, current_time):


        # Calculate current detections center point. 
        center_point = calculate_center_point(detection=current_detection)

        prev_time = self.tracked_objects[ID]['last_seen']
        first_seen = self.tracked_objects[ID]['first_seen']
        first_center = self.tracked_objects[ID]['first_center']
        prev_center = self.tracked_objects[ID]['prev_center']
        points_history = self.tracked_objects[ID]['center_points_history']

        self.tracked_objects[ID]['center_points_history'].append(center_point)

        if len(self.tracked_objects[ID]['center_points_history']) > self.max_center_point_history_length:
            self.tracked_objects[ID]['center_points_history'] = self.tracked_objects[ID]['center_points_history'][-self.max_center_point_history_length:]        

        # Estimate the current detections velocity path. 
        velocity = self.estimate_velocity(center_point,ID)
        
        # Update the tracked objects dictionary with fresh data.
        self.tracked_objects[ID].update({
            'center_point' : center_point,
            'last_seen' : current_time,
            'velocity' : velocity,
            'prev_center' : center_point,
        })

        # Update the current detection dictionary being parsed with the fresh, updated values. 
        current_detection.update({
            'ID' : ID,
            'velocity' : velocity,
            'first_timestamp' : first_seen,
            'previous_timestamp' : prev_time,
            'current_timestamp' : current_time,
            'first_center_point' : first_center,
            'previous_center_point' : prev_center,
            'current_center_point' : center_point,
            'center_points_history' : points_history
        })
    

    def prune_outdated_detections(self, parsed_detections : list[dict]) -> None:
        
        '''
            Iterate over parameterised detections and prune those exceeding the set time limit threshold.

            Parameters:
                * parsed_detections : list[dict] -> list of detection data entries.

            Returns:
                * None. 
        '''

        # Initialise list to store ID values of detections to be pruned. 
        stale_detections = []

        # Get current timestamp. 
        current_time = perf_counter()

        # Iterate over tracked objects dictionary. 
        for ID, detections in self.tracked_objects.items():
            
            # Calculate the elapsed time from the detections last timestamp. 
            elapsed_time = current_time - detections['last_seen']

            # If ID not present in the detections list and elapsed time  greater than deregistration threshold. 
            if ID not in parsed_detections and \
                elapsed_time > self.deregistration_time:

                # Append ID value to stale detections list. 
                stale_detections.append(ID)

        # Iterate over the IDs present. 
        for ID in stale_detections:

            # Use IDs to delete entries from tracked objects. 
            del self.tracked_objects[ID]

    
    def estimate_velocity(self, current_center_point : tuple, ID : int, rolling_window : int = 10) -> tuple:
        
        '''
            Function to estimate a detections velocity by subtracting its previous positions on its x and y axis from its current. 

            Parameters:

                * current_center_point : tuple -> (x,y) detections current center point.
                * previous_position : tuple -> (x,y) detection center point prior to its current. 
                * ID : int -> Detections unique identifier. 

            Returns:
                * estimated_velocity : tuple -> tuple containing the distance covered from its previous center point to its current point on 
                    an x/y axis. 
        
        '''

        # Fetch a detections previous center point. 
        previous_center_point = self.tracked_objects[ID].get('prev_center', current_center_point)
        
        # Initialise a detections velocity history if not yet present.
        if 'velocity_history' not in self.tracked_objects[ID]:
            self.tracked_objects[ID]['velocity_history'] = []


        # Calculate a detections change in position. 
        delta_x, delta_y = current_center_point[0] - previous_center_point[0], current_center_point[1] - previous_center_point[1]

        # Append calculated velocity to that given objects history. 
        self.tracked_objects[ID]['velocity_history'].append((delta_x, delta_y))
        
        # Maintain a rolling window of last five velocity values. 
        if len(self.tracked_objects[ID]['velocity_history']) > rolling_window:
            # Remove oldest entry. 
            self.tracked_objects[ID]['velocity_history'].pop(0)

        # Calculate average velocity from the past five frames.
        avg_x, avg_y = np.mean(self.tracked_objects[ID]['velocity_history'], axis=0)

        # Return smoothed velocity. 
        return avg_x, avg_y

    
    def apply_velocity_estimations(self) -> None:

        '''
            Functions to apply velocity estimations to a current detections center points to attempt to smooth out occlusions or low latency. 
        '''

        # Fetch current timestamp.
        current_time = perf_counter()

        # Iterate over detections within tracked_objects dictionary. 
        for ID, detection in self.tracked_objects.items():
            
            # Calucate elapsed time with detections last seen timestamp.
            elapsed_time = current_time - detection['last_seen']

            # If elapsed time less than deregistration threshold.
            if elapsed_time <= self.deregistration_time:
                
                # Fetch velocity values from detection.
                velocity = detection['velocity']

                # Apply velocity values to center point values on x and y axis. 
                detection['center_point'] = (
                    detection['center_point'][0] + velocity[0] * elapsed_time,
                    detection['center_point'][1] + velocity[1] * elapsed_time
                )
    

    def smooth_center_points(self, previous_center : float, current_center : float, alpha : float = 0.5) -> tuple[float, float]:

        '''
            Help provide more consistent center point data by leveraging current and previous center points along with a given 
            interval the alpha.

            Parameters:
                * previous_center : float -> (x,y) the previous center point.
                * current_center : float -> (x,y) the current center point.
                * alpha : float -> weight that determines the importance of the original center point. 
                    - alpha = 0: Use only the previous center point.
                    - alpha = 1: Use only the current center point.
                    - 0 < alpha < 1: Blend the two points proportionally.

            Returns:
                * Smoothed center point utilising the weighted average for more accurate calculations. 
        '''

        return (
            alpha * current_center[0] + (1 - alpha) * previous_center[0],
            alpha * current_center[1] + (1 - alpha) * previous_center[1]
        )
