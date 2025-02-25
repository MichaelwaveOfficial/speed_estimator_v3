from .BboxUtils import calculate_center_point, measure_euclidean_distance
from time import time 

class ObjectTracking(object):

    ''' Module to parse detection data, track them by assigning IDs and pruning them when no longer required. '''

    def __init__(self, euclidean_distance_threshold : float = 10, deregistration_time : int = 10, frame_rate : int = 30):

        '''
        
        '''

        self.tracked_objects = {}

        self.ID_increment_counter = 0

        self.euclidean_distance_threshold = euclidean_distance_threshold

        self.deregistration_time = deregistration_time

        self.frame_rate = frame_rate

    
    def update_tracker(self, detections):

        ''' '''

        if detections is None or not isinstance(detections, list):
            raise ValueError('Detections being parsed not a list of dictionaries.')

        updated_at = time()

        parsed_detections = []

        for current_detection in detections:

            current_center_point = calculate_center_point(current_detection)
           
            matched_ID = self.match_detection_center_points(current_detection, current_center_point)

            if matched_ID is not None:
                self.update_object(matched_ID, current_detection, updated_at, current_center_point)
            else:
                self.register_object(current_detection, updated_at, current_center_point)

            parsed_detections.append(current_detection)

        self.prune_outdated_objects(updated_at)

        return parsed_detections
    

    def match_detection_center_points(self, detection, current_center_point):

        closest_ID = None 
        shortest_distance = float('inf')

        for ID, prev_detection in self.tracked_objects.items():

            previous_center = prev_detection['center_points'][-1]

            euclidean_distance_squared = measure_euclidean_distance(current_center_point, previous_center)

            # Accumulate the detections real world dimensions.
            avg_dimensions_width = detection['avg_class_dimensions']['width']
            # Get the detections bbox width and height.
            detection_width = abs(detection['x2'] - detection['x1'])

            scaled_euclidean_distance_threshold = self.euclidean_distance_threshold * (detection_width / avg_dimensions_width)

            if euclidean_distance_squared <= scaled_euclidean_distance_threshold and \
                euclidean_distance_squared < shortest_distance:

                closest_ID = ID 
                shortest_distance = euclidean_distance_squared

        return closest_ID
    

    def register_object(self, detection, seen_at, current_center_point):

        classname = detection.get('classname', 'Unknown')

        self.tracked_objects[self.ID_increment_counter] = {
            'center_points' : [current_center_point],
            'first_detected' : seen_at,
            'last_detected' : seen_at,
            'classname' : classname
        }

        detection['ID'] = self.ID_increment_counter
        self.ID_increment_counter += 1
    

    def update_object(self, ID, detection, updated_at, current_center_point, center_points_window = 100):

        self.tracked_objects[ID]['center_points'].append(current_center_point)
        self.tracked_objects[ID]['last_detected'] = updated_at

        # Maintain a rolling window of last five velocity values. 
        if len(self.tracked_objects[ID]['center_points']) > center_points_window:
            self.tracked_objects[ID]['center_points'].pop(0)

        detection.update(self.tracked_objects[ID])

        detection['ID'] = ID
         
    
    def prune_outdated_objects(self, updated_at):

        '''
            Iterate over parameterised detections and prune those exceeding the set time limit threshold.

            Parameters:
                * parsed_detections : list[dict] -> list of detection data entries.
            Returns:
                * None. 
        '''

        # Initialise list to store ID values of detections to be pruned. 
        stale_detections = [ID for ID, detection in self.tracked_objects.items()
                            if (updated_at - detection['last_detected']) > self.deregistration_time]

        # Iterate over the IDs present. 
        for ID in stale_detections:
            # Use IDs to delete entries from tracked objects. 
            del self.tracked_objects[ID]
