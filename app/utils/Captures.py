import datetime
import cv2 
from ..Settings import CAPTURES_DIR_PATH, BASE_YOLO_CONFIDENCE_THRESHOLD
import time
import os
from .Annotations import Annotations


class Captures(object):


    def __init__(self, annotations : Annotations, speed_limit = 0, deregistration_time=12):
        self.annotations = annotations
        self.speed_limit = speed_limit
        self.captured_offenders = {}
        self.deregistration_time = deregistration_time


    def capture_offense(self, detection, frame):

        detection['offender'] = True

        captured_at = datetime.datetime.now().strftime('%a-%b-%Y_%I-%M-%S%p')

        filename = os.path.join(CAPTURES_DIR_PATH, f'{captured_at}.jpg')

        cropped_frame = self.annotations.capture_traffic_violation(frame, detection, captured_at)
       
        try:
            cv2.imwrite(filename, cropped_frame)
        except Exception as e:
            print(f'Error occurded writing out capture to application directory! \n{e}')
    

    def compare_speed(self, detections, frame):

        ''' '''

        detected_at = time.time()
        already_captured = False

        for detection in detections:

            ID = detection.get('ID')
            speed = detection.get('speed')
            confidence_score = detection.get('confidence_score')

            if  speed is not None and \
                speed > self.speed_limit and \
                confidence_score > BASE_YOLO_CONFIDENCE_THRESHOLD and \
                not already_captured:

                if ID not in self.captured_offenders:
                    self.captured_offenders[ID] = { 'last_detected' : detected_at, 'already_captured' : already_captured }

                self.captured_offenders[ID]['last_detected'] = detected_at

                if not self.captured_offenders[ID]['already_captured']:
                    self.capture_offense(detection, frame)
                    self.captured_offenders[ID]['already_captured'] = True
                    already_captured = True

        self.prune_outdated_objects(detected_at)

        # Return updated detections.
        return detections
    

    def prune_outdated_objects(self, updated_at):

        '''
            Iterate over parameterised detections and prune those exceeding the set time limit threshold.

            Parameters:
                * parsed_detections : list[dict] -> list of detection data entries.
            Returns:
                * None. 
        '''

        # Initialise list to store ID values of detections to be pruned. 
        stale_detections = [ID for ID, detection in self.captured_offenders.items()
                            if (updated_at - detection['last_detected']) > self.deregistration_time]

        # Iterate over the IDs present. 
        for ID in stale_detections:
            # Use IDs to delete entries from tracked objects. 
            del self.captured_offenders[ID]

        