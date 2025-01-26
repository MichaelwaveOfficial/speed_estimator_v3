from ..Settings import *
import numpy as np 
from ultralytics import YOLO 
import torch 


class ObjectDetection(object):

    '''
        Class to handle obiject detection leveragng ultralytics YOLO models. 
    '''

    def __init__(self, model : str):

        # Select device for processing in attempt for hardware accelaration.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load specific model from constructor. 
        self.detection_model = YOLO(model)

        #  Most likely classnames for traffic management and their average sizes in mm found in the UK. 
        self.classes_of_interest = {
            'car' : {
                'width' : 1.821, 'height' : 1.534
            },
            'motorcycle' : {
                'width' : 0.995, 'height' : 2.190
            },
            'bus' : {
                'width' :  2.560, 'height' : 4.200
            },
            'truck' : {
                'width' :  2.400, 'height' : 2.590
            }
        } 

        # Confidence threshold for a detection to be considered relevant.
        self.confidence_threshold = 0.70

    
    def run_inference(self, frame : np.ndarray) -> dict:

        '''
            Parameters:

            * frame : np.ndarray -> input image for the detection model to run inference on.
            
            Returns:
            
            * filtrated_detections : dict -> dictionary containing detection metadata to be processed. 

            Function to run desired model inference on the provided frame input to return the detections data to later be 
            passed onto annoatations later on in the pipeline.
        '''

        # Initialise empty list. 
        filtrated_detections = []

        # detections from a given frame formatted into a structured output. 
        detections = self.detection_model(frame, verbose=False, device=self.device)[0]

        # Iterate over each detection within a given inferred run. 
        for detection in detections.boxes.data.tolist():
            
            # Unpack values, retrieve class ID.
            x1, y1, x2, y2, confidence_score, class_ID = detection
            
            # Fetch classname from model stored list of names leveraging class ID as the index. 
            classname = self.fetch_class_name(class_ID=class_ID)

            # If the classname is of interest (specified in the list).
            if classname in self.classes_of_interest and \
                confidence_score >= self.confidence_threshold:
                
                # Construct new detection substituting ID integer with string classname for better legibility. 
                detection_dict = {
                    'x1' : x1,
                    'y1' : y1,
                    'x2' : x2,
                    'y2' : y2,
                    'classname' : classname,
                    'avg_class_dimensions' : self.classes_of_interest[classname],
                    'confidence_score' : confidence_score
                }
                
                # Append modified detection to the filtrations list. 
                filtrated_detections.append(detection_dict)
            
        # Return filtrated data.
        return filtrated_detections
    

    def fetch_class_name(self, class_ID : int) -> str:

        '''
            Method to fetch the string equivalent of the detections classname from the given class ID produced
            by the model.

            Parameters:

                * class_ID : int -> Integer value representing the classification value.

            Returns:

                * classname : str -> string value for better user legibility. 
        '''

        return str(self.detection_model.names.get(class_ID, 'Unknown'))
    

    def check_for_hardware_acceleration(self) -> str:

        '''
            Simple function to inform users if hardware acceleration is being utlised or not. 
        '''

        if str(self.device) == 'cuda':
            print(f'Hardware acceleration initialised with {self.device}')
        else:
            print(f'Hardware acceleration failed, {self.device} initialised.')
    