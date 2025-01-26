import cv2 
import numpy as np


'''

    This module needs work, thought going into how to separate concerns with labels to make things more manageable. 
'''


def annotate_bbox(
        frame : np.ndarray,
        detection : dict, 
        colour=(10, 250, 10),
        thickness=3,
        length=25,
        ) -> np.ndarray:
    
    '''
        Parse the given detection dictionary and annotate the corners of its bounding box. 

        Parameters: 

            * frame : np.ndarray -> Frame to be annotated upon.
            * detection : dictionary -> Dictionary containing a detections data. 
            * colour : tuple -> Base bounding box colour. 
            * thickness : int -> Base thickness for the annotation lines. 
            * length : int -> Base length for the annotation lines. 

        Returns:

            * frame : np.ndarray -> Annotated frame with a detections given bounding box. 
    '''
   
    x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])

    # Top left.
    cv2.line(frame, (x1, y1), ( x1, y1 + length ) , colour, thickness)
    cv2.line(frame, (x1, y1), ( x1 + length, y1 ) , colour, thickness)
    
    # Bottom left. 
    cv2.line(frame, (x1, y2), ( x1, y2 - length ) , colour, thickness)
    cv2.line(frame, (x1, y2), ( x1 + length, y2 ) , colour, thickness)
    
    # Top right. 
    cv2.line(frame, (x2, y1), ( x2 - length, y1 ) , colour, thickness)
    cv2.line(frame, (x2, y1), ( x2, y1 + length ) , colour, thickness)
    
    # Bottom right. 
    cv2.line(frame, (x2, y2), ( x2, y2 - length ) , colour, thickness)
    cv2.line(frame, (x2, y2), ( x2 - length, y2 ) , colour, thickness)

    return frame


def annotate_detection_data(
        frame : np.ndarray,
        detection : dict,
        font = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale = 1,
        font_color = (0, 255, 0), 
        font_thickness = 3,
    ) -> np.ndarray:
    return


def calculate_center_point(detection):

    '''
        Simple function to calculate the center point of a given detection. This can be useful for both 
        annotation and tracking purposes. 

        Parameters:

            * bbox : dict -> the detections metadata to calculate the center point. 

        Returns: 

            * center_x, center_y : tuple -> two floating point values representing the detections center 
                on the x and y axis. 

    
    '''

    x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])

    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

    return int(center_x), int(center_y)


def measure_euclidean_distance(p1, p2):

    '''
        Function to measure the straight-line distance between two points. Gets the sqaure values of the inputs and sqaures the output
            to help reduce computation. 

        Paramaters:

            * p1 : tuple -> (x1, y1), detection start position.
            * p2 : tuple -> (x2,y2), detection end position.

        Returns:

            * euclidean_distance : float -> distance between one position and another. 
    '''

    return (( p1[0] - p2[0] ) **2 + ( p1[1] - p2[1] ) **2) **0.5
