import cv2 
import numpy as np


'''

    This module needs work, thought going into how to separate concerns with labels to make things more manageable. 

    Clean up corners with elipses.

        * Detections classes have different colours.

        * Tracking has cp lines from first and current.

        * Speed estimation has speed

        * All recieve IDs + bboxes?

        * Annotations then call it quits.

'''


def annotate_bbox_corners(
        frame : np.ndarray,
        detections : list[dict],
        vision_type : str = 'object_detection', 
        colour=(10, 250, 10),
        size_factor=0.1,
        thickness_factor=0.01
    ) -> np.ndarray:
    
    '''
        Dynamically annotate a given detection adjusting the size and border radius of the annotated bounding box in relativity to 
            the detections size. 

        Parameters: 
            * frame : np.ndarray -> frame to be drawn upon.
            * detection : dict -> detection dictionary containing desired values to plot data points.
            * colour : tuple -> BGR values to determine annotation colour. 
            * size_factor : float -> Scaling factor relative to detection size. 
            * thickness_factor : float -> Line thickness factor relative to detection size. 
           
        Returns:
            * annotated_frame : np.ndarray -> Annotated frame with a detections given bounding box. 
    '''

    for detection in detections:

        # Initalise minimum and maximum contraints.
        min_corner_radius, max_corner_radius = 5, 30
        min_thickness, max_thickness = 1, 5

        # Fetch detection bounding box values, typecast to full integer values. 
        x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
        offender = bool(detection.get('offender', False))

        if offender:
            colour = (10, 10, 255)
        else:
            colour = (10, 255, 10)

        # Calculate detection dimensions.
        detection_width = x2 - x1
        detection_height = y2 - y1
        detection_size = min(detection_width, detection_height)

        # Dynamically calculate a detections line thickness and corner radius for annotation.
        detection_corner_radius = max(min(int(detection_size * size_factor), max_corner_radius), min_corner_radius)
        detection_thickness = max(min(int(detection_size * thickness_factor) * 2, max_thickness), min_thickness)

        ''' Bounding Box Corners. '''

        # Top left arc.
        cv2.ellipse(
            frame,
            (x1 + detection_corner_radius, y1 + detection_corner_radius),
            (detection_corner_radius, detection_corner_radius),
            0, 180, 270,
            colour,
            detection_thickness
        )
        
        # Bottom left arc.
        cv2.ellipse(
            frame,
            (x1 + detection_corner_radius, y2 - detection_corner_radius),
            (detection_corner_radius, detection_corner_radius),
            0, 90, 180,
            colour,
            detection_thickness
        )

        #Top right arc.
        cv2.ellipse(
            frame,
            (x2 - detection_corner_radius, y1 + detection_corner_radius),
            (detection_corner_radius, detection_corner_radius),
            0, 270, 360,
            colour,
            detection_thickness
        )

        # Botton right arc.
        cv2.ellipse(
            frame,
            (x2 - detection_corner_radius, y2 - detection_corner_radius),
            (detection_corner_radius, detection_corner_radius),
            0, 0, 90,
            colour,
            detection_thickness
        )

        
        annotate_detection_data(frame=frame, detection=detection, vision_mode=vision_type)

    return frame


def annotate_detection_data(
        frame : np.ndarray,
        detection : dict,
        font = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale = 1.25,
        font_color = (0, 255, 0),
        bg_colour = (0, 0, 0),
        border_radius=6,
        padding=10,
        font_thickness = 3,
        vision_mode : str = 'object_detection'
    ) -> np.ndarray:

    x1, y1 = int(detection['x1']), int(detection['y1'])
    ID, classname, confidence_score = str(detection.get('ID')), str(detection.get('classname')), str(round(detection.get('confidence_score'), 2))
    speed = str(detection.get('speed', 0))

    label = f"ID : {ID} classname: {classname} confidence score: {confidence_score}"

    if vision_mode == 'object_detection':
        label = label

    if vision_mode == 'object_tracking':
        label = f"ID : {ID}"

        annotate_center_point_trail(frame, detection)

 
    if vision_mode == 'speed_estimation':

        label = f"ID: {ID} Speed: {speed}mph"

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding

    # Draw the rounded rectangle (background)
    cv2.rectangle(
        frame,
        (x1 + border_radius, y1),  # Top-left inner corner
        (x1 + box_width - border_radius, y1 + box_height),  # Bottom-right inner corner
        bg_colour,
        -1  # Fill the rectangle
    )
    cv2.rectangle(
        frame,
        (x1, y1 + border_radius),
        (x1 + box_width, y1 + box_height - border_radius),
        bg_colour,
        -1
    )
    # Draw the rounded corners
    cv2.ellipse(
        frame,
        (x1 + border_radius, y1 + border_radius),  # Top-left corner center
        (border_radius, border_radius),
        180, 0, 90,
        bg_colour,
        -1
    )
    cv2.ellipse(
        frame,
        (x1 + box_width - border_radius, y1 + border_radius),  # Top-right corner center
        (border_radius, border_radius),
        270, 0, 90,
        bg_colour,
        -1
    )
    cv2.ellipse(
        frame,
        (x1 + border_radius, y1 + box_height - border_radius),  # Bottom-left corner center
        (border_radius, border_radius),
        90, 0, 90,
        bg_colour,
        -1
    )
    cv2.ellipse(
        frame,
        (x1 + box_width - border_radius, y1 + box_height - border_radius),  # Bottom-right corner center
        (border_radius, border_radius),
        0, 0, 90,
        bg_colour,
        -1
    )

    # Add the text on top of the background
    text_position = (x1 + padding, y1 + box_height - padding)

    annotated_frame = cv2.putText(
        frame,
        label,
        text_position,
        font,
        font_scale,
        font_color,
        font_thickness
    )

    return annotated_frame


def annotate_center_point(
        frame,
        center_point,
        colour=(0, 0, 255),
        radius=5,
        thickness=-1,
    ) -> np.ndarray:

    center_x, center_y = center_point

    cv2.circle(
        frame,
        (center_x, center_y),
        radius,
        colour,
        thickness
    )


def annotate_center_point_trail(
        frame,
        detection,
        colour=(180, 50, 50),
        thickness=8
    ):

    if 'center_points' not in detection:
        return frame
    
    center_points_list = detection['center_points']
    points_list_length = len(center_points_list)

    initial_center_point = center_points_list[0]
    final_center_point = center_points_list[-1]

    annotate_center_point(frame, initial_center_point, (10, 10, 255))

    for x in range(1, points_list_length):
        cv2.line(
            frame,
            center_points_list[x - 1],
            center_points_list[x],
            colour,
            thickness
        )

    annotate_center_point(frame, final_center_point, (10, 10, 255))

    return frame
    
    

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

    return (p1[0] - p2[0]) **2 + (p1[1] - p2[1]) **2
