import cv2 
from .BboxUtils import calculate_center_point
import numpy as np


class Annotations(object):

    ''' '''

    def __init__(self):
        
        self.bbox_colours = {
            'standard' : (10, 255, 10),
            'offender' : (10, 10, 255),
            'trail' : (255, 10, 10)
        }

        self.min_corner_radius = 5
        self.max_corner_radius = 30
        self.min_thickness = 1
        self.max_thickness = 5
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 2
        self.font_colour = (255, 255, 255)
        self.font_thickness = 2
        self.bg_colour = (0,0,0)
        self.border_radius = 6
        self.label_padding = 25
        self.padding = 64
        self.center_point_radius = 5
        self.trail_colour = (255, 10, 10)
        self.trail_thickness = 12
        self.end_point_thickness = 24
        self.size_factor = 0.5
        self.thickness_factor = 0.01
        self.thickness = 8


    def annotate_frame(self, frame, detections : list[dict], vision_type : str):

        ''' '''

        for detection in detections:

            # Annotate vehicle corners.
            frame = self.annotate_bbox_corners(frame, detection)

            # Annotate metadata labels.
            frame = self.annotate_label(frame, detection, vision_type)

        return frame
    

    def fetch_bbox_colour(self, detection : dict) -> tuple[int, int, int]:

        '''
            Fetch relevant colour to highlight whether or not the detection is considered an offender due 
                to violations of the road rules. 

            Paramaters:
                * detection : (dict) : dictionary containing data of interest.
            Returns:
                * colour : (tuple[int, int, int]) : correct BGR values for bbox. 
        '''

        # If present, fetch detections offender status. Otherwise, false as not present. 
        offender = bool(detection.get('offender', False))

        # Assign bbox colour depending on detection status.
        return self.bbox_colours['offender'] if offender else self.bbox_colours['standard']

            
    def annotate_bbox_corners(self, frame, detection):

        '''
        Dynamically annotate a given detection adjusting the size and border radius of the annotated bounding box in relativity to 
            the detections size. 

        Parameters: 
            * frame : np.ndarray -> frame to be drawn upon.
            * detection : dict -> detection dictionary containing desired values to plot data points.
        
        Returns:
            * annotated_frame : np.ndarray -> Annotated frame with a detections given bounding box. 
        '''

        # Fetch detection bounding box values, typecast to full integer values. 
        x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])

        # Calculate detection dimensions.
        detection_size = min(x2 - x1, y2 - y1)

        # Dynamically calculate a detections line thickness and corner radius for annotation.
        corner_radius = max(min(int(detection_size * self.size_factor), self.max_corner_radius), self.min_corner_radius)
        thickness = max(min(int(detection_size * self.thickness_factor) * 2, self.max_thickness), self.min_thickness)

        # Fetch appropriate colour for detection.
        colour = self.fetch_bbox_colour(detection=detection)       

        # Store corner values within a list.
        bbox_corners = [
            ((x1 + corner_radius, y1 + corner_radius), 180, 270),
            ((x1 + corner_radius, y2 - corner_radius), 90, 180),
            ((x2 - corner_radius, y1 + corner_radius), 270, 360),
            ((x2 - corner_radius, y2 - corner_radius), 0, 90)
        ]
        
        # Iterate over bbox corner values, annotate with ellipses.
        for (center_x, center_y), start_angle, end_angle in bbox_corners:

            cv2.ellipse(
                frame, 
                (int(center_x), int(center_y)),
                (corner_radius, corner_radius),
                0, start_angle, end_angle,
                colour, 
                thickness
            )

        return frame
    

    def fetch_text_properties(self, label : str, frame : np.ndarray) -> tuple[tuple[int, int], float]:
        
        ''' '''

        current_font_scale = self.font_scale
        min_font_scale = 0.8
        max_text_width = frame.shape[1] - self.label_padding * 2

        text_width, text_height = cv2.getTextSize(label, self.font, current_font_scale, self.font_thickness)[0]

        while text_width > max_text_width and current_font_scale > min_font_scale:

            current_font_scale -= 0.1
            text_width, text_height = cv2.getTextSize(label, self.font, current_font_scale, self.font_thickness)[0]

        return (text_width, text_height), current_font_scale
    

    def calculate_label_position(self, bbox_bottom, center_point, text_size):

        text_width, text_height = text_size

        return {
            'x' : int(center_point[0] - text_width // 2), 'y' : int(bbox_bottom + text_height + self.label_padding),
            'width' : text_width + 2 * self.label_padding, 'height' : text_height + 2 * self.label_padding
        }
    

    def get_label_dimensions(self, label, frame):

        text_width, text_height = self.fetch_text_properties(label, frame)[0]

        box_width = text_width + 2 * self.label_padding
        box_height = text_height + 2 * self.label_padding

        return box_width, box_height


    def draw_label_background(self, frame, position) -> np.ndarray:

        ''' '''

        x, y = position['x'], position['y']
        w, h = position['width'], position['height']
        radius = self.border_radius

        cv2.rectangle(frame, (x + radius, y), ( x + w - radius, y + h), self.bg_colour, -1)
        cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), self.bg_colour, -1)

        label_corners = [
            # top left.
            (x + radius, y + radius, 180, 270),
            # top right.
            (x + w - radius, y + radius, 270, 360),
            # bottom left.
            (x + radius, y + h - radius, 90, 180),
            # bottom right.
            (x + w - radius, y + h - radius, 0, 90)
        ]

        for center_x, center_y, start_angle, end_angle in label_corners:

            cv2.ellipse(
                frame, 
                (center_x, center_y),
                (radius, radius),
                0, start_angle, end_angle,
                self.bg_colour,
                -1
            )

        return frame


    def draw_label_text(self, frame, label, position, text_size, font_scale) -> None:

        ''' '''

        text_x = position['x'] + self.label_padding
        text_y = position['y'] + text_size[1] + self.label_padding

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            self.font,
            font_scale,
            self.font_colour,
            self.font_thickness,
            lineType=cv2.LINE_AA
        )

    
    def annotate_label(self, frame : np.ndarray, detection : dict, vision_type : str) -> np.ndarray:

        '''
        '''

        detection_label = self.create_label(detection=detection, vision_type=vision_type)

        # Fetch detection bounding box values, typecast to full integer values. 
        y2= int(detection['y2'])
        center_x, center_y = calculate_center_point(detection)

        if vision_type == 'object_tracking':
            self.annotate_center_point_trail(frame=frame, center_points=detection.get('center_points', []))

        text_size, font_scale = self.fetch_text_properties(detection_label, frame)
        label_position = self.calculate_label_position(y2, (center_x, center_y), text_size)

        self.draw_label_background(frame, label_position)
        self.draw_label_text(frame, detection_label, label_position, text_size, font_scale)

        return frame
    

    def create_label(self, detection : dict, vision_type : str, captured_at = None) -> str:

        ''' Generate appropriate label to support required vision type. '''

        detection_labels = {
            'object_detection': f"ID: {detection.get('ID')} | Class: {detection.get('classname')} | Confidence Score: {detection.get('confidence_score'):.2f}",
            'object_tracking':  f"ID: {detection.get('ID')}",
            'speed_estimation': f"ID: {detection.get('ID')} | Speed: {detection.get('speed')}mph",
            'plate_reading': f"ID: {detection.get('ID')} | Plate: {detection.get('license_plate', {}).get('plate_text', '')}",
            'traffic_violation' : f"ID: {detection.get('ID')} | Captured: {captured_at} | Speed: {str(detection.get('speed'))}mph"
        }

        return detection_labels.get(vision_type, 'object_detection')


    def annotate_center_point(self, frame : np.ndarray, center_point : tuple[int, int]) -> np.ndarray:

        '''
            Annotate a detections center point onto the frame. 

            Paramaters:
                * frame : (np.ndarray) : The frame to be drawn upon.
                * center_point : (tuple[int, int]) : center x and center y coordinates of the detections center. 
            Returns:
                * frame : (np.ndarray) : The modified frame with a drawn center point.
        '''

        # Unpack center x and y values. 
        center_x, center_y = center_point

        # Use cv2 function to draw center point dot. 
        cv2.circle(
            frame,
            (center_x, center_y),
            self.center_point_radius,
            self.bbox_colours['offender'],
            self.thickness
        )

        return frame 
        

    def annotate_center_point_trail(self, frame : np.ndarray, center_points : list[tuple[int, int]]) -> np.ndarray:

        '''
            Annotate a detections center point onto the frame. 

            Paramaters:
                * frame : (np.ndarray) : The frame to be drawn upon.
                * detection : (dict) : Detection dictionary containing a list of its prior center points to annotate 
                    its trail over time. 
            Returns:
                * frame : (np.ndarray) : Modified frame where trail has been drawn. 
        '''

        if len(center_points) < 2:
            return frame

        # Iterate over center points list entries and render each onto frame.
        for x in range(1, len(center_points)):

            cv2.line(
                frame,
                center_points[x - 1],
                center_points[x],
                self.bbox_colours['trail'],
                self.trail_thickness
            )

        # Annotate last most center point.
        self.annotate_center_point(frame, center_points[0])
        # Render current most center point value.
        self.annotate_center_point(frame, center_points[-1])

        return frame
    

    def capture_traffic_violation(self, frame, detection, captured_at):

        ''' '''

        ''' Plate Extraction. '''

        license_plate = detection.get('license_plate', {}).get('plate_text', 'OCCLUDED')

        ''' Fetch bbox coords. '''
        x1, x2, y1, y2 = detection['x1'], detection['x2'], detection['y1'], detection['y2']
        

        ''' Copy and annotate frame. '''
        captured_frame = frame.copy()
        # Annotate detection of concern.
        annotated_frame = self.annotate_bbox_corners(captured_frame, detection)

        ''' Calcualte padded coords for cropping. '''
        h, w = annotated_frame.shape[:2]
        padded_x1 = int(max(0, x1 - self.padding))
        padded_y1 = int(max(0, y1 - self.padding))
        padded_x2 = int(min(w, x2 + self.padding))
        padded_y2 = int(min(h, y2 + self.padding))

        ''' Crop and upscale detection. '''

        cropped_frame = annotated_frame[padded_y1:padded_y2 + self.padding, padded_x1:padded_x2 + self.padding]
        overlay_width = w // 4
        overlay_height = int((padded_y2 - padded_y1) * (overlay_width / (padded_x2 - padded_x1)))
        upscaled_cropped_frame = cv2.resize(cropped_frame, (overlay_width, overlay_height))

        ''' Append crop to annotated fram.e '''
        overlay_x = w - overlay_width - self.padding
        overlay_y = self.padding
        if overlay_y + overlay_height <= h and overlay_x + overlay_width <= w:

            annotated_frame[
                overlay_y:overlay_y + overlay_height,
                overlay_x:overlay_x + overlay_width
            ] = upscaled_cropped_frame

        ''' Append license plate label to frame. '''
        # Add license plate label separately.
        plate_label_bg_height = annotated_frame.shape[0] // 14 
        plate_label_bg_y = overlay_y + overlay_height - plate_label_bg_height
        plate_label_bg_x = overlay_x
        plate_label_bg_width = overlay_width

        # Create a black rectangle for the bottom label
        annotated_frame[plate_label_bg_y:overlay_y + overlay_height, plate_label_bg_x:overlay_x + overlay_width] = (0, 0, 0)

        # Center the label text
        (plate_text_width, plate_text_height), plate_font_scale = self.fetch_text_properties(license_plate, annotated_frame)
        text_x = plate_label_bg_x + (plate_label_bg_width - plate_text_width) // 2
        text_y = plate_label_bg_y + (plate_label_bg_height + plate_text_height) // 2

        # Add the PLATE label text
        cv2.putText(
            annotated_frame,
            license_plate, 
            (text_x, text_y),  
            self.font,
            plate_font_scale,  
            self.font_colour,
            self.font_thickness
        )

        ''' Append capture metadata to bottom of frame. '''

        # Create bottom label. 
        label = self.create_label(detection, 'traffic_violation', captured_at)
        (label_text_width, label_text_height), label_font_scale = self.fetch_text_properties(label, annotated_frame)
        
        label_height = label_text_height + self.padding * 2
        label_y1 = h - label_height

        annotated_frame[label_y1:h, 0:w] = (0, 0, 0)
        
        bg_cx = w // 2
        bg_cy = label_y1 + label_height // 2

        label_pos_x1 = bg_cx - (label_text_width // 2)  
        label_pos_y1 = bg_cy + label_text_height // 2

        cv2.putText(
            annotated_frame,
            label,
            (label_pos_x1, label_pos_y1),
            self.font,
            label_font_scale,
            self.font_colour,
            self.font_thickness,
        )

        return annotated_frame
    