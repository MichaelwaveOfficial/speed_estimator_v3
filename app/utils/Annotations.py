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

            frame = self.annotate_bbox_corners(frame, detection)

            frame = self.annotate_labels_onto_frame(frame, detection, vision_type)

        return frame

            
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

        ''' Ingest and parse data to dynmaically handle sizing. '''

        # Fetch detection bounding box values, typecast to full integer values. 
        x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])

        # If present, fetch detections offender status. Otherwise, false as not present. 
        offender = bool(detection.get('offender', False))

        # Calculate detection dimensions.
        detection_width = x2 - x1
        detection_height = y2 - y1
        detection_size = min(detection_width, detection_height)

        # Dynamically calculate a detections line thickness and corner radius for annotation.
        detection_corner_radius = max(min(int(detection_size * self.size_factor), self.max_corner_radius), self.min_corner_radius)
        detection_thickness = max(min(int(detection_size * self.thickness_factor) * 2, self.max_thickness), self.min_thickness)

        # Assign bbox colour depending on detection status.

        colour = self.bbox_colours['offender'] if offender else self.bbox_colours['standard']

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

        return frame
    

    def adjust_font_scale(self, frame):

        base_scale = self.font_scale

        self.font_scale(max(1.5, (frame.shape[0] / frame.shape[1])) * base_scale)
    
    
    def get_text_dimensions(self, text, frame):

        min_font_scale = 1.25

        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
        text_width, text_height = text_size
        max_text_width = frame.shape[1] - self.label_padding

        while text_width > max_text_width and self.font_scale > min_font_scale:
            self.font_scale -= 0.2
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
            text_width, text_height = text_size

        if text_width > max_text_width:
            while cv2.getTextSize(
                text + ' ', self.font, self.font_scale, self.font_thickness
                )[0][0] > max_text_width and \
                     len(text) > 1:
                        text = text[:-1]

        return text_width, text_height
    

    def get_label_dimensions(self, label):

        (text_width, text_height), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)

        box_width = text_width + 2 * self.label_padding
        box_height = text_height + 2 * self.label_padding

        return box_width, box_height


    def draw_label_bg(self, label, frame, x1, y1):

        box_width, box_height = self.get_label_dimensions(label=label)

        # Draw the rounded rectangle (background)
        cv2.rectangle(
            frame,
            (x1 + self.border_radius, y1),  # Top-left inner corner
            (x1 + box_width - self.border_radius, y1 + box_height),  # Bottom-right inner corner
            self.bg_colour,
            -1  # Fill the rectangle
        )
        cv2.rectangle(
            frame,
            (x1, y1 + self.border_radius),
            (x1 + box_width, y1 + box_height - self.border_radius),
            self.bg_colour,
            -1
        )
        # Draw the rounded corners
        cv2.ellipse(
            frame,
            (x1 + self.border_radius, y1 + self.border_radius),  # Top-left corner center
            (self.border_radius, self.border_radius),
            180, 0, 90,
            self.bg_colour,
            -1
        )
        cv2.ellipse(
            frame,
            (x1 + box_width - self.border_radius, y1 + self.border_radius),  # Top-right corner center
            (self.border_radius, self.border_radius),
            270, 0, 90,
            self.bg_colour,
            -1
        )
        cv2.ellipse(
            frame,
            (x1 + self.border_radius, y1 + box_height - self.border_radius),  # Bottom-left corner center
            (self.border_radius, self.border_radius),
            90, 0, 90,
            self.bg_colour,
            -1
        )
        cv2.ellipse(
            frame,
            (x1 + box_width - self.border_radius, y1 + box_height - self.border_radius),  # Bottom-right corner center
            (self.border_radius, self.border_radius),
            0, 0, 90,
            self.bg_colour,
            -1
        )


    def annotate_labels_onto_frame(self, frame, detection, vision_type : str):

        ''' '''

        x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])

        center_x, center_y = calculate_center_point(detection)

        ID, classname, confidence_score = str(detection.get('ID')), str(detection.get('classname')), str(round(detection.get('confidence_score'), 2))
        speed = str(detection.get('speed', 0))

        label = ''

        if vision_type == 'object_detection':
            label = f"ID : {ID} | Classname: {classname} | Confidence Score: {confidence_score}"

        if vision_type == 'object_tracking':
            label = f"ID : {ID}"

            self.annotate_center_point_trail(frame, detection)

        if vision_type == 'speed_estimation':
            label = f"ID: {ID} | Speed: {speed}mph"

        if vision_type == 'plate_reading':
            label = f'ID: {ID} | Plate: NA11 BHJ'
    
        _, box_height = self.get_label_dimensions(label)
        text_width, text_height = self.get_text_dimensions(label, frame)

        label_x =  int(center_x - text_width / 2)
        label_y = int(y2 + text_height + self.label_padding)
        
        self.draw_label_bg(label, frame, label_x, label_y)

        # Add the text on top of the background
        text_position = (label_x + self.label_padding, label_y + text_height + self.label_padding)

        annotated_frame = cv2.putText(
            frame,
            label,
            text_position,
            self.font,
            self.font_scale,
            self.font_colour,
            self.font_thickness
        )

        return annotated_frame


    def crop_detection(self, frame, detection, captured_at):

        ''' '''

        test_plate = 'NA11 BHJ'

        x1, x2, y1, y2 = detection['x1'], detection['x2'], detection['y1'], detection['y2']
        ID = detection['ID']

        annotated_frame = frame.copy()

        # Annotate detection of concern.
        annotated_frame = self.annotate_bbox_corners(annotated_frame, detection)

        h, w = annotated_frame.shape[:2]

        padded_x1 = int(max(0, x1 - self.padding))
        padded_y1 = int(max(0, y1 - self.padding))
        padded_x2 = int(min(w, x2 + self.padding))
        padded_y2 = int(min(h, y2 + self.padding))

        cropped_frame = annotated_frame[padded_y1:padded_y2 + self.padding, padded_x1:padded_x2 + self.padding]

        overlay_width = w // 4
        overlay_height = int((padded_y2 - padded_y1) * (overlay_width / (padded_x2 - padded_x1)))

        upscaled_cropped_frame = cv2.resize(cropped_frame, (overlay_width, overlay_height))

        overlay_x = w - overlay_width - self.padding
        overlay_y = self.padding

        if overlay_y + overlay_height <= h and overlay_x + overlay_width <= w:

            annotated_frame[
                overlay_y:overlay_y + overlay_height,
                overlay_x:overlay_x + overlay_width
            ] = upscaled_cropped_frame

        # Add license plate label separately (example)
        plate_label_bg_height = annotated_frame.shape[0] // 14  # Adjust as needed
        plate_label_bg_y = overlay_y + overlay_height - plate_label_bg_height
        plate_label_bg_x = overlay_x
        plate_label_bg_width = overlay_width

        # Create a black rectangle for the label
        annotated_frame[plate_label_bg_y:overlay_y + overlay_height, plate_label_bg_x:overlay_x + overlay_width] = (0, 0, 0)

        # Center the label text
        text_width, text_height = cv2.getTextSize(test_plate, self.font, self.font_scale, self.font_thickness)[0]
        text_x = plate_label_bg_x + (plate_label_bg_width - text_width) // 2
        text_y = plate_label_bg_y + (plate_label_bg_height + text_height) // 2

        # Add the label text
        cv2.putText(
            annotated_frame,
            test_plate, 
            (text_x, text_y),  
            self.font,
            self.font_scale,  
            self.font_colour,
            self.font_thickness
        )


        label = f'ID: {ID} | Captured: {captured_at} | Speed: {str(detection["speed"])}mph'
        
        final_frame = self.annotate_label_onto_bg(annotated_frame, label)

        return final_frame
    

    def annotate_label_onto_bg(self, frame, label, plate_present=False):

        frame = frame.copy()

        h, w, _ = frame.shape

        text_width, text_height = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]

        label_height = text_height + self.padding * 2
        label_y1 = h - label_height

        if plate_present:
            frame[label_y1:h, 0:w // 4] = (0, 0, 0)
        else:
            frame[label_y1:h, 0:w] = (0, 0, 0)

        bg_cx = w // 2
        bg_cy = label_y1 + label_height // 2

        label_pos_x1 = bg_cx - (text_width // 4)  # Quarter adjustment for centering
        label_pos_y1 = bg_cy + text_height // 2

        current_font_scale = max(0.5, min(1.0, w / 1000))  # Dynamic font scale

        cv2.putText(
            frame,
            label,
            (label_pos_x1, label_pos_y1),
            self.font,
            current_font_scale,
            self.font_colour,
            self.font_thickness,
        )

        return frame


    def annotate_center_point(self, frame, center_point):

        ''' '''

        center_x, center_y = center_point

        cv2.circle(
            frame,
            (center_x, center_y),
            self.center_point_radius,
            self.bbox_colours['offender'],
            self.thickness
        )
        

    def annotate_center_point_trail(self, frame, detection):

        ''' '''

        if 'center_points' not in detection:
            return frame
    
        center_points_list = detection['center_points']
        points_list_length = len(center_points_list)

        initial_center_point = center_points_list[0]
        final_center_point = center_points_list[-1]

        self.annotate_center_point(frame, initial_center_point)

        for x in range(1, points_list_length):

            cv2.line(
                frame,
                center_points_list[x - 1],
                center_points_list[x],
                self.bbox_colours['trail'],
                self.trail_thickness
            )

        self.annotate_center_point(frame, final_center_point)

        return frame
