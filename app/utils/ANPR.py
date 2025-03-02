import easyocr 
import cv2 
import numpy as np 
import re 
from .ObjectDetection import ObjectDetection
from .BboxUtils import calculate_center_point


class ANPR(object):

    ''' '''

    def __init__(self, detection_model : ObjectDetection, ocr_lang='en', ocr_gpu=True):
        
        ''' '''

        # Use object detection modules detection logic. 
        self.detection_model = detection_model

        # Initialise ocr text reader. 
        self.ocr_text_reader = easyocr.Reader([ocr_lang], gpu=ocr_gpu)

        # Maximum plate length allowed.
        self.max_character_length = 7 
        
        # Regular expression representing plate format. 
        self.UK_PLATE_REGEX = r'^([A-Z]{2})([0-9]{2})([A-Z]{3})$'


    def process_detection_plates(self, frame : np.ndarray, detections : list[dict]) -> list[dict]:

        ''' '''

        # Iterate over each detection dictionary entry.
        for detection in detections:

            plate_found = False 

            detection.setdefault('license_plate', {}).setdefault('plate_text', 'OCCLUDED')

            detection_frame_crop = self.crop_frame_from_detection_data(frame, detection)

            # Detect licence plate through applied transfer learning. 
            plate_detections = self.detection_model.run_inference(detection_frame_crop)

            if not plate_detections:
                detection['license_plate']['plate_text'] = 'OCCLUDED'
                continue
    
            # Iterate over plates within the plate detections dictionary. 
            for plate in plate_detections:

                if self.match_plate_to_car(detection, plate):

                    license_plate = self.extract_license_plate(frame, detection, plate)

                    # If plate text has been returned.
                    if license_plate:

                        detection['license_plate'] = {
                            **plate,
                            'plate_text' : license_plate
                        }
                        plate_found = True
                        break
                    
            if not plate_found:
                detection['license_plate']['plate_text'] = 'OCCLUDED'

        return detections
    

    def extract_license_plate(self, frame, detection, license_plate):

        ''' '''

        abs_coords = (
            int(detection['x1'])  + int(license_plate['x1']),
            int(detection['y1'])  + int(license_plate['y1']),
            int(detection['x1'])  + int(license_plate['x2']),
            int(detection['y1'])  + int(license_plate['y2']),
        )

        # y1:y2, x1:x2
        cropped_license_plate = frame[abs_coords[1]:abs_coords[3], abs_coords[0]:abs_coords[2]]

        ocr_read_plate_text = self.read_license_plate(cropped_license_plate)
        raw_plate_text = ''.join(ocr_read_plate_text).upper().strip()
        
        return self.correct_plate_text(raw_plate_text)


    def crop_frame_from_detection_data(self, frame, detection):

        x1, y1, x2, y2 = map(int, (detection['x1'], detection['y1'], detection['x2'], detection['y2']))

        return frame[y1: y2, x1:x2]
    

    def validate_plate_format(self, plate_text : str) -> bool:
        return re.match(self.UK_PLATE_REGEX, plate_text) is not None


    def correct_plate_text(self, plate_text : str) -> str | None:

        ''' '''

        if len(plate_text) != 7:
            return None 

        # Mapping dictionaries for characters that can be easily mistaken.
        character_conversion = {'O': '0','I': '1','J': '3','A': '4','G': '6','S': '5'}
        integer_conversion = {'0': 'O','1': 'I','3': 'J','4': 'A','6': 'G','5': 'S'}
        
        plate_text = plate_text.upper()

        plate_text_groups = re.match(self.UK_PLATE_REGEX, plate_text)

        if not plate_text_groups:
            return None

        area_code, reg_year, suffix = plate_text_groups.groups()


        area_code = ''.join(integer_conversion.get(key, key) for key in area_code)
        reg_year = ''.join(character_conversion.get(key, key) for key in reg_year)
        suffix = ''.join(integer_conversion.get(key, key) for key in suffix)
        
        corrected_plate_text = area_code + reg_year + suffix

        print(plate_text)

        if self.validate_plate_format(plate_text=corrected_plate_text):
            return corrected_plate_text
        
        return None
     

    def preprocess_plate(self, cropped_plate : np.ndarray, kernel_size : int = 3, operation_iterations : int = 1) ->  np.ndarray:

        ''' '''

        # Size of the kernal in pixels (w, h). 
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Convert frame to greyscale, reducing colour channels as a separation of concerns. 
        grey_cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

        # Threshold the image to generate a greater contract making text more legible for the model. 
        thresholded_plate = cv2.threshold(
            grey_cropped_plate,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]

        # Apply morphological operations to mitigate artifacts generated by noise.
        morphologically_applied_plate = cv2.morphologyEx(thresholded_plate, cv2.MORPH_CLOSE, kernel, operation_iterations)

        # Apply a gaussian blur to smooth the image further. 
        smoothed_plate = cv2.GaussianBlur(morphologically_applied_plate, (kernel_size, kernel_size), 0)

        return smoothed_plate
    

    def read_license_plate(self, cropped_plate):

        try:
            processed_plate = self.preprocess_plate(cropped_plate)
            # Use OCR model to read text from pre-processed plate.
            return self.ocr_text_reader.readtext(processed_plate, detail=0)
        except Exception as e:
            print(f'OCR failed\n{e}')

        return None
    

    def match_plate_to_car(self, detection, license_plate):

        ''' '''

        plate_cx, plate_cy = calculate_center_point(license_plate)
        
        abs_plate_cx = plate_cx + detection['x1']
        abs_plate_cy = plate_cy + detection['y1']

        return (
            detection['x1'] <= abs_plate_cx <= detection['x2'] and 
            detection['y1'] <= abs_plate_cy <= detection['y2']
        )
