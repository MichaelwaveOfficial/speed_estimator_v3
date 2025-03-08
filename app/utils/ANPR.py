import easyocr 
import cv2 
import numpy as np 
import re 
from .ObjectDetection import ObjectDetection
import time 
from rapidfuzz import fuzz


class ANPR(object):

    ''' '''

    def __init__(self, detection_model : ObjectDetection, ocr_lang='en', ocr_gpu=True, deregistration_time : int = 12, plate_similarity_threshold : int = 85):
        
        ''' '''

        # Use object detection modules detection logic. 
        self.detection_model = detection_model

        # Initialise ocr text reader. 
        self.ocr_text_reader = easyocr.Reader([ocr_lang], gpu=ocr_gpu)

        # Maximum plate length allowed.
        self.UK_MAX_PLATE_LENGTH = 7 
        
        # Regular expression representing plate format. 
        self.UK_PLATE_REGEX =  re.compile(r'^([A-Z]{2})([0-9]{2})([A-Z]{3})$')

        self.detection_plates = {} 

        self.deregistration_time = deregistration_time

        self.plate_similarity_threshold = plate_similarity_threshold

        # Mapping dictionaries for characters that can be easily mistaken.
        self.char_2_int_dict = {'O': '0','I': '1','J': '3','A': '4','G': '6','S': '5'}
        self.int_2_char_dict = {'0': 'O','1': 'I','3': 'J','4': 'A','6': 'G','5': 'S'}


    def process_detection_plates(self, frame : np.ndarray, detections : list[dict]) -> list[dict]:

        ''' '''

        updated_at = time.time()

        # Iterate over each detection dictionary entry.
        for detection in detections:

            ID = detection.get('ID')

            detection.setdefault('license_plate', {}).setdefault('plate_text', '')
            self.detection_plates.setdefault(ID, {'plate_text' : 'OCCLUDED', 'last_seen' : 0})

            # Check if plate has already been handled. 
            if updated_at - self.detection_plates[ID]['last_seen'] < self.deregistration_time:
                detection.setdefault('license_plate', {})['plate_text'] = self.detection_plates[ID]['plate_text']
                continue

            plate_found = False 

            # Crop frame for focusing model inference. 
            detection_frame_crop = self.crop_frame_from_detection_data(frame, detection)

            # Detect licence plate through applied transfer learning. 
            plate_detections = self.detection_model.run_inference(detection_frame_crop)

            if plate_detections:

                sorted_plates = sorted(plate_detections, key = lambda plate: plate['confidence_score'], reverse=True)

                for plate in sorted_plates:
           
                    license_plate = self.extract_license_plate(frame, detection, plate)

                    # If plate text has been returned.
                    if license_plate:

                        self.detection_plates[ID]['plate_text'] = license_plate
                        self.detection_plates[ID]['last_seen'] = updated_at
                        plate_found = True
                        continue
                    
            detection['license_plate']['metadata'] = plate_detections
            detection['license_plate']['plate_text'] = self.detection_plates[ID]['plate_text']

            if not plate_found:
                self.detection_plates[ID]['plate_text'] = 'OCCLUDED'

        self.prune_outdated_objects(updated_at)

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

        cv2.imshow('cropped plate', cropped_license_plate)

        ocr_read_plate_text = self.read_license_plate(cropped_license_plate)
        
        return self.correct_plate_text(ocr_read_plate_text)


    def crop_frame_from_detection_data(self, frame, detection):

        x1, y1, x2, y2 = map(int, (detection['x1'], detection['y1'], detection['x2'], detection['y2']))

        return frame[y1: y2, x1:x2]
    

    def validate_plate_format(self, plate_text : str) -> bool:
        return self.UK_PLATE_REGEX.match(plate_text) is not None


    def correct_plate_text(self, raw_plate_text : str) -> str | None:

        ''' 
            Function to process and clean OCR model output to rectify plate text
                to expected output matching UK license plate formats.
        '''

        if not raw_plate_text or len(raw_plate_text) <= 0:
            return None

        # Clean parsed plate text removing spaces. trailing characters and converting to upper case.
        cleansed_plate_text = ''.join(raw_plate_text).upper().strip().replace(' ', '')
        print('cleansed text', cleansed_plate_text)

        # Return early if validation already matches.
        if self.validate_plate_format(cleansed_plate_text):
            print('early validation achieved ', cleansed_plate_text)
            return cleansed_plate_text
        
        # If too short or too long, cull result.
        if not 6 <= len(cleansed_plate_text) <= 8:
            print(cleansed_plate_text, 'disposed, did not pass length checks.')
            return None 
        
        # Group areas of interest from parsed text compared to the regex.
        cleansed_plate_text_groups = self.UK_PLATE_REGEX.match(cleansed_plate_text)

        # If no groups, persist with fallback cleansing.
        if not cleansed_plate_text_groups:

            if len(cleansed_plate_text) == self.UK_MAX_PLATE_LENGTH:

                area_code, reg_year, suffix = cleansed_plate_text[:2], cleansed_plate_text[2:5], cleansed_plate_text[5:]

                # Convert misinterpreted characters into expected group types.
                # area_code ++ suffix should be letters.
                # reg_year should be numeric.
                area_code = self.convert_text_groups(area_code, 'area_code')
                reg_year = self.convert_text_groups(reg_year, 'reg_year')
                suffix = self.convert_text_groups(suffix, 'suffix')

            else:

                print('plate disposed ', cleansed_plate_text)
                return None

        else:               

            # Assign groups to variables for greater access ++ contextualisation.
            area_code, reg_year, suffix = cleansed_plate_text_groups.groups()
            print(f'area_code:{area_code}, year:{reg_year}, suffix:{suffix}')

            # Convert misinterpreted characters into expected group types.
            # area_code ++ suffix should be letters.
            # reg_year should be numeric.
            area_code = self.convert_text_groups(area_code, 'area_code')
            reg_year = self.convert_text_groups(reg_year, 'reg_year')
            suffix = self.convert_text_groups(suffix, 'suffix')
        
        # Aggregrate groups together, form corrected plate text. 
        corrected_plate_text = area_code + reg_year + suffix
        print('corrected character plate ', corrected_plate_text)

        # Generate similarity percentage from corrected and original text.
        plate_similarity_ratio = fuzz.ratio(corrected_plate_text, cleansed_plate_text)

        print(
            'input plate',
            cleansed_plate_text,
            'corrected_plate',
            corrected_plate_text,
            'at confidence', plate_similarity_ratio
        )
        
        # If similarity float meets set threshold ++ meets validation, return final plate.
        if plate_similarity_ratio >= self.plate_similarity_threshold and \
            self.validate_plate_format(corrected_plate_text):
            print('final product ', corrected_plate_text)
            return corrected_plate_text
        
        # If no conditions met, return none.
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

        ''' '''

        try:
            # Take cropped_plate frame and preprocess it for better model digestion.
            processed_plate = self.preprocess_plate(cropped_plate)
            # Use OCR model to read text from pre-processed plate.
            return self.ocr_text_reader.readtext(processed_plate, detail=0)
        except Exception as e:
            print(f'OCR model failed to read text.\n{e}')

        return None
    

    def prune_outdated_objects(self, updated_at):

        '''
            Iterate over parameterised detections and prune those exceeding the set time limit threshold.

            Parameters:
                * parsed_detections : list[dict] -> list of detection data entries.
            Returns:
                * None. 
        '''

        # Initialise list to store ID values of detections to be pruned. 
        stale_detections = [ID for ID, detection in self.detection_plates.items()
                            if (updated_at - detection['last_seen']) > self.deregistration_time]

        # Iterate over the IDs present. 
        for ID in stale_detections:
            # Use IDs to delete entries from tracked objects. 
            del self.detection_plates[ID]

    
    def convert_text_groups(self, text_grouping : str, group : str) -> str:

        '''
        
        '''

        if group in ['area_code', 'suffix']:
            return ''.join(self.int_2_char_dict.get(char, char) for char in text_grouping)
        elif group == 'reg_year':
            return ''.join(self.char_2_int_dict.get(char, char) for char in text_grouping)
        else:
            return text_grouping