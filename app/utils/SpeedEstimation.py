from .BboxUtils import measure_euclidean_distance


class SpeedEstimation(object):

    '''
        Module to estimate the speed of a detection leveraging the time taken between its detected center points. 
    '''

    
    def __init__(self, frame_rate : int = 30, speed_limit : int = None):
        
        # Supplied medias frame rate. 
        self.frame_rate = frame_rate

        # Dictionary to store a detections most recent speeds. 
        self.detections_speeds = {}

        # Speed limit set by user. 
        self.speed_limit = speed_limit

    
    def apply_estimations(self, detections : list[dict], frame_rate : int, speed_limit : int) -> list[dict]:

        '''
        
        '''

        for detection in detections:

            if 'ppm' not in detection:
                detection['ppm'] = detection.get('ppm', self.calibrate_ppm(detection=detection))

            if 'previous_center_point' in detection and \
                'current_center_point' in detection:

                frame_elapsed_time = detection['current_timestamp'] - detection['previous_timestamp']
                avg_elapsed_time = detection['current_timestamp'] - detection['first_timestamp']

                ''''
                
                    Frame elapsed time always ZERO. ASSIGNMENT OR CALCULATIONS WRONG??
                '''

                if frame_elapsed_time > 0 and avg_elapsed_time > 0:

                    frame_distance = measure_euclidean_distance(
                        detection['previous_center_point'], detection['current_center_point']
                    )

                    avg_distance = measure_euclidean_distance(
                        detection['first_center_point'], detection['current_center_point']
                    )

                    frame_speed = self.calculate_speed(frame_distance, detection['ppm'], frame_elapsed_time, frame_rate)
                    avg_speed = self.calculate_speed(avg_distance, detection['ppm'], avg_elapsed_time, frame_rate)

                    smoothed_speed = self.speed_weighted_average(frame_speed, avg_speed)

                    detection['speed_history'] = [smoothed_speed]

                    detection['speed'] = round(self.detection_rolling_average(detection['speed_history']), 2)

                    print(f'detection ID:{detection["ID"]}\ndetection speed:{detection["speed"]}')

            detection['previous_center_point'] = detection['current_center_point']

        return detections
    

    def capture_offense(self, speed: float):

        '''
            WIP

                * If calculated speed > set limit --> red bbox, otherwise green.
                * Capture screencap, bbox annotation + speed.
                * How much over the limit?
        '''

        if speed > self.speed_limit:
            return True

        return False
    

    def speed_weighted_average(self, raw_speed : float, avg_speed : float, alpha : int = 0.7) -> float:

        '''
            Calculate a detections weighted average to smooth the output of the speed calculations. This is done by leveraging
                its real time speed from frame by frame calculations and its overall average speed. 

            Parameters:
                * raw_speed : float -> Instantaneous speed, frame by frame. 
                * avg_speed : float -> Overall average speed of the detection up to a given point. 
                * alpha : int -> The weighting factor for the raw speed. Must be between 0 and 1:
                    - alpha = 1: Use only the raw speed.
                    - alpha = 0: Use only the average speed.
                    - 0 < alpha < 1: Blend the raw speed and average speed proportionally.

            Returns:
                * float -> Smoothed speed, calculated as the weighted average of both the raw and average speeds. 
        '''

        return (alpha * raw_speed) + ((1 - alpha) * avg_speed)
    

    def calculate_speed(self, pixel_distance : float, ppm : float, elapsed_time : float, frame_rate : int):
        
        '''
            Function to calculate the speed for a detections given distance in pixels, its pixels per meters value and 
                the elapsed time it has taken to travel that distance. 

            Parameters:
                * pixel_distance : float -> distance covered by the detection.
                * ppm : float -> calculated pixel per meter value to guestimate how much distance is being covered in the media.
                * elapsed_time : float -> Time taken for the detection to travel that pixel distance. 
                * frame_rate : int -> Given videos current frame rate. 

            Returns:
                * float -> Calculated speed with the correct given unit of measurement. 
        '''
        
        # Divide elapsed time by frame rate. 
        elapsed_time = (elapsed_time / frame_rate)
        
        # Ensure provided values are not null.
        if elapsed_time <= 0 or ppm <= 0:
            raise ValueError(f'Elapsed time :{elapsed_time} and/or PPM : {ppm} values must be greater than zero.')

        # Conver the distance into meters. 
        distance_in_meters = pixel_distance / ppm
        
        # Calculate the speed as mps. 
        speed_mps = distance_in_meters  / elapsed_time

        # Return the calculated speed. 
        return self.unit_conversion(speed=speed_mps)
    

    def unit_conversion(self, speed : float, measurement : str = 'mph') -> float:

        '''
            Simple helper function to convert speed measurement into a recognised unit of measurement determined by the user. 

            Parameters:
                * speed : float -> The detections calculated speed.
                * measurement : str -> The unit of measurement desired.

            Returns:
                * float -> speed multipled by the concered unit of measuremnt. 
        
        '''

        # Conversion values dictionary contianing key value pairs. 
        conversion_factors = {
            'mph': 2.23,  
            'kmh': 3.6      
        }

        # If provided measurements not in conversions dictionary, let user know. 
        if measurement not in conversion_factors:
            raise ValueError(f"Unsupported measurement unit: {measurement}")

        # Return speed multiplied by specified conversion factor. 
        return speed * conversion_factors[measurement]
    
    
    def calibrate_ppm(self, detection : dict) -> float:

        '''
            Attempt to calibrate pixels per meter by obtaining the average real-world dimensions for a detection and 
                leveraging it against the detections bounding box width and height ro determine its scale.

            Parameters:
                * detection : dict -> single detection dictionary containing metadata. 

            Returns:
                * float -> pixels per meter values for that detection to later be used for speed estimation.
        '''

        # Accumulate the detections real world dimensions.
        real_width, real_height = detection['avg_class_dimensions']['width'], detection['avg_class_dimensions']['height']

        # Get the detections bbox width and height.
        detection_width = abs(detection['x2'] - detection['x1'])
        detection_height = abs(detection['y2'] - detection['y1'])

        # Calculate its ppm width and height. 
        ppm_width = detection_width / real_width
        ppm_height = detection_height / real_height

        # Return detections scale.
        return (ppm_width + ppm_height) / 2
    

    def detection_rolling_average(self, speeds : dict[float], window_length : int = 5):

        '''
            Calculates a rolling average of detections speeds from the paramaterised window length.

            Parameters:
                * detection_speeds : dict[float] -> detections stored speeds. 
                * window_length : int -> number of detections to iterate over. 

            Returns:
                * float -> Rolling average of speeds for given number of detections. 
        '''

        return sum(speeds[-window_length:]) / min(len(speeds), window_length)
