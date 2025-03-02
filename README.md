
# 🏠 AI-Powered Traffic Management System.
## ⚠️ Work in Progress

This project is currently an ongoing development and currently visible for demonstration purposes only. Further documentation
    and bug fixing will be provided as the project expands and features are implemented.

## Overview

Embracing the future of AI, automating processes that directly impact our daily lives -- like traffic management. This system is to demonstrate how a
simple implementation can be used to enhance road safety and efficiency whilst providing data driven insights. 

By leveraging deep learning techniques to monitor and manage traffic in real time through the integration of an object detection model from Ultralytics, detecting vehicles of interest, tracks their movement and estimates their speeds. With this implementation, roads can be made safer by handling negligent road users and handling them accordingly.

    ### Key Benefits:

        * Enhanced Road Safety: Quickly identify offenders violating trafic laws to enable proactive interventions.
        * Automation: Monitoring process is now automated to alleviate the need for manaul, human driven, oversight.
        * Data-Driven Insights: Returns detailed analytics on traffic which can help lead to better informed decision making.


## 📖 Table of Contents

-[Features](#Features)
-[Prerequisites](#Prerequisites)
-[Setup](#Setup)
-[Configuration](#Configuration)
-[RunningTheProject](#Run)

# 🚀 Features

    ✔️ Real-time Object Detection with YOLO V11
        - High confidence detection for mitigation of false positives.
        - Classname filtration to cull irrelevant detections.

        ![object detection](/docs/detection.jpg)

    ✔️ Object Tracking
        - Assign IDs to detections.
        - Estimate detection tracjectory.

        ![object tracking](/docs/tracking.jpg)

    ✔️ Speed Estimation
        - Weighted average of frame based and average speed estimation.
        - Vehicles exceeding the set limit are captured.
        - Customisable speed limits.

        ![object speed estimation](/docs/estimation.jpg)

    ✔️ Annotations
        - Classnames
        - Confidence Scores
        - IDs
        - Speed

# 🔧 Prerequisites

    * In order for this project to be viable, these components are required. 

    ### Hardware:

        * Devices with CUDA comptabible GPUs are favoured. Application should be viable on most devices however
            performance will be staggered, unable to leverage hardware acceleration.

    ### Software:

        * Python 3.11
        * Ultralytics YOLO V11 (https://www.ultralytics.com/yolo)
            - Current V11 model weights included, access site should there be model issues to access
                most up to date.
        * OpenCV
        * Numpy
        

# 🛠 Setup

Clone this git repo with:

    * git clone https://github.com/MichaelwaveOfficial/speed_estimator_v3.git

    * cd project folder

Install dependencies:

    * pip install -r requirements.txt 


# ▶️ Run

Run the project:

    * python main.py

