### In-Line Tablet Quality Control using Computer Vision | Real-Time Defect Detection Pipeline

## Abstract
Conventional pharmaceutical tablet quality control relies largely on off-line testing performed on limited samples, which is time-consuming, labor-intensive, and prone to missing critical defects. This limitation highlights the need for an automated in-line inspection approach capable of ensuring consistent product quality while minimizing waste.

This project presents an in-line tablet quality control system based on machine vision and deep learning for real-time inspection of tablet cores. The proposed system employs a conveyor-mounted imaging setup integrated with controlled illumination and a high-resolution RGB camera to capture detailed surface images of tablets under motion.

A custom dataset was developed using images collected across multiple orientations and lighting conditions to ensure robustness. Two common surface defect categories, namely edge chipping and capping, were annotated and used for model training and evaluation.

A YOLO-based deep learning model was trained and optimized to perform real-time defect detection, focusing on high accuracy, low latency, and industrial throughput. Additionally, a HMI dashboard was developed to provide live inspection statistics and quality reports, ensuring traceability and compliance with audit requirements.

By enabling early defect detection and intervention, the system improves patient safety, manufacturing efficiency, and reduces material wastage, contributing to a more sustainable production ecosystem.


## Sustainable Development Goals (SDGs)
<p align="left"> <img src="https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-03.jpg" width="80"/> <img src="https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-09.jpg" width="80"/> <img src="https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-12.jpg" width="80"/> </p> 


Core Technologies:
<p align="left"> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="50"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="50"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" width="50"/> <img src="https://upload.wikimedia.org/wikipedia/commons/3/3b/Grafana_icon.svg" width="50"/> <img src="https://www.vectorlogo.zone/logos/influxdata/influxdata-icon.svg" width="50"/> <img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="50"/> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/raspberrypi/raspberrypi-original.svg" width="50"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="50"/> 


## System Architecture:
```
Camera → Raspberry Pi → Image Processing (OpenCV)
           ↓
      YOLO Model (Inference)
           ↓
   Defect Detection Output
           ↓
   InfluxDB → Grafana Dashboard
```


## Features:
-Real-time tablet inspection \
-Detection of edge chipping and capping defects \
-Live dashboard for monitoring production quality \
-Scalable and containerized deployment using Docker 


## System Setup
![Setup](images/setup.png)


## Sample Dataset Images
![Dataset](images/ds.png)
if you want dataset, kindly contact at anooshakhalid999@gmail.com


## Detection Results
![Results](images/detection.png)
![Results](images/live_stream.png)


## Demo Video
[Demo Video](videos/Demo.mp4)


## Dashboard Preview:
![Dashboard](images/dashboard1.png)
![Dashboard](images/dashboard2.png)


## Installation & Setup:
1️⃣ Clone Repository \
```git clone https://github.com/AnooshaKhalid/inline-tablet-vision-hmi.git ``` \
``` cd inline-tablet-vision-hmi ``` \
2️⃣ Setup Environment \
```pip install -r requirements.txt ``` \
3️⃣ Run with Docker \
```docker-compose up --build``` \
4️⃣ Start System \
```python pi_inference.py```


## License
This project is developed for academic purposes (FYDP).

## Acknowledgment
Special thanks to faculty, mentors, and contributors who supported this project. \
Computer Systems Engineering - CIS \
NED University of Engineering & Technology