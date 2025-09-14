# objects-detection
This project implements real-time object detection using Python and OpenCV.   It uses a pre-trained deep learning model (SSD MobileNet V3) that can recognize more than 80 everyday objects such as people, cars, animals, furniture, and electronic devices.    The program captures live video from your computer’s webcam, processes each frame, and detects objects by drawing bounding boxes and labels on the screen.   It also applies Non-Maximum Suppression (NMS) to improve accuracy by removing duplicate or overlapping detections.    The system is lightweight and runs in real-time on CPU, making it suitable for learning, experimentation, and small-scale projects in computer vision.

🚀Try it yourself

If you want to try this project:

1.Make sure you have Python 3 installed.

2.Install required libraries:(pip install opencv-python numpy
)

3.Put these files in the same folder:

-main.py

-coco.names

-ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt

-frozen_inference_graph.pb

4.Run the program:(python main.py)

5.The webcam will open and start detecting objects.

6.Press q to exit anytime.
