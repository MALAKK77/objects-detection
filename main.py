import cv2
import numpy as np

thres = 0.45  
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        for i in indices:
            if isinstance(i, (list, tuple, np.ndarray)):
                i = i[0]  # unpack index if needed

            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            class_id = int(classIds[i])  # make sure it's an integer
            label = classNames[class_id - 1].upper()

            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(img, label, (x + 10, y + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"{round(confs[i] * 100, 2)}%",
                        (x + 200, y + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
