import cv2
import torch
import numpy as np
import time
import subprocess

# Load the model
model = torch.hub.load('.', 'custom', path='best.pt', source='local')
model.conf = 0.55  # Minimum detection threshold

# Camera setup
camera = cv2.VideoCapture(1)

def move_normal():
    print("move_static")
    subprocess.run(['python', 'move_static.py']) 

def move_out():
    print("move_out")
    subprocess.run(['python', 'move_out.py']) 

def move_in():
    print("move_in")
    subprocess.run(['python', 'move_in.py']) 

def judge_pet(results, objs):
    detected_classes = results.xyxy[0].cpu()[:, -1].numpy()
    return any(model.names[int(cls)] in objs for cls in detected_classes)

print("now active and running")
while True:
    move_normal()

    # Capture image
    ret, imgs = camera.read()

    # Detection
    results = model(imgs)

    # Decision
    if judge_pet(results, ['pet']):
        time.sleep(1)
        ret, imgs = camera.read()
        results = model(imgs)

        if judge_pet(results, ['cap', 'label']):
            move_out()
        else:
            move_in()

        time.sleep(0.5)
        move_normal()

    # Image display and other processing are omitted...

    # Parameters for hit area
    # pos_x = 240
    # Display detection results on the image
    for detection in results.xyxy[0]:  # Each detection
        # Unpack all values
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])

        # Format the label
        label = f'{model.names[int(cls)]} {conf:.2f}'

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
