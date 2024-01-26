import cv2
import torch
import numpy as np
import time

# モデルの読み込み
model = torch.hub.load('.', 'custom', path='best.pt', source='local')
model.conf = 0.55  # 検出の下限値
# model.classes = [0]  # 例: 人間だけを検出する場合

# カメラ設定
camera = cv2.VideoCapture(1)

def move_normal():
    print("move_static.py")
    # subprocess.run(['python', 'move_static.py'])

def move_out():
    print("move_out.py")
    # subprocess.run(['python', 'move_out.py'])

def move_in():
    print("move_in.py")
    # subprocess.run(['python', 'move_in.py'])

def judge_pet(results, objs):
    detected_classes = results.xyxy[0].cpu()[:, -1].numpy()
    return any(model.names[int(cls)] in objs for cls in detected_classes)

while True:
    move_normal()

    # 画像取得
    ret, imgs = camera.read()

    # 検出
    results = model(imgs)

    # 判断
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

    # 画像表示などの処理は省略...

    # ヒットエリアのためのパラメータ
    # pos_x = 240
    # 検出結果を画像に描画して表示
    for detection in results.xyxy[0]:  # Each detection
        # Unpack all values
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])

        # Format the label
        label = f'{model.names[int(cls)]} {conf:.2f}'

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
