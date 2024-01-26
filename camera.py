import cv2
import torch
import numpy as np
import subprocess
import time

# 検出する際のモデルを読込
model = torch.hub.load('.', 'custom', path='best.pt', source='local')  # localのyolov5sを使用

# 検出の設定
model.conf = 0.4  # 検出の下限値。設定しなければすべて検出
# model.classes = [0]  # 0:person クラスだけ検出する。設定しなければすべて検出

# 映像の読込元指定
camera = cv2.VideoCapture(1)  # カメラ：Ch.(ここでは0)を指定


def move_normal():
    print("move_static.py")
    # p = subprocess.run(['python', 'move_static.py'])

def move_out():
    print("move_out.py")
    # p = subprocess.run(['python', 'move_out.py'])

def move_in():
    print("move_in.py")
    # p = subprocess.run(['python', 'move_in.py'])


def judge(results):
    def judge_pet(results, obj):
        classes_detected = results.xyxy[0].cpu()[:, -1].numpy()
        unique_classes, counts = np.unique(classes_detected, return_counts=True)
        for cls_index, count in zip(unique_classes, counts):
            if model.names[int(cls_index)] == obj and count > 0:
                return True
        return False


    if (judge_pet(results, 'pet')):
        time.sleep(10)

        ret, imgs = camera.read()

        results = model(imgs, size=160)

        if (judge_pet(results, 'cap') or judge_pet(results, 'label')):
            move_out()

        else:
            move_in()

        time.sleep(2)
        move_normal()


# ヒットエリアのためのパラメータ
# pos_x = 240

while True:
    move_normal()

    # 画像の取得
    ret, imgs = camera.read()  # 映像から１フレームを画像として取得

    # 推定の検出結果を取得
    results = model(imgs, size=160)  # 160ピクセルの画像にして処理

    # 検出結果を画像に描画して表示
    for detection in results.xyxy[0]:  # Each detection
        # Unpack all values
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])

        # Format the label
        label = f'{model.names[int(cls)]} {conf:.2f}'

        # 枠描画
        cv2.rectangle(imgs, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # 文字枠と文字列描画
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(imgs, (x1, y1 - 20), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(imgs, label, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        judge(results)

    # ヒットエリアのラインを描画
    # cv2.line(imgs, (pos_x, 0), (pos_x, imgs.shape[0]), (128, 128, 128), 3)

    # 描画した画像を表示
    cv2.imshow('YOLOv5 Detection', imgs)

    # 「q」キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
