import cv2
import torch
import numpy as np

# プリセットのモデルをロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, source='local')

# best.pt ファイルから重みをロード
weights_path = 'best.pt'  # 重みファイルのパスを指定
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['model'].state_dict())

# 検出の設定
model.conf = 0.5  # 検出の下限値。設定しなければすべて検出
model.classes = [0]  # 0:person クラスだけ検出する。設定しなければすべて検出

# 映像の読込元指定
camera = cv2.VideoCapture(0)  # カメラ：Ch.(ここでは0)を指定

# ヒットエリアのためのパラメータ
pos_x = 240

while True:
    # 画像の取得
    ret, imgs = camera.read()  # 映像から１フレームを画像として取得

    # 推定の検出結果を取得
    results = model(imgs, size=160)  # 160ピクセルの画像にして処理

    # 検出結果を画像に描画して表示
    for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class
        label = f'{model.names[int(cls)]} {conf:.2f}'
        x1, y1, x2, y2 = map(int, box)

        # ヒットしたかどうかで枠色と文字色の指定
        cc = (255, 255, 0) if x1 > pos_x else (0, 255, 255)
        cc2 = (128, 0, 0) if x1 > pos_x else (0, 128, 128)

        # 枠描画
        cv2.rectangle(imgs, (x1, y1), (x2, y2), color=cc, thickness=2)

        # 文字枠と文字列描画
        cv2.rectangle(imgs, (x1, y1 - 20), (x1 + len(label) * 10, y1), cc, -1)
        cv2.putText(imgs, label, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, cc2, 1, cv2.LINE_AA)

    # 検出されたオブジェクトの数を表示
    classes_detected = results.xyxy[0][:, -1]  # 検出されたクラスのインデックス
    unique_classes, counts = np.unique(classes_detected, return_counts=True)
    for i, (cls_index, count) in enumerate(zip(unique_classes, counts)):
        cv2.putText(imgs, f'{model.names[int(cls_index)]}: {count}', (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ヒットエリアのラインを描画
    cv2.line(imgs, (pos_x, 0), (pos_x, imgs.shape[0]), (128, 128, 128), 3)

    # 描画した画像を表示
    cv2.imshow('YOLOv5 Detection', imgs)

    # 「q」キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
camera.release()
cv2.destroyAllWindows()
