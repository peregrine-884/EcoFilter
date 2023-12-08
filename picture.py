import cv2
import datetime  # Import the datetime module for timestamp

# カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(1)

# 繰り返しのためのwhile文
while True:
    # カメラからの画像取得
    ret, frame = cap.read()

    # カメラの画像の出力
    cv2.imshow('camera', frame)

    # 繰り返し分から抜けるためのif文
    key = cv2.waitKey(10)
    if key == ord('e'):  # 'e' キーが押されたらプログラムを終了する
        break
    elif key == ord('s'):  # 's' キーが押されたら画像を保存
        # 一意のファイル名を生成するために現在の日時を使用
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'image/captured_image_{timestamp}.jpg'

        # JPEG形式で画像保存
        cv2.imwrite(filename, frame)
        print(f"Captured image saved as '{filename}'.")

# メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()
