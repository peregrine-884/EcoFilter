# Intelligent Trash Bin with AI Technology

## Introduction
 We have developed **an intelligent trash bin** equipped with AI technology. This trash bin automatically identifies whether a PET bottle is recyclable or not, and removes non-recyclable items. As depicted in the gif below, it accurately detects labeled or capped PET bottles.
![動作デモGIF](img/intro2.gif)

 This machine utilizes deep learning from YOLOv5 to detect PET bottles, labels, and caps. In this project, we created an original dataset consisting of various PET bottles with labels or caps, as well as those without such decorations. Additionally, we programmed the Jetson Nano to control a servo motor, allowing the machine to remove bottles with decorations.

## Table of contents
1. [Background](#background)
1. [Requirements](#requirements)
1. [Set up](#set-up) *(写真付きでハードウェアの作り方)*
2. [File details](#file-details)
3. [Running the application](#running-application)
4. [How it works](#how-it-works)
4. [How the Intelligent Trash Bin Ejects PET Bottles](#how-the-intelligent-trash-bin-ejects-pet-bottles)
5. [Data collection](#data-collection)
6. [Training yolov5](#training-yolov5)
7. [Future direction](#future-directions)

## Background
In waste management facilities, collected PET bottles undergo a sorting process where labels, caps, and contaminated bottles are separated to recycle clean PET bottles. This sorting is often performed manually, which is labor-intensive. By utilizing this smart trash bin, the workload at these facilities can be significantly reduced.

## Requirements
* Hardware
    * Jetson nano
    * Web camera
    * Servo motor
    * Trash box
    * Mini light
    * SD card 
* Software *(バージョンのチェックを実機でする)*
    * Python==3.6.8
	* Ubuntu20.04
    * torch
	* cuda
    * Yolov5
    * RPi.GPIO

## Set up
### Software
* First, you must **install Ubuntu 20.04 OS image** for Jetson Nano at URL:
[Jetson Nano with Ubuntu 20.04 OS image](https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image)

    1. Get a 32 GB (minimal) SD card to hold the image.
    2. Download the image JetsonNanoUb20_3b.img.xz (8.7 GByte!) from our [Sync](https://ln5.sync.com/dl/403a73c60/bqppm39m-mh4qippt-u5mhyyfi-nnma8c4t).
    3. Flash the image on the SD card with the [Imager](https://www.raspberrypi.org/software/) or [balenaEtcher](https://www.balena.io/etcher/). 
    4. According to [issue #17](https://github.com/Qengineering/Jetson-Nano-image/issues/17#) only flash the xz directly, not an unzipped img image.
    5. Insert the SD card in your Jetson Nano and enjoy.
    6. Password: jetson

* Next, you must **set up PWM control pin** in Jetson nano to operate a servo motor. On terminal, input next commands.<br>
References:<br>
[SPI on Jetson – Using Jetson-IO](https://jetsonhacks.com/2020/05/04/spi-on-jetson-using-jetson-io/)<br>
[JetPack 4.3 (r32.3.1) で追加された Jetson-IO tool を使用して Pinmux テーブルを設定してみた。](https://qiita.com/kitazaki/items/a445994f1f46a1b15f78)<br>
 
```
$ sudo /opt/nvidia/jetson-io/jetson-io.py
```
Next, we can see the screen below.<br>
![set up pin 1](https://jetsonhacks.com/wp-content/uploads/2020/05/JetsonIO-Main.png)

* Select Configure 40-pin expansion header<br>
![set up pin 2](img/set_up_pin2.avif)

* Select pwm0, pwm2. (push Shift button)<br>
![set up pin 3](img/set_up_pin3.avif)

* Last, select Save and reboot reconfigure pins<br>
You can check that setting up successes with the command `$ ls -l /boot/*.dtb` and weather new .dtb file is created or not.


If you can’t open jetson-io.py, you should try some fix ways. Especially, we fixed this trouble with the below commands.<br>

Reference:[Jetson Nano の GPIO にサーボモータをつないで制御してみる](https://wisteriahill.sakura.ne.jp/CMS/WordPress/2020/12/07/jetson-nano-gpio-servo-motor/)
```
cd /boot
sudo mkdir dtb
sudo cp *.dtb* dtb/
```
Then you can try again first command.<br>
* Last you **[clone our GitHub](https://github.com/hayato-hayashi/experiment-3)**
```
$ git clone https://github.com/hayato-hayashi/experiment-3.git
$ pip install -r requirements.txt
```

### Hardware
In constructing this machine, we covered a trash box with cardboard on. Then, we installed a **camera**, **servo motor**, **mini light**, and **Jetson Nano 2GB** at several positions as demonstrated in the images below.<br>

Cut the plastic sheet into several rectangular sizes. Then, use instant glue to fix them into a cylindrical shape.<br>
As shown in the following image, make a hole in the cardboard and insert the cylinder, securing it with tape at the **top of the hole**.<br>
*(筒の写真)　下敷きの切り分けるサイズを指定*<br>
To prevent light from entering the inside of the cylinder, attach a board made of cardboard or similar material, as shown in the next image.<br>
*[hardware 3](img/hardware3.png)*<br>
Adjust the position of the camera and the cylinder so that the entire cylinder fits within the camera's field of view.<br>
*[hardware 4](img/hardware4.png)*<br>
We attached a wood bar approximately ~ cm long to the servo motor. Adjust **the lengths of the cylinder's top and bottom surfaces**, as well as **the length of the wooden rod**, to ensure it can sufficiently push up and eject a 500ml PET bottle.<br>

As shown in the next image, wooden and metal parts are attached to the trash bin. We cut the wooden boards to a certain size.<br>
*(上側の段ボール、蓋を取り外した状態の画像)　木の板のサイズ（金属棒のサイズ）指定*<br>
The top half of the trash bin is modified to be easily removable, as shown in the following image. Screws are inserted into the lid of the trash bin, and grooves are made in the body of the bin to accommodate the screws. This allows the top of the trash bin to be lifted relatively easily to open it.<br>
*(蓋に刺したねじや本体に作った溝の様子の画像)*<br>
Holes are drilled at specific locations on the body of the trash bin. The power cord of the Jetson Nano 2GB is passed through these holes to connect to the power supply.<br>
*(本体に開けた穴の画像)*<br>


## File details
```
experiment-3
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── README.zh-CN.md
├── benchmarks.py
├── camera.py
├── createModel.py
├── detect.py
├── export.py
├── hubconf.py
├── jetsoncam.py
├── label_Inflated_water.py
├── models
│   ├── __init__.py
│   ├── common.py
│   ├── experimental.py
│   ├── hub
│   │   ├── anchors.yaml
│   │   ├── yolov3-spp.yaml
│   │   ├── yolov3-tiny.yaml
│   │   ├── yolov3.yaml
│   │   ├── yolov5-bifpn.yaml
│   │   ├── yolov5-fpn.yaml
│   │   ├── yolov5-p2.yaml
│   │   ├── yolov5-p34.yaml
│   │   ├── yolov5-p6.yaml
│   │   ├── yolov5-p7.yaml
│   │   ├── yolov5-panet.yaml
│   │   ├── yolov5l6.yaml
│   │   ├── yolov5m6.yaml
│   │   ├── yolov5n6.yaml
│   │   ├── yolov5s-LeakyReLU.yaml
│   │   ├── yolov5s-ghost.yaml
│   │   ├── yolov5s-transformer.yaml
│   │   ├── yolov5s6.yaml
│   │   └── yolov5x6.yaml
│   ├── segment
│   │   ├── yolov5l-seg.yaml
│   │   ├── yolov5m-seg.yaml
│   │   ├── yolov5n-seg.yaml
│   │   ├── yolov5s-seg.yaml
│   │   └── yolov5x-seg.yaml
│   ├── tf.py
│   ├── yolo.py
│   ├── yolov5l.yaml
│   ├── yolov5m.yaml
│   ├── yolov5n.yaml
│   ├── yolov5s.yaml
│   └── yolov5x.yaml
├── move_in.py
├── move_out.py
├── move_static.py
├── picture.py
├── recode.py
├── requirements.txt
├── setup.cfg
├── train.py
├── tutorial.ipynb
├── utils
│   ├── __init__.py
│   ├── activations.py
│   ├── augmentations.py
│   ├── autoanchor.py
│   ├── autobatch.py
│   ├── aws
│   │   ├── __init__.py
│   │   ├── mime.sh
│   │   ├── resume.py
│   │   └── userdata.sh
│   ├── callbacks.py
│   ├── dataloaders.py
│   ├── docker
│   │   ├── Dockerfile
│   │   ├── Dockerfile-arm64
│   │   └── Dockerfile-cpu
│   ├── downloads.py
│   ├── flask_rest_api
│   │   ├── README.md
│   │   ├── example_request.py
│   │   └── restapi.py
│   ├── general.py
│   ├── google_app_engine
│   │   ├── Dockerfile
│   │   ├── additional_requirements.txt
│   │   └── app.yaml
│   ├── loggers
│   │   ├── __init__.py
│   │   ├── clearml
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── clearml_utils.py
│   │   │   └── hpo.py
│   │   ├── comet
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── comet_utils.py
│   │   │   └── hpo.py
│   │   └── wandb
│   │       ├── __init__.py
│   │       └── wandb_utils.py
│   ├── loss.py
│   ├── metrics.py
│   ├── plots.py
│   ├── segment
│   │   ├── __init__.py
│   │   ├── augmentations.py
│   │   ├── dataloaders.py
│   │   ├── general.py
│   │   ├── loss.py
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── torch_utils.py
│   └── triton.py
└── val.py
```

## Running application
After setting up about software and hardware preparation, Input below command and wait a several minutes.
```
$ python3 camera.py
```

After this, some logs output at terminal and be written ~~“start ”~~準備が完了したログをターミナルに表示させたい。
ターミナルの画像を貼り付けたい. Then you can use our intelligent trash box. Try to insert some plastic bottles. Please note that you shouldn’t insert a new bottle while this trash box is processing a bottle you entered. If all goes well, you should see the plastic bottle putting in the box or putting out ~~from the box~~画像を張り付けておいたが、gifのほうがいいかもしれない. 

![demo 1](img/demo1.png)

## How it Works

This application leverages YOLOv5 on the Jetson Nano 2GB, optimizing for rapid object detection within the constraints of limited memory. The primary focus is on processing video feed from a camera to detect three key objects: plastic bottles, their labels, and caps. Based on the detection of these items, the system determines whether a plastic bottle is recyclable. If deemed recyclable, the bottle is accepted by the trash bin; otherwise, a motor mechanism ejects it from the bin.

*(Diagram or photograph illustrating the detection)*

Detailed Process:
1. **Initialization**: The Jetson Nano and application are started, ready to process incoming video feed.
2. **Bottle Detection**: As a plastic bottle is introduced into the trash bin's entry point, the application, powered by YOLOv5 on the Jetson Nano, detects the bottle along with its label and cap. 
3. **Recyclability Assessment**: The application evaluates whether the bottle can be recycled based on the presence of a label and cap.
    * If the bottle lacks a label and cap, it is deemed recyclable and accepted by the trash bin.
    * If a label or cap is detected, the system rejects the bottle, activating a motor to eject it from the bin.
4. **Motor Mechanism**: A servo motor manipulates a panel to manage the bottle's fate:
    * The panel swings back to accept recyclable bottles.
    * It pushes forward to eject non-recyclable bottles outside the bin.

This smart recycling system not only simplifies waste management but also promotes environmental sustainability by ensuring proper segregation of recyclable materials.

## How the Intelligent Trash Bin Ejects PET Bottles
One of the key challenges in designing an automated system for waste management is ensuring that non-recyclable items can be efficiently separated and ejected from the bin. In our Intelligent Trash Bin project, we've implemented a couple of ingenious solutions to address this challenge effectively.

### Innovative Ejection Mechanism
Our design incorporates a specialized ejection mechanism that leverages a combination of mechanical ingenuity and precise control. Here are the main innovations we've made:

* **Adjustable Plastic Cylinder**: The bin's intake is equipped with a plastic cylinder that acts as a conduit for PET bottles. We meticulously adjusted the lengths of the upper and lower surfaces of this cylinder to optimize the ejection process. This adjustment ensures that a wooden rod, swung by a servo motor, can effectively push up and eject the PET bottle from the cylinder.

* **Strategic Intake Placement**: The intake is created by cutting a portion of the cardboard that encases the trash bin and inserting the adjusted plastic cylinder. A crucial innovation here is the placement of the intake with tape on the upper side rather than the lower side of the cut-out in the cardboard. This placement makes it easier for the intake to lift during the ejection process, facilitating smoother discharge of the PET bottles.

These design choices are central to the Intelligent Trash Bin's ability to differentiate and eject non-recyclable PET bottles. By fine-tuning the physical components and their interactions, we've achieved a system that not only automates waste segregation but does so with high efficiency and reliability.

## Data Collection
To accurately recognize plastic bottles, caps, and labels, we undertook a comprehensive data collection process. Our goal was to gather images that reflect the variety of ways a plastic bottle can appear when introduced into the trash bin. We created **a holding area** at the trash bin's entrance, made from transparent plastic sheets, to ensure bottles remained in place during image capture. ~~The camera was positioned to capture the entire holding area within its field of view. *(カメラの位置の詳しく解説をsetupで)*

Collection Procedure:
1. Preparing the Bottles: We started with a collection of approximately 150 plastic bottles in various conditions. Instead of preparing separate groups of bottles with and without labels and caps, we utilized the same set of bottles for multiple stages of data collection. Initially, we photographed each bottle in its current state, capturing images of bottles with labels and caps intact. Subsequently, we removed the caps from these bottles and took additional photographs. Finally, we removed both the labels and caps, capturing images of the bottles in a completely unadorned state. This method allowed us to create a diverse dataset from a fixed number of bottles, ensuring a wide range of conditions were represented in our training data.

2. Capturing Images: The photographic process was carefully designed to avoid detection inaccuracies due to the orientation of bottle insertion. For each stage of bottle preparation (with labels and caps, with caps removed, and with both removed), we captured images in two orientations: cap-first and cap-last. This approach ensured our model would learn to recognize bottles irrespective of how they were introduced into the bin. Each bottle's various states were documented from multiple angles to further enhance the model's accuracy and robustness in real-world scenarios.

3. Annotation Process: Each image was annotated to identify the entire bottle, label, and cap positions using the VoTT (Visual Object Tagging Tool). The annotations were initially saved in the Pascal VOC format (XML output), ~~which were later converted to the YOLOv5 format through web tools for compatibility~~.Pascal VOCからyolov5形式に変換する手順をもう少し詳しく

4. Image Augmentation: To enhance our dataset, we manipulated the brightness of images for augmentation, effectively increasing our dataset without needing to physically collect more samples. This process resulted in approximately 1000 annotated images ready for training.

5. Usage: The annotated data will be utilized for training the YOLOv5 model, enabling our application to correctly detect plastic bottles, labels, and caps. This is crucial for the automatic segregation of recyclable materials, improving recycling efficiency.

## Training YOLOv5
*(
足りない情報: 
    Google Colaboratoryで使用した具体的なコードやコマンドラインの例がない。実際の学習プロセスを理解するためには、実行したスクリプトやコマンドの詳細が有益です
    学習にかかった時間や、使用したGPUの種類など、学習環境に関する具体的な情報が不足しています。これらの情報は、プロジェクトの再現性や、読者が同様の学習環境を設定する際の参考になります。
)*

We conducted transfer learning using **YOLOv5** on our dataset. Our annotations were initially in the Pascal VOC XML format, which we converted to the YOLO format for training purposes. Unlike the typical approach using Docker environments for machine learning tasks, we utilized **Google Colaboratory** for our training process. This platform allowed us to leverage its powerful GPUs for training, significantly speeding up the process. Here's an overview of the steps we followed:

1. Annotation Conversion: Converted our dataset annotations from Pascal VOC's XML format to YOLO format to make them compatible with YOLOv5.
2. Training in Google Colaboratory:
    * Uploaded our YOLO-formatted dataset to Google Colab.
    * Ran our transfer learning script on YOLOv5 using the Colab notebook, specifying our dataset path.

This approach enabled efficient use of resources and streamlined our model training phase.

## Future directions
The current model has been trained using approximately a hundred PET bottles collected from various locations within Gifu University's campus, resulting in a prediction capability limited to the **preferences of Gifu University students**. In the future, I plan to enhance the prediction capability by incorporating data on plastic bottle preferences from individuals across different age groups, including the elderly and children.

