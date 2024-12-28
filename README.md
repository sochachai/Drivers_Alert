# Drivers Alert System
This computer vision project aims to build a driver alert app for the sake of driving safety. <br>
It consists of two components: 1. drowsiness alert 2. distraction alert <br>

## Running the apps
Run "python driver_drowsiness_alert.py" for the drowsiness alert system <br>
Run "python driver_distraction_alert.py" for the distraction alert system <br>
Screenshots to showcase the functioning of the apps are stored in the folders "App_Screenshot/drowsiness" and "App_Screenshot/distraction" <br>
Sound alerts will be raised when drowsiness or distraction are detected by the python programs. For the drowsiness alert system, Chinese versions(Mandarin and Cantonese) of sound alerts are used. For the distraction alert system, languages are not specified for the sound alert and the default English version is used. <br>
I plan to merge the two alert system into one. This has not been done yet. <br>

## Development of the apps
### Drowsiness Alert 
1. "shape_predictor_68_face_landmarks.dat" is used for facial detection. This file can be downloaded from "https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat" <br>
2. Eye Aspect Ratio can be derived from the face landmarks described in the previous step and is used to determine the closing/drowsiness level of the eyes. Reference of Eye Aspect Ratio can be found in "https://arxiv.org/abs/2408.05836" <br>
3. A threshold of culmulative drowsiness level is set to prevent false alarm of drowsiness in case of blinks. <br>

### Distraction Alert
Model is trained with camera-shot images invoked by OpenCV and 200 epochs are run. The following describes the training process. <br>
0. "shape_predictor_68_face_landmarks.dat" cannot detect side faces. I choose to use YOLO for the purpose of distraction alert. <br>
1. git clone https://github.com/ultralytics/yolov5 (to the local Pycharm Project) <br>
2. git clone https://github.com/tzutalin/labelImg (to the local Pycharm Project) <br>
3. cd to the local directory "yolov5" and run "python image_collector.py" to store camera-captured driver's images of distraction(looking away) and normal(looking straight) into the directories "yolov5/data/images/train" and "yolov5/data/images/validation" <br>
4. cd to the local directory "labelImg" and run "python labelImg.py" to process the images acquired in Step 3 to yolo files and store them in local directories "yolov5/data/labels/train" and "yolov5/data/labels/validation" <br>
5. cd to the local directory "yolov5" and add the classes "away" and "straight" to the list "names" <br>
6. cd to the local directory "yolov5" and run "python train.py --img 320 --batch 16 --epochs 200 --data dataset.yml --weights yolov5s.pt --workers 0" <br>

### Results
The drowsiness alert system works pretty well (even with glasses on driver's face). The distraction alert system trained with YOLO also works as the app screenshots have shown but there is much room to improve. <br>
For the model training evaluation, please refer to the files in YOLO_model_training_evaluation for details. As the results.csv has shown, the validation accuracy improves over time as epoch number increases. <br>

### Notes
1. The local content of yolov5 and labelImg cannot be uploaded in this repository. Please refer to the screenshot "Project_Infrastructure" on the configuration of the local Python Project. <br>
2. Sometimes when mislabelling(ususally duplicated labeling) occured during the image labeling step described in Step 4 in the section of "Disctraction Alert", the distraction alert app fails to run.  In this case, cd to yolov5 and run "python check_duplicate_labeling.py" to see if there exists any duplicated labeling of the images. 





