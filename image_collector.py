import ssl
ssl._create_default_https_context = ssl._create_unverified_context # to solve yolov5s loading error
import uuid   # Unique identifier
import os
import time
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


'''
# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
results = model(imgs)
results.print()

#%matplotlib inline

plt.imshow(np.squeeze(results.render()))
#plt.imshow(results.render())
plt.show(block = True)
plt.interactive(False)

print(results.render())
#classification = np.argmax(results, axis=1)
print(results.pandas().xyxy[0])
'''
IMAGES_PATH = os.path.join('yolov5/data', 'images/validation') #/data/images
#print(IMAGES_PATH)
labels = ['straight', 'away']
number_imgs = 10

cap = cv2.VideoCapture(0)
# Loop through labels
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)

    # Loop through image range
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))

        # Webcam feed
        ret, frame = cap.read()

        # Naming out image path
        imgname = os.path.join(IMAGES_PATH, label + '.' + str(uuid.uuid1()) + '.jpg')

        # Writes out image to file
        cv2.imwrite(imgname, frame)

        # Render to the screen
        cv2.imshow('Image Collection', frame)

        # second delay between captures
        time.sleep(5)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

