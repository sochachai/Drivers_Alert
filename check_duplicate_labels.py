import os
path = '/Users/weicongsu/PycharmProjects/Drivers_Alert/yolov5/data/labels' #path of labels
labels = os.listdir(path)
text_files_to_be_edited = {}
for x in labels:
    with open(os.path.join(path, x), 'rb') as f:
        lines = f.read().splitlines()
        for y in lines:
          try:
            if int(y[:2])>16:
                print('16'+y.decode('utf-8')[2:])
                #if your last label is 2 digit ie nc=17 [0-16] label  or change slice value
                print(x)      # it will print .txt file name
                text_files_to_be_edited[x] = '16'+y.decode('utf-8')[2:]
          except:
            pass

print(text_files_to_be_edited)
print(len(text_files_to_be_edited))


