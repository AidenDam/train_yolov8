# custom dataset download from roboflow

import glob
import random
from PIL import Image, ImageDraw
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

model = YOLO('yolov8n.pt')

dir = 'datasets/{0}/'

for it in ('train', 'valid', 'test'):
    tmp_dir = dir.format(it)
    
    paths = glob.glob(tmp_dir + 'images/*.jpg')
    
    path = random.choice(paths)

    for path in paths:
        res = model(path)[0].boxes
        annotation = [' '.join(str(i) for i in ([cls] + box)) +  '\n' for cls, box in zip(res.cls.int().tolist(), res.xywhn.tolist())]
        
        path = path.replace('images', 'labels')
        path = path[:-3] + 'txt'
        with open(path) as f:
            data = f.readlines()
            data = ['80' + st[1:] for st in data]
        data += annotation

        with open('test.txt', 'w') as f:
            f.writelines(data)

    # check format dataset
    # res = model(path, conf=0.5)[0].boxes
    # annotation = [' '.join(str(i) for i in ([cls] + box)) +  '\n' for cls, box in zip(res.cls.int().tolist(), res.xywhn.tolist())]

    # label_path = path.replace('images', 'labels')
    # label_path = label_path[:-3] + 'txt'
    # with open(label_path) as f:
    #     data = f.readlines()
    #     data = ['80' + st[1:] for st in data]
    # data += annotation
    # data = np.array([list(map(float, i.split())) for i in data])
    # data[..., 1:] = xywhn2xyxy(data[..., 1:]).astype('int')
    # print(data)
    # with Image.open(path) as im:
    #     draw = ImageDraw.Draw(im)
    #     for bbox in data:
    #         draw.rectangle(tuple(bbox[1:]), outline=(225, 0, 0), width=2)
    #         draw.text(bbox[1:3], str(int(bbox[0])), fill=(255, 255, 255, 128)) 
    #     im.show()

    # break