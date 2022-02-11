import numpy as np
import pandas as pd

class OfficeLabelConverter():
    def __init__(self):
        self.label_name_list = np.array(['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket',
            'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch',
            'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet',
            'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet',
            'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor',
            'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil',
            'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
            'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker',
            'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'])

    def label_encoder(self, label):
        label_encoding = []
        for name in label:
            label_encoding.append(np.where(self.label_name_list==name)[0][0])
        return np.array(label_encoding)

    def label_decoder(self, label):
        label_decoding = []
        for numerical_label in label:
            label_decoding.append(self.label_name_list[numerical_label])
        return np.array(label_decoding)