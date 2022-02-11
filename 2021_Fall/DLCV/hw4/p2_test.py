import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models
from p2_dataset import *
from p2_model import *
from office_label_converter import *
import random

# constants
BATCH_SIZE = 32

parser = argparse.ArgumentParser(description="Self-supervised Learning")
parser.add_argument('--setting', type=str, help='setting (A, B, C, D or E)')
parser.add_argument('--test_csv', type=str, help="Testing images csv file")
parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
parser.add_argument('--output_csv', type=str, help="Output filename")
args = parser.parse_args()
resnet_path = f"p2_2_{args.setting}_resnet.pth"
classifier_path = f"p2_2_{args.setting}_classifier.pth"

if __name__ == '__main__':
    
    # Create dataset
    test_set = OfficeHome(args.test_csv, args.test_data_dir)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    resnet = models.resnet50(pretrained=False)
    classifier = Classifier()
    resnet.load_state_dict(torch.load(resnet_path))
    classifier.load_state_dict(torch.load(classifier_path))
    
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    resnet = resnet.to(device)
    classifier = classifier.to(device)
    
    resnet.eval()
    classifier.eval()
    
    predictions = []
    filenames = []

    for i, (img, _, filename) in enumerate(test_loader):
        with torch.no_grad():
            class_logits = classifier(resnet(img.to(device)))
        predictions.extend(class_logits.argmax(dim=-1).cpu().numpy().tolist())
        filenames.extend(filename)
    
    converter = OfficeLabelConverter()
    labels = converter.label_decoder(predictions)
    
    with open(args.output_csv, "w") as f:
    
        # The first row must be "Id, Category"
        f.write("id,filename,label\n")
        
        # For the rest of the rows, each image id corresponds to a predicted class.
        for idx, (fn, pred) in  enumerate(zip(filenames, labels)):
            f.write(f"{idx},{fn},{pred}\n")
    
