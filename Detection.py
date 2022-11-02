import asyncio
import multiprocessing
import queue
import cv2
import numpy as np
import torchvision
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import threading

out = "Computing"

vgg16 = models.vgg16_bn(pretrained=True)
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the vgg16 classifier
vgg16.load_state_dict(torch.load("/home/user/Building-Inspection/VGG16_v2-OCT_Building_half_dataset.pt"))

vgg16.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   ##Assigning the Device which will do the calculation

def pre_image(image_path,model):
   img = Image.open(image_path)
   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]
   transform_norm = transforms.Compose([transforms.ToTensor(), 
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()  
      output =model(img_normalized)
     # print(output)
      index = output.data.cpu().numpy().argmax()
      classes = ["Crack", "No Crack"]
      class_name = classes[index]
      return class_name

def pre_image_cam(img,model):
   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]
   transform_norm = transforms.Compose([transforms.ToTensor(), 
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()  
      output =model(img_normalized)
     # print(output)
      index = output.data.cpu().numpy().argmax()
      classes = ["Crack", "No Crack"]
      class_name = classes[index]
      return class_name

def taskPredict(q):
    global out
    task = q.get()
    out = (pre_image_cam(task, vgg16))
    q.task_done()


#predict_class = pre_image("/home/user/Building-Inspection/Photos_NYP/Retaining-Wall-Crack.jpg",vgg16)
#print(predict_class)

countCW = 0
countUW=0


def testPrediction():
    directory="/home/user/Building-Inspection/DATA_Maguire_20180517_ALL/W/TEST/UW"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if pre_image(f, vgg16) == "Crack":
                countCW += 1
            else:
                countUW += 1

    print("Crack = " + str(countCW))
    print("\nNo Crack = " + str(countUW))

def testTask():
    print("Test")


frameCount = 0

cam = cv2.VideoCapture(0)
q= queue.Queue(maxsize=0)


while True:
    for i in range(20):
        worker = threading.Thread(target=taskPredict, args=(q,), daemon=True)
        worker.start()
    countCrack = 0
    countNoCrack = 0
    output = None

    # Read and resize image
    ret, frame = cam.read()
    frameCount += 1
    if (frameCount == 10):
        q.put(frame)
        frameCount = 0
    cv2.putText(frame, '%s' %(out),(10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Crack Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cam.release()
        cv2.destroyAllWindows()
        exit()