import queue
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from inspect import currentframe, getframeinfo
from pathlib import Path
import threading
import mask

filename = getframeinfo(currentframe()).filename
parent_src= Path(filename).resolve().parent.parent

out = "Computing"

vgg16 = models.vgg16_bn(pretrained=True)
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 2)]) # Add our layer with 2 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the vgg16 classifier
pretrained_model_path = parent_src / "Model" / "VGG16_v2-OCT_Building_half_dataset.pt"
vgg16.load_state_dict(torch.load(pretrained_model_path))

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

def taskPredict(q):  #Evaluate the image as a task to enable multithreading
    global out
    task = q.get()
    out = (pre_image_cam(task, vgg16))
    q.task_done()


countCW = 0
countUW=0


def testPrediction():
    parent = parent_src.parent
    directory = parent / "DATA_Maguire_20180517_ALL" / "W" / "TEST" / "UW"
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
    
    # Create multiple threads which will pick the images from the queue
    # This is neccessary so the video doesn't lag while validating the frame
    for i in range(20):
        worker = threading.Thread(target=taskPredict, args=(q,), daemon=True)
        worker.start()

    # Read and resize image
    ret, image = cam.read()
    frameCount += 1

    # Every 10th frame is put in queue which is evaluted in taskPredict
    if (frameCount == 10):
        q.put(image)
        frameCount = 0
    
    # Mask overlay to mark the crack
    if out == "Crack":  
        image = cv2.resize(image, (640, 480))
        # Applying Gaussian Blur on input image
        image_blurred = cv2.GaussianBlur(image, (25, 25),0)
        image_sobel = mask.sobel(image_blurred)
        # Change image from BGR to Gray + BGR to Gray
        imageGRAY = cv2.cvtColor(image_sobel, cv2.COLOR_BGR2GRAY)
        imageGRAY = cv2.GaussianBlur(imageGRAY, (7,7),0)
        # Change pixels greater than 0,3*max image pixel to 255
        imageGRAY[imageGRAY > 0.3*np.max(imageGRAY)] = 255
        # Change pixels values to 0 if is < 126 otherwise to 255
        _, imageGRAY = cv2.threshold(imageGRAY, 126, 255, cv2.THRESH_BINARY)
        # Merge pixels using opencv closing morphology functions
        image_conn = mask.merger(imageGRAY)
        # Normalize image
        image_norm = cv2.normalize(image_conn, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # Finding contours on binary image
        cont, hierarchy = cv2.findContours(image_norm.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Approximation contours by mean of 10 points before and 10 points after
        new_cont = mask.approx_cont(cont, look_back=10)
        # Draw raw contours on input image
        image_cont = cv2.drawContours(image.copy(), cont ,-1, (0,0,255), 2)
        # Draw approximated contours on input image
        image_cont_approx = cv2.drawContours(image.copy(), new_cont ,-1, (0,0,255), 2)
        # Draw mask on image
        image_cont_approx_mask = cv2.drawContours(image.copy(), new_cont ,-1, (0,0,255), -1)
        image_final = cv2.addWeighted(image_cont_approx_mask, 0.5, image, 1 - 0.5, 0, image)
    
    # No overlay if theres no crack
    else:
        image_final = image

    # Show the video stream and write the output of the validation in the image
    cv2.putText(image_final, '%s' %(out),(10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Crack Detection", image_final)
    
    # Ending the program
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cam.release()
        cv2.destroyAllWindows()
        exit() # Exit kills every thread running