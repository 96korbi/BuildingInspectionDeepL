import numpy as np  
import torch
import torch.nn as nn
import torchvision 
from torch.autograd import Variable
from torchvision import transforms, models
import PIL 
import cv2
#This is the Label
Labels = { 0 : 'Perfect',
           1 : 'Defected'
        }
# Let's preprocess the inputted frame
data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   ##Assigning the Device which will do the calculation
#vgg16  = torch.load("VGG16_v2-OCT_Building_half_dataset.pt") #Load vgg16 to CPU
# Load the pretrained vgg16 from pytorch
vgg16 = models.vgg16_bn(pretrained=True)
#vgg16.load_state_dict(torch.load("/home/user/Building-Inspection/vgg16-397923af.pth"))
print(vgg16.classifier[6].out_features) # 1000 


# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the vgg16 classifier
print(vgg16)
vgg16.load_state_dict(torch.load("VGG16_v2-OCT_Building_half_dataset.pt"))
vgg16  = vgg16.to(device)   #set where to run the vgg16 and matrix calculation
vgg16.eval()                #set the device to eval() mode for testing
#Set the Webcam 
def Webcam_720p():
    cap.set(3,1280)
    cap.set(4,720)
def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]
    return result,score
def preprocess(image):
    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    print(image)                             
    image = data_transforms(image)
    image = image.float()
    #image = Variable(image, requires_autograd=True)
    #image = image.cuda()
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 vgg16 seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor
    
    
#Let's start the real-time classification process!
                                  
cap = cv2.VideoCapture(0) #Set the webcam
Webcam_720p()
fps = 0
show_score = 0
show_res = 'Nothing'
sequence = 0
while True:
    ret, frame = cap.read() #Capture each frame
    
    
    if fps == 4:
        image        = frame[100:450,150:570]
        image_data   = preprocess(image)
        print(image_data)
        prediction   = vgg16(image_data)
        result,score = argmax(prediction)
        fps = 0
        show_res= result
        show_score = score
        
    fps += 1
    cv2.putText(frame, '%s' %(show_res),(950,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv2.putText(frame, '%s' %(show_score), (950,300), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.rectangle(frame,(400,150),(900,550), (250,0,0), 2)
    cv2.imshow("ASL SIGN DETECTER", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow("ASL SIGN DETECTER")