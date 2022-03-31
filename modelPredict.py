import numpy as np
import torch
import torchvision
import os
import cv2



path = '/home/pi/Desktop/bitirme-main/parameters'
pathImage = '/home/pi/Desktop/resim-main'

image = cv2.imread(pathImage)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image*255
image = image.astype(np.uint8)

HIDDEN_SIZE_1 =200
HIDDEN_SIZE_2 = 100
POOLING = 10

class myNet(nn.Module):      #LeNet, Common net

    def __init__(self):
        super(myNet, self).__init__()                               #[64,3,256,256]              
                            
        self.conv1 = nn.Conv2d(3, 64, kernel_size= 5)              #[BS,64,14,18]
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size= 5)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.LazyLinear( HIDDEN_SIZE_1)
        self.bn3 = nn.BatchNorm1d(HIDDEN_SIZE_1)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.bn4 = nn.BatchNorm1d(HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, len(diseases) )
        self.bn5 = nn.BatchNorm1d(len(diseases))

    def forward(self, input):
        x = F.max_pool2d(F.relu((self.bn1(self.conv1(input)))), 4)       #(bs,64,9,9)
        #x = F.max_pool2d(F.relu((self.conv1(input))), 4)       #(bs,64,9,9)
        
        
        #print('First x shape {}'.format(x.shape ))
        x = F.max_pool2d(F.relu((self.conv2_drop(self.bn2(self.conv2(x))))),4)
        #x = F.max_pool2d(F.relu((self.conv2_drop(self.conv2(x)))),4)
        
        
        #print('Second x shape {}'.format(x.shape ))
        x = torch.flatten(x,start_dim = 1)
        
        #x = x.view(x.size(0), -1)
        #print('Third x shape {}'.format(x.shape ))
        x = F.relu(self.bn3(self.fc1(x)))   
        #print('Fourth x shape {}'.format(x.shape ))
        x = F.relu(self.bn4(self.fc2(x)))  
        #print('Fifth x shape {}'.format(x.shape ))
        output = F.softmax(self.bn5(self.fc3(x)),dim = 1)   

        return output

model = myNet()

model.load_state_dict(torch.load(path))
output = model.forward(image)
_, prediction = torch.max(output, dim=1)
print('Prediction: ', prediction)









