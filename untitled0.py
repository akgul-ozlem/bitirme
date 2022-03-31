#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:38:44 2022

@author: ozlem
"""



#%% Necessary modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader  
import torchvision.transforms as transforms 
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torch.optim as optim
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score,  f1_score, classification_report
from sklearn.metrics import confusion_matrix


#%% Path to the dataset
path = '/home/ozlem/Downloads/tomato_data'

train_path = '/home/ozlem/Downloads/tomato_data_processed/train'

valid_path = '/home/ozlem/Downloads/tomato_data_processed/valid'

test_path = '/home/ozlem/Downloads/tomato_data/test'

#%%Parameters

BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9 
NO_EPOCHS = 1
HIDDEN_SIZE_1 =200
HIDDEN_SIZE_2 = 100
POOLING = 10


#%% Dataset Exploration
diseases = os.listdir(train_path)
#print(diseases)
#print(type(diseases))
#print("Total disease classes are: {}".format(len(diseases)))

#Exploring the dataset 
Train = ImageFolder(train_path, transform=transforms.ToTensor())
Valid = ImageFolder(valid_path, transform=transforms.ToTensor())



'''
#Two example images       
image1, label1 = Train[0]                   # Reaching th eimage in the dataset and its assigned label
image1 = image1.permute(1,2,0)              #Permute the entries to obtain an image of the form [256,256,3]
plt.imshow(image1)
plt.show()

print(image1.size(), Train.classes[label1])

image2, label2 = Train[3476]                   # Reaching th eimage in the dataset and its assigned label
image2 = image2.permute(1,2,0)              #Permute the entries to obtain an image of the form [256,256,3]
plt.imshow(image2)
plt.show()
print(image2.size(), Train.classes[label2])

#print(Train.classes[label])               #Learn the disease associated with the label number
#print(Train.__len__())                    #18345 total length of image data 
print(Train.class_to_idx)                 #the dictionary of diseases and their respective label numbers
#help(Train)                               #Explore the imagefolder class 
'''
print(Train.class_to_idx) 



#%% Cuda

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print('Device name:', torch.cuda.get_device_name(0))
else: 
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

torch.cuda.empty_cache()
print(device)

#%% Datalaoders

Train_loader = DataLoader(Train,
                          batch_size = BATCH_SIZE,
                          shuffle=True)
Valid_loader = DataLoader(Valid,
                          batch_size = BATCH_SIZE,
                          shuffle=True)

#%% Model

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
 



'''
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
        #print('First x shape {}'.format(x.shape ))
        x = F.max_pool2d(F.relu((self.conv2_drop(self.bn2(self.conv2(x))))),4)
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

'''





'''
class myNet(nn.Module):
    def __init__(self):
        super(myNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,stride = 1)
        self.norm1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.hidden = nn.LazyLinear(len(diseases), bias = True)


    def forward(self,x):
        x = F.relu(self.norm1(self.conv1(x)))
        
        x = torch.flatten(x,start_dim =1)
        
        output =F.softmax( self.hidden(x),dim = 1)
        
        return output
''' 


   
#%%

#optimizer= optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM)
   
is_last_epoch = False


def train(model,train_loader,test_loader,epochs,optimizer):
    for epoch in range(epochs):
        
        truth = torch.tensor ([]) 
        all_preds = torch.tensor ([])   
        
        
        for data,label in train_loader:
            
            
            #print('before train: ', model.training())
            #for param in model.parameters():
            #    print(param.requires_grad)# = True
            
            #print('after param requires grad:', model.training())
            
            model.train()
            
            #print('after model.train:', model.training)
            output = model.forward(data)
            #print('outğuts:', output)
            #print('label:', label)
            loss = F.cross_entropy(output,label)
            loss.backward()
            
            train_loss = loss.item()
            
            
            optimizer.step()
            optimizer.zero_grad()
            
            if epoch == epochs-1:
                is_last_epoch = True
            
            
            
            accuracy, val_loss, all_preds, truth = eval(model,test_loader,all_preds, truth , is_last_epoch)
            print(f"Epoch: {epoch+1}/{epochs}..", f"Training loss: {train_loss:.3f}", f"Validation loss: {val_loss:.3f} " , f"Validation Accuracy: {accuracy:.3f}")
        
            
        
        cm = confusion_matrix(truth, all_preds)
        df_cm = pd.DataFrame(cm, range(10), range(10))
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=0.8) # for label siz,
        ax= sn.heatmap(df_cm/np.sum(df_cm), annot=True, annot_kws={"size": 6}, fmt ='.2%' , cmap= 'Blues') # font size
        
        ax.set_xlabel('\n Predicted disease label')
        ax.set_ylabel('\n Actual label')

        ax.xaxis.set_ticklabels(['Bacterial Spot','Early Blight', 'Late Blight','Leaf Mold','Septoria Leaf Spot','Spider Mites','Target Spot','Leaf Curl Virus','Mosaic Virus','Healthy'])
        ax.yaxis.set_ticklabels(['Bacterial Spot','Early Blight', 'Late Blight','Leaf Mold','Septoria Leaf Spot','Spider Mites','Target Spot','Leaf Curl Virus','Mosaic Virus','Healthy'])

        plt.show()
    
    
    
def eval(model,test_loader,all_preds,truth,is_last_epoch):
    with torch.no_grad():
        for data,label in test_loader:
            model.eval()
     
            output = model.forward(data)
            loss = F.cross_entropy(output,label)
            
            valid_loss = loss.item()
    
            targets = label.numpy()
            
            _, predictions = torch.max (output,dim = 1)
            
            if is_last_epoch == True:
                all_preds = torch.cat((all_preds,predictions),dim = 0)
                truth = torch.cat((truth,predictions),dim = 0)
            
            #print('predictions:', predictions)
            
            accuracy = accuracy_score(predictions,targets) 
            model.train()
            
            
            return accuracy, valid_loss,all_preds,truth

#%%
model = myNet()
optimizer= optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM)
train(model,Train_loader,Valid_loader,NO_EPOCHS,optimizer)

