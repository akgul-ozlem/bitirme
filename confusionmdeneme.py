#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:46:26 2022

@author: ozlem
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
 

#%% Path to the dataset
path = '/home/ozlem/Downloads/tomato_data'

train_path = '/home/ozlem/Downloads/tomato_data_processed/train'

valid_path = '/home/ozlem/Downloads/tomato_data_processed/valid'

test_path = '/home/ozlem/Downloads/tomato_data/test'

#%%Parameters

BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9 
NO_EPOCHS = 10
HIDDEN_SIZE_1 =200
HIDDEN_SIZE_2 = 100
POOLING = 10


#%% Dataset Exploration
diseases = os.listdir(train_path)
print(diseases)
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

#%%

all_preds = torch.tensor([])
for data,label in Train_loader:
    all_preds = torch.cat((all_preds,label),dim = 0)
    print(label)
    print(all_preds.shape)
    
#%%


# Cij truth i , obs in j 
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cm = confusion_matrix(y_true, y_pred)
print(cm)
    
'''  
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')   
''' 
#%%
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt



df_cm = pd.DataFrame(cm, range(3), range(3))
# plt.figure(figsize=(10,7))
sn.set(font_scale=0.8) # for label size
ax = sn.heatmap(df_cm/ np.sum(df_cm), annot=True, annot_kws={"size": 10}, fmt ='.2%' ,cmap='Blues') # font size
ax.set_xlabel('\n Predicted disease label')
ax.set_ylabel('\n Actual label')
ax.xaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])
ax.yaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])
plt.show()
    
    
    
    
    
    
    
    
    