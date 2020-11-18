from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing import image
from sklearn import preprocessing
from PIL import Image
import os


img_height, img_width=150,150

vgg_model=load_model('./models/vgg_model.h5')

svm_model =pickle.load(open('./models/svm_pickle', 'rb'))

labels = {0:"Acne",1:"Dark Spots",2:"Scars",3:"Wrinkles"}


def index(request):
    context={'a':1}
    return render(request,'index.html',context)



def predictImage(request):
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+ filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width,3))
    img = image.img_to_array(img)
    img = img/255.0
    input_img = np.expand_dims(img, axis=0) 
    input_img_feature=vgg_model.predict(input_img)
    input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
    prediction_RF = svm_model.predict(input_img_features)[0]
    for key in labels.keys():
        if key==prediction_RF:
            prediction_RF= labels[key]
    context={'filePathName':filePathName,'predictedLabel':prediction_RF}
    return render(request,'index.html',context) 

def viewDataBase(request):
    
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 