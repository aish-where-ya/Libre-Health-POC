from flask import Flask, render_template, url_for, request, redirect      
from keras.models import load_model
from keras.models import model_from_json
from matplotlib import image
from matplotlib import pyplot
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']=''

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        modelName = request.form.get("model")
        labelType = request.form.get("labeltype")
        if request.files:
            image = request.files["imgUpload"]
            #plt.savefig('/static/images/new_plot.png')
            #print (type(image))
        score, pred_label = prediction(modelName,image)
        #print(score, pred_label)
        return render_template('index.html', score=score, predicted_label=pred_label)   
    else:
    	print("bye")  
    	
    return render_template("index.html")
    
def model_load(modelName):
    if modelName == "resnet":
        modelWeights = os.path.join(os.getcwd(), 'model_assets', 'resnet50','weights_model_resnet.h5')
        modelJson = os.path.join(os.getcwd(), 'model_assets', 'resnet50','model_resnet.json')
    elif modelName == "densenet":
        modelWeights = os.path.join(os.getcwd(), 'model_assets', 'densenet121','weights_model_densenet.h5')
        modelJson = os.path.join(os.getcwd(), 'model_assets', 'densenet121','model_densenet.json')
    elif modelName == "vgg":
        modelWeights = os.path.join(os.getcwd(), 'model_assets', 'vgg19','weights_model_vgg.h5')
        modelJson = os.path.join(os.getcwd(), 'model_assets', 'vgg19','model_vgg.json')      
        
    with open(modelJson, 'r') as f:
        model = model_from_json(f.read())
            
    model.load_weights(modelWeights)
    
    return model    

def prediction(modelName,image):
    model = model_load(modelName)
    img = load_img(image)
    img_array = img_to_array(img)
    img_array.resize(224,224)
    img_array = img_array.reshape([-1,224,224,1]) 
    
    labels = np.array(['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','No Finding','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax'])
    pred = model.predict(img_array, steps=10)
    mean_pred = np.mean(pred,axis=0)
    score = np.amax(mean_pred)
    pred_label = labels[np.argmax(mean_pred)]
    return score, pred_label
    
if __name__ == "__main__":
    app.run(debug=True)
