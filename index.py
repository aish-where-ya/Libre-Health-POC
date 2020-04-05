from flask import Flask, render_template, url_for, request, redirect      
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        score, pred_label = prediction(modelName,image)
        resImage = image.filename.split('.')[0]+modelName+".png"
        return render_template('index.html', score=score, predicted_label=pred_label, resultImage=resImage)   
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
    elif modelName == "ssdMobilenet":
        modelWeights = os.path.join(os.getcwd(), 'model_assets', 'ssdMobilenet','weights_ssdMobilenet.h5')
        modelJson = os.path.join(os.getcwd(), 'model_assets', 'ssdMobilenet','model_ssdMobilenet.json')      
        
    with open(modelJson, 'r') as f:
        model = model_from_json(f.read())
            
    model.load_weights(modelWeights)
    
    return model    

def prediction(modelName,image):
    if modelName == "ssdMobilenet":
    	score, pred_label = prediction_ssd(modelName,image)
    else :
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
    	fig,ax = plt.subplots()
    	ax.imshow(img)
    	fig.savefig(os.path.join(os.getcwd(),'static',image.filename.split('.')[0]+modelName+".png"))
    return score, pred_label
    
def prediction_ssd(modelName,image):
    model = model_load("vgg")
    score, pred_label = prediction("vgg", image)
    modelssd = model_load(modelName)
    img = load_img(image)
    img_array = img_to_array(img)
    img_array.resize(128,128,3)
    pred = modelssd.predict(x=np.array([img_array]))
    region = pred[0]
    x0, y0, w0, h0 = region
    #plotting
    fig,ax = plt.subplots()
    ax.imshow(img.resize((128,128)))
    rect_pred = patches.Rectangle((x0, y0), w0, h0, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect_pred)
    ax.plot(x0, y0, 'o', color='b') # top-left of the bounding box
    ax.plot(x0+w0, y0+h0, '*', color='c') # bottom-right of the bounding-box
    #plt.show()
    fig.savefig(os.path.join(os.getcwd(),'static',image.filename.split('.')[0]+modelName+".png"))
    return score, pred_label

    
if __name__ == "__main__":
    app.run(debug=True)
