from flask import Flask
from flask import Flask, redirect, url_for, request, render_template,jsonify
import  tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import  os
import pandas as pd
from werkzeug.utils import secure_filename
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.ensemble import  RandomForestClassifier
import  multiprocessing as mp



import pickle
app=Flask(__name__)
MODEL_PATH = r'models\vgg16.h5'
model_2_path=r"models\rfc.pkl"
sooraj_model1_path=r"models\soorajmodel.h5"
sooraj_model2_path=r"models\soorajmodel2.h5"
# Load  trained model
model = load_model(MODEL_PATH)
model_2=pickle.load(open(model_2_path,'rb'))
sooraj_model1=load_model(sooraj_model1_path)
sooraj_model2=load_model(sooraj_model2_path)
print('Models loaded. Check http://127.0.0.1:5000/')
def save_spectrogram1(audio_fname):
    y, sr = librosa.load(audio_fname, sr=None)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    fig1 = plt.gcf()
    plt.axis('off')
    # plt.show()
    plt.draw()
    plt.close(fig1)
    image_fname="img"
    fig1.savefig(image_fname, dpi=200)
    return image_fname + ".png"
def img_pred(img_path, model):

  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.true_divide(x, 255)
  x = np.expand_dims(x,axis=0)
  preds = model.predict(x)
  ans=np.argmax(preds)


  return ans
def mod_2(aud_path,model_2):
    X, sample_rate = librosa.load(aud_path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
    pred=model_2.predict(mfccs.reshape(1,-1))[0]
    return pred
def mod_1(audio_path,model):
    img_path = save_spectrogram1(audio_path)
    ans = img_pred(img_path, model)
    return ans
def soorajmodel1(path, model):

    X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)

    # get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128, fmax=8000)
    db_spec = librosa.power_to_db(spectrogram)
    # temporally average spectrogram
    log_spectrogram = np.mean(db_spec, axis=0)


    X_test = np.array([log_spectrogram])

    X_test = X_test[:, :, np.newaxis]

    predict = model.predict(X_test)
    predict = predict.argmax(axis=1)
    predict = predict.astype(int).flatten()

    return predict[0]


def soorajmodel2(file_path,model2):
    n_mfcc = 30
    sampling_rate=44100
    n=n_mfcc
    f={'file_path':[file_path]}
    df= pd.DataFrame(f)
    X = np.empty(shape=(df.shape[0],n , 216, 1))
    input_length = 44100 * 2.5
    cnt=0
    data, sample_rate = librosa.load(file_path, res_type='kaiser_fast',duration=2.5,sr=44100,offset=0.5)

    if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")


    n_mfcc = 30

    MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
    MFCC = np.expand_dims(MFCC, axis=-1)
    X[cnt,] = MFCC


    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean)/std

    prediction_model_2 = model2.predict(X)

    prediction_model_2=prediction_model_2.argmax(axis=1)
    prediction_model_2 = prediction_model_2.astype(int).flatten()


    return prediction_model_2[0]
def model_predict(audio_path,s):
  d = ["angry", "calm", "happy", "sad"]
  if s==0:
      ans=ensemble(audio_path)
  if s==1:
      ans=mod_1(audio_path,model)
  elif s==2:
      ans =mod_2(audio_path, model_2)
  elif s==3:
      ans=soorajmodel1(audio_path,sooraj_model1)
  elif s==4:
      ans=soorajmodel2(audio_path,sooraj_model2)
  # return ans


  return d[ans]
def ensemble(audio_path):
    vote=[]
    vote.append(mod_1(audio_path,model))
    vote.append(mod_2(audio_path,model_2))
    vote.append(soorajmodel1(audio_path,sooraj_model1))
    vote.append(soorajmodel2(audio_path, sooraj_model2))
    return max(set(vote), key = vote.count)
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict",methods=["GET","POST"])
def pred():
    if request.method=="POST":
        f = request.files['file']
        s=[int(x) for x in request.form.values()]


        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path,s[0])
        os.remove(file_path)


        return str(preds)

    return None
if __name__=="main":
    app.run()


