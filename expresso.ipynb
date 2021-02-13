import PIL
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from PIL import ImageOps


import numpy as np



import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model=load_model("final_model.h5")
#model._make_predict_function()


def predict_emotion(path):
    
    
    #frame=input("Enter image ")
    i=Image.open(path)
    im2 = ImageOps.grayscale(i) 
    faces=face_cascade.detectMultiScale(np.array(im2),1.3,5)
    if len(faces)!=0:
            
        (x,y,w,h)=faces[0]
        crop=np.array(im2)[y:y+h, x:x+w]
        s=cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
        norm_img = np.zeros((300, 300))
        norm_img = cv2.normalize(s, norm_img, 0, 255, cv2.NORM_MINMAX)
        #PIL_image = Image.fromarray(np.uint8(norm_img)).convert('RGB')
        PIL_image=np.asarray(norm_img)
        #PIL_image=PIL_image[:,:,1]
        #PIL_image.save("out.jpg")
        #plt.imshow(PIL_image)
        #print(PIL_image.shape)
        #model.predict(PIL_image)
        PIL_image=PIL_image.reshape(1,64,64,1)
        pre=model.predict(PIL_image)
        o=np.argmax(pre)
        if o==1:
            return "happy"
        elif o==2:
            return "scared"
        elif o==3:
            return "sad"
        elif o==4:
            return"angry"
        elif o==5:
            return "neutral"
        else:
            return "Try again with a different image"
        

        #return o
    else:
        return "No face detected. Try again with a different image."







