
from flask import Flask, render_template, redirect, request
import streamlit as st
import expresso
app=Flask(__name__)

@app.route('/')
def hello():
    return render_template("index1.html")

@app.route('/', methods=['POST'])
def predict():
    if request.method =='POST':
        f=request.files['userfile']
        #path="./assets/images/{}".format(f.filename)
        path=st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
        f.save(path)
        pred= expresso.predict_emotion(path)

        result_dic={
            'image': path,
            'prediction': pred
        }

    

        return render_template("index1.html", your_result=result_dic)

if __name__ =='__main__':
    app.run(debug = True)




