
from flask import Flask, render_template, redirect, request
import expresso
ap=Flask(__name__)

@ap.route('/')
def hello():
    return render_template("index1.html")

@ap.route('/', methods=['POST'])
def predict():
    if request.method =='POST':
        f=request.files['userfile']
        path="./static/{}".format(f.filename)
        f.save(path)
        pred= expresso.predict_emotion(path)

        result_dic={
            'image': path,
            'prediction': pred
        }

    

        return render_template("index1.html", your_result=result_dic)

if __name__ =='__main__':
    ap.run(debug = True)




