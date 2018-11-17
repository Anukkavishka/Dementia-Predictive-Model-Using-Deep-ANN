from flask import Flask,render_template,request

from scipy.misc import imsave,imread,imresize
import numpy as np

from tensorflow.python.keras.models import Sequential

import re
import sys
import os
sys.path.append(os.path.abspath('./model'))
print(sys.path.append(os.path.abspath('./model')))

from load import *

app = Flask(__name__)

global model,graph
model,graph = init()

#helper functions by url patterns
@app.route('/')
def index() :
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict() :
    test_data=request.get_data()


if __name__ == '__main__' :
    port = int (os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port)
    

