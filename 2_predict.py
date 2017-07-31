#**** Copyright 2017 datawrangl3r.
 #
#**** Licensed under the Apache License, Version 2.0 (the "License");
#**** you may not use this file except in compliance with the License.
#**** You may obtain a copy of the License at
#
#****     http://www.apache.org/licenses/LICENSE-2.0
#
#**** Unless required by applicable law or agreed to in writing, software
#**** distributed under the License is distributed on an "AS IS" BASIS,
#**** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#**** See the License for the specific language governing permissions and
#**** limitations under the License.
# ==============================================================================

"""Predict a handwritten integer (MNIST beginners).
Make sure that:
* saved model (mnist_model.ckpt file) in the same location as that of this script.
* Invoke this script as:
    $ python 2_predict.py
"""

import sys
import tensorflow as tf
from flask import Flask, request
from PIL import Image,ImageFilter

app = Flask(__name__)

### Model Setup
# Define the model (same as when creating the model file)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
    
"""
    Load the model.ckpt file
    file is stored in the same directory as this python script is started
    Use the model to predict the integer. Integer is returend as list.
    Based on the documentatoin at
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
sess = tf.Session()
sess.run(init_op)
saver.restore(sess, "mnist_model.ckpt")

##############


@app.route("/predictint", methods=['GET'])
def predictint():
    """
    This function returns the predicted integer.
    The input is the pixel values from the imageprepare() function.
    """
    ######Input
    imname = request.args.get("imageName")
    ######

    imvalu = prepareImage(imname)
    prediction=tf.argmax(y,1)
    pred = prediction.eval(feed_dict={x: [imvalu]}, session=sess)
    return str(pred[0])


def prepareImage(argv):
    """
    This function returns the pixel values.
    Place the image file in the location where this code exists
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Height becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva  # Vector of values

if __name__ == "__main__":
    app.run()