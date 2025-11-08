import os
import time
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Flask App Configuration
app = Flask(__name__)
app.secret_key = 'supersecretkey123'  # Change this to a secure random key
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Model Loading
print('Model loading...')
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)

# Load weights
model_03.load_weights(
    r"C:\Users\yashwanth\Documents\Advance_Brain_Tumor_Classification-main\model_weights\vgg19_model_02.weights.h5"
)
print('Model loaded successfully. Open: http://127.0.0.1:5000/')

# Helper Functions

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Astrocytoma Tumor"

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((240, 240))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01

# Authentication Config

USERNAME = 'admins'
PASSWORD = '123456'

@app.route('/', methods=['GET', 'POST'])
def login():
    """Login route"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == USERNAME and password == PASSWORD:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout route"""
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/home')
def index():
    """Main app page"""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', version=time.time())

@app.route('/predict', methods=['POST'])
def upload():
    """Prediction route"""
    if 'user' not in session:
        return redirect(url_for('login'))

    f = request.files['file']
    basepath = os.path.dirname(__file__)
    uploads_dir = os.path.join(basepath, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, secure_filename(f.filename))
    f.save(file_path)

    value = getResult(file_path)
    result = get_className(value)
    return result

# Contact Page Route
@app.route('/contact')
def contact():
    """Contact Information Page"""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('contact.html')

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
