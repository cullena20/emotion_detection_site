from flask import Flask, request, url_for, render_template, redirect, send_file
from werkzeug.utils import secure_filename
import imghdr
import os
from tensorflow.keras.models import load_model
from machinelearning import ml
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2048 * 2048  # only accepts files up to 2 mb
app.config['UPLOAD_EXTENSIONS'] = ['.jpeg', '.jpg', '.png']
app.config['UPLOAD_PATH'] = 'uploads'

model = load_model("best_cnn_model.h5")


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


def serve_image(image):
    '''
    Takes in an image as a numpy array so that it can be displayed directly on
    a website
    '''
    image = Image.fromarray(image.astype('uint8'))
    bytesio = io.BytesIO
    image.save(bytesio, format='JPEG', quality=70)
    bytesio.seek(0)
    image = base64.b64encode(bytesio.getvalue())
    return image


@app.route('/')
def index():
    return render_template('index.html')


# i put in a lot of security stuff here that isn't totally necessary for my purposes but why not
@app.route('/', methods=["POST"])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or file_ext != validate_image(file.stream):
            return render_template('index.html', error='Please submit an image (.jpg, .jpeg, or .png)')
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, app.config['UPLOAD_PATH'], filename)
        file.save(file_path)
        predictions, face_locations = ml.make_predictions(model, file_path)
        image = ml.draw_predictions(file_path, face_locations, predictions)
        # serve image is not working
        image = serve_image(image)
        return render_template('index.html', predictions=predictions, image=image.decode('ascii'))
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run()

# im basically copying code for displaying the image
# figure this out after the gym
# also lots of error handling
# but i got the general thing yippee!

# learn more about os module. it always comes up but i am not familiar with it
# learn more about this io stuff