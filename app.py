from flask import Flask, request, url_for, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from machinelearning import ml
from PIL import Image
import io
import base64


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024  # only accepts files up to 6 mb
app.config['UPLOAD_EXTENSIONS'] = ['.jpeg', '.jpg', '.png', '.heic']
app.config['UPLOAD_PATH'] = 'uploads'


model = load_model("vgg16_emotion_detection.h5")


def serve_image(image):
    '''
    Takes in an image as a numpy array so that it can be displayed directly on
    a website
    '''
    image = Image.fromarray(image.astype('uint8'))
    data = io.BytesIO()
    image.save(data, 'JPEG')
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=["POST"])
def upload_file():
    print(model)
    file = request.files['file']
    filename = secure_filename(file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1].lower()
        print(file_ext)
        if file_ext not in app.config['UPLOAD_EXTENSIONS']: # or file_ext != validate_image(file.stream) might be source of some problems
            return render_template('index.html', error='This image is not recognized. Please submit a valid image (jpg, jpeg, png).')
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, app.config['UPLOAD_PATH'], filename)
        file.save(file_path)
        try:
            predictions, face_locations, error = ml.make_predictions(model, file_path)
            print(predictions, error)
        except ValueError:
            print("Value Error")
            return render_template('index.html', error='The model is having trouble analyzing this image. (faces are not being properly sized) To be fixed.')
        except Exception:
            print("Exception")
            return render_template('index.html', error='Something went wrong. To be fixed.')
        image = ml.draw_predictions(file_path, face_locations, predictions)
        encoded_img_data = serve_image(image)
        if error is not None:
            return render_template('index.html', error=error)
        return render_template('index.html', predictions=predictions, img_data=encoded_img_data.decode('utf-8'))
    return redirect(url_for('index'))


# this was to learn how to display images and is not used
@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0')


# learn more about os module. it always comes up but i am not familiar with it
# learn more about this io stuff