import numpy as np
import PIL
import matplotlib.image as mpimg
import face_recognition
import cv2


def get_faces(image_fp):
    '''
    Returns a list of faces identified in an image
    Input: image filepath
    Output: 
        faces: list of numpy arrays containing faces identified from image
        face_locations: coordinates of faces in image
    '''
    image = mpimg.imread(image_fp)
    faces = list()
    face_locations = face_recognition.face_locations(image)
    for i in range(len(face_locations)):
        top, right, bottom, left = face_locations[i]
        face_image = image[top:bottom, left:right]
        faces.append(face_image)
    return faces, face_locations


def process_faces(faces):
    '''
    Preprocesses images of faces to run through machine learning model
    Input: list containing images of faces as numpy arrays
    Output: list containg processed images of faces as numpy arrays
    '''
    processed_faces = list()
    for i in range(len(faces)):
        pil_face = PIL.Image.fromarray(faces[i])
        pil_face.thumbnail((48, 48))
        face_image = np.array(pil_face)
        face_image = face_image / 255
        face_image = np.expand_dims(face_image, axis=0)
        processed_faces.append(face_image)
    return processed_faces


def make_predictions(model, image_fp):
    '''
    Performs facial emotion detection from an image using a pretrained model
    Input: 
        model: a h5 file containing the trained model
        image_fpL the filepath of an image to be analyzed
    Output: 
        faces: a list containing the faces identified in the image as numpy arrays
        predictions: a list of predicted emotions from faces
    '''
    faces, face_locations = get_faces(image_fp)
    faces = process_faces(faces)
    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    predictions = list()
    for face in faces:
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        predictions.append(emotion)
    return predictions, face_locations


def draw_predictions(image_fp, face_locations, predictions):
    '''
    Draws boxes around faces and writes in predictions
    '''
    image = cv2.imread(image_fp)
    start = list()  # top right
    end = list()  # bottom left
    for i in face_locations:
        start.append((i[1], i[0]))
        end.append((i[3], i[2]))
    for j in range(len(start)):
        image = cv2.rectangle(image, start[j], end[j], (36,255,12), 2)
        cv2.putText(image, predictions[j], (end[j][0], start[j][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image
    
# i could change this to return coordinates of faces instead and use open cv to display that