# Emotion Detection Site
This is a website that performs emotion detection on submitted images. A user submits an
image and a machine learning model detects the faces in the image and performs emotion
analysis on them. Faces are classified as one of seven emotions: anger, disguist, fear, 
happiness, sadness and neutrality. The website was built using Flask. I used the
face_recogniton library to detect faces in the image. I used Keras and the fer2013
database to train the emotion detection model. To learn more about the methodology,
see my other repository containing the Colab notebooks at https://github.com/cullena20/emotion_detection.