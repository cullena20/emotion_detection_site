# Emotion Detection Site
This is a website that performs emotion detection on submitted images. A user submits an
image and a machine learning model detects the faces in the image and performs emotion
analysis on them. Faces are classified as one of seven emotions: anger, disguist, fear, 
happiness, sadness and neutrality. The website was built using Flask. I used the
face_recogniton library to detect faces in the image. I used Keras and the fer2013
database to train the emotion detection model. To learn more about the methodology,
see my other repository containing the Colab notebooks at https://github.com/cullena20/emotion_detection.

## Downloading the model
The model file is too large to be pushed to this repository. Download the model from the following
link: https://drive.google.com/file/d/1mN3txFHks9UNZ3Tw53Sxmhvo26ljPh2e/view?usp=sharing.


## Note
This project has some problems. For some reason, no faces are detected in lots of images. I mostly
did this project for the ai part and also to get better at deploying a model on a website. I would
say I've accomplished what I wanted so I may not bother with this troubleshooting that I can't find
a solution. I'm leaving the print statements here for now in case I want to come back to this project.