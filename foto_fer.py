# library import
import cv2
import sys
from _models.vgg_fer import VGGFerModel
from image_proc_func.functions import draw_bbox_with_emotion

# read launch args
img_path = sys.argv[1]
img = cv2.imread(img_path)

if len(sys.argv) == 3 and sys.argv[2] == 'va':
    val_ar = True
else:
    val_ar = False

# initialize model
model = VGGFerModel(val_ar=val_ar)

# load xml for cv2 face detection
trained_face_data = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')

# draw labeled rectangles
grayscale_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# location of faces and drawing rectangles
face_coordinates = trained_face_data.detectMultiScale(grayscale_frame)
for (x, y, w, h) in face_coordinates:
    draw_bbox_with_emotion(img, model, x, y, w, h)

# show image
cv2.imshow('Facial Emotions Recognition', img)
cv2.waitKey(0)
