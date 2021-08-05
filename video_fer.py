# library import
import cv2
import sys
from _models.vgg_fer import VGGFerModel
from image_proc_func.functions import draw_bbox_with_emotion

# read launch args
video_path = sys.argv[1]

if len(sys.argv) == 3:
    if sys.argv[2] == 'va':
        val_ar = True
else:
    val_ar = False

# initialize model
model = VGGFerModel(val_ar=val_ar)

# load xml for cv2 face detection
trained_face_data = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')

# define webcam for video capturing
vid = cv2.VideoCapture(video_path)

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()

    # frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # location of faces and drawing rectangles
    face_coordinates = trained_face_data.detectMultiScale(grayscale_frame)
    for (x, y, w, h) in face_coordinates:
        draw_bbox_with_emotion(frame, model, x, y, w, h)

    # show frame
    cv2.imshow('Facial Emotions Recognition', frame)

    # press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing camera
vid.release()
# destroy all the windows
cv2.destroyAllWindows()
