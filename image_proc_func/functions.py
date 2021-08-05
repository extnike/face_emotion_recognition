import cv2


def get_emotion_from_image(model_for_prediction, img, x: int, y: int, w: int, h: int):
    """
    Function to get emotion from given photo and bounding box with given coordinates

    :param model_for_prediction: pretrained model for FER on 9 class
    :param img: image to find emotion
    :param x: x of top-left corner of face bbox
    :param y: y of top-left corner of face bbox
    :param w: width of face bbox
    :param h: height of of face bbox
    :return: name of emotion
    """
    # cut face from image
    face_frame = img[y:y + h, x:x + w]
    # make prediction with given model
    prediction = model_for_prediction.predict(face_frame)
    return prediction


def add_caption_of_emotion(img, emotion_name: str, x: int, y: int):
    """
    Add given caption of emotion on given image, having x and y as top-left corner of face bbox

    :param img: image to write emotion name
    :param emotion_name:
    :param x: x of top-left corner of face bbox
    :param y: y of top-left corner of face bbox
    :return: put text on given image
    """
    # font settings
    font_name = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    font_line_type = cv2.LINE_AA
    font_color = (255, 255, 255)

    # label_location. I put label in 2px up of bbox with the same x
    label_org = (x, y - 2)

    # put text on img with given emotion name
    cv2.putText(img,
                text=emotion_name,
                org=label_org,
                fontFace=font_name,
                fontScale=font_scale,
                color=font_color,
                thickness=font_thickness,
                lineType=font_line_type)
    return img


def draw_bbox_with_emotion(img, model, x: int, y: int, w: int, h: int):
    """
    Function to draw face bbox with name of demonstrated emotion

    :param img: Image to work with
    :param model: Model to emotion prediction
    :param x: x of top-left corner of face bbox
    :param y: y of top-left corner of face bbox
    :param w: width of bbox
    :param h: height of bbox
    :return: add bbox with labeled emotion to img
    """
    # get emotion prediction
    prediction = get_emotion_from_image(model, img, x, y, w, h)
    
    # if model returns string value of emotion
    if isinstance(prediction, str):
        emotion_name = prediction
        rectangle_color = (150, 150, 150)
        
    # if model returns list of emotion, valence and arousal values
    if isinstance(prediction, list):
        # get values from predicted list
        emotion_name = prediction[0]
        valence = round(prediction[1], 2)
        arousal = round(prediction[2], 2)
        # construct emotion name with valence and arousal values
        emotion_name = emotion_name + "; V. =" + str(valence) + "; A. = " + str(arousal)
        # red color for negative valence
        if valence < 4:
            rectangle_color = (0, 0, 255)
        # green color for negative valence
        elif valence > 4:
            rectangle_color = (0, 255, 0)
        # grey color for zero valence
        else:
            rectangle_color = (100, 100, 100)
    # draw bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), rectangle_color, thickness=2)
    # write caption
    add_caption_of_emotion(img, emotion_name, x, y)
    return img
