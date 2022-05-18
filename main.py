import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
def zoom(img, zoom_factor=1.7):
    if img.size!= 0:
        y_size = img.shape[0]
        x_size = img.shape[1]
        # define new boundarie
        x1 = int(0.5*x_size*(1-1/zoom_factor))
        x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
        y1 = int(0.5*y_size*(1-1/zoom_factor))
        y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))
        # first crop image then scale
        img_cropped = img[y1:y2,x1:x2]
        if (img_cropped.size != 0 and zoom_factor!= 0):
            zoom = cv2.resize(img_cropped, (img.shape[1],img.shape[0]))
            
        return zoom
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray)
    for (x,y, w, h) in faces:
        #mouth
        h1 = int(h // 3)
        w1 = int(w // 3)
        mouth = frame[y+h1*2:y+h1*3-h1//3,x+w1:x+w1*2+w1//4]
        zoomed_mouth = zoom(mouth)
        if type(zoomed_mouth) == type(frame):
            if (mouth.shape == zoomed_mouth.shape):
                frame[y+h1*2:y+h1*3-h1//3,x+w1:x+w1*2+w1//4] = zoomed_mouth
        #eye
        h2 = int(h // 4)
        w2 = int(w // 4)
        eye = frame[y+h2:y+h2*2,x+h2//2:x+w2*4-h2//2]
        zoomed_eye = zoom(eye,zoom_factor = 1.2)
        if type(zoomed_eye) == type(frame):
            print(eye.shape, zoomed_eye.shape)
            if (eye.shape == zoomed_eye.shape):
                frame[y+h2:y+h2*2,x+h2//2:x+w2*4-h2//2] = zoomed_eye
        frame[y:y+h,x:x+w] = cv2.fastNlMeansDenoisingColored(frame[y:y+h,x:x+w],None,10,10,7,21)
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
