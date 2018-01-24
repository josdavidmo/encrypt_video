from classes import Lorenz
from classes import Protocolo
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

master = Lorenz()
slave = Lorenz()

sender = Protocolo(master)
receiver = Protocolo(slave)

key = sender.get_sequence(500, 0.1)
receiver.synchronize(key, 0.1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    blue, green, red = cv2.split(frame)

    # Display the resulting frame
    cv2.imshow('frame', red)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
