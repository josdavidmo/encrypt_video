from classes import Lorenz
from classes import Protocol
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

master = Lorenz()
slave = Lorenz()

sender = Protocol(master)
receiver = Protocol(slave)

key = sender.get_sequence(25)
print receiver.synchronize(key)

#print sender.get_sequence(1)
#print receiver.get_sequence(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    fram = sender.encrypt(frame)
    frame = receiver.decrypt(frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
