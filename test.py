from classes import Lorenz
from classes import Protocol
import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

master = Lorenz()
slave = Lorenz()

sender = Protocol(master)
receiver = Protocol(slave)

key = sender.get_sequence(25)
print receiver.synchronize(key)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    start_time = time.time()
    frame = sender.encrypt(frame)
    print("--- Encrypt %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    frame = receiver.decrypt(frame)
    print("--- Decrypt %s seconds ---" % (time.time() - start_time))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
