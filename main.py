from classes import Lorenz
from classes import Protocol
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import math
import numpy as np
import time

cap = cv2.VideoCapture(0)

master = Lorenz()
slave = Lorenz()

sender = Protocol(master)
receiver = Protocol(slave)

key = sender.get_sequence(25)
receiver.synchronize(key)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    start_time = time.time()
    encrypt_image = sender.encrypt(frame.copy())
    image = receiver.decrypt(encrypt_image)
    print("--- Encrypt %s seconds ---" % (time.time() - start_time))
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
