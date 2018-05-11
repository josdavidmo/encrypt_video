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
    h = 0.01
    length = (len(frame) + len(frame[0])) * h
    sequence = np.rint(sender.get_sequence(length, h) * 100)
    sequence_x = sequence[0:len(frame)][:, 0]
    sequence_y = sequence[0:len(frame)][:, 1]
    sequence_z = sequence[0:len(frame)][:, 2]
    sequence_x_width = sequence[len(frame) + 1:][:, 0]
    img = sender.permute(frame, sequence_x, sequence_x_width, 1)
    img = sender.difusion(img, sequence_x, sequence_y, sequence_z, 1)
    #encrypt_image = sender.encrypt(frame.copy())
    print("--- Encrypt %s seconds ---" % (time.time() - start_time))
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
