from classes import Lorenz
from classes import Protocol
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

master = Lorenz()
slave = Lorenz()

sender = Protocol(master)
receiver = Protocol(slave)

key = sender.get_sequence(25)
receiver.synchronize(key)

plt.ion()

while(True):
    plt.clf()

    # Capture frame-by-frame
    ret, frame = cap.read()

    start_time = time.time()
    frame = sender.encrypt(frame)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([frame], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
    print("--- Encrypt %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    frame = receiver.decrypt(frame)
    print("--- Decrypt %s seconds ---" % (time.time() - start_time))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    plt.pause(0.1)

# When everything done, release the capture
plt.ioff()
cap.release()
cv2.destroyAllWindows()
