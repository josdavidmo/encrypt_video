import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

from classes import Lorenz
from classes import Protocol

cap = cv2.VideoCapture(0)

master = Lorenz()
slave = Lorenz()
slave_uaci = Lorenz()

sender = Protocol(master)
sender_uaci = Protocol(slave_uaci)
receiver = Protocol(slave)

key = sender.get_sequence(25)
receiver.synchronize(key)
sender_uaci.synchronize(key)

plt.ion()

f, axarr = plt.subplots(2, 3)
plt.show()

while (True):
  axarr[0][0].clear()
  axarr[0][1].clear()
  axarr[0][2].clear()
  axarr[1][0].clear()
  axarr[1][1].clear()
  axarr[1][2].clear()

  # Capture frame-by-frame
  ret, frame = cap.read()

  start_time = time.time()
  encrypt_image = sender.encrypt(frame.copy())
  print("--- Encrypt %s seconds ---" % (time.time() - start_time))
  start_time = time.time()
  frame = receiver.decrypt(encrypt_image.copy())
  print("--- Decrypt %s seconds ---" % (time.time() - start_time))
  num_pixels = len(frame) * len(frame[0])

  encrypt_encrypt_image = encrypt_image.copy()
  row = np.random.randint(0, len(frame), 1)[0]
  column = np.random.randint(0, len(frame[0]), 1)[0]
  encrypt_encrypt_image[row][column] = (
                                           encrypt_encrypt_image[row][column] +
                                           np.random.randint(0, 255, 1)[
                                             0]) % 256
  encrypt_encrypt_image = sender_uaci.encrypt(encrypt_encrypt_image)
  color = ('b', 'g', 'r')

  for i, col in enumerate(color):
    histr = cv2.calcHist([encrypt_image], [i], None, [256], [0, 256])
    probability_histr = histr / num_pixels
    entropy = -np.sum(probability_histr * np.log2(probability_histr))
    print("--- Entropy %s %s ---" % (col, entropy))
    npcr = float(num_pixels - np.count_nonzero(np.equal(
      encrypt_image[:, :, i], encrypt_encrypt_image[:, :, i]))) / float(
      num_pixels)
    print("--- NPCR %s %s ---" % (col, npcr * 100))
    uaci = float(np.sum(
      abs((frame[:, :, i] - encrypt_image[:, :, i]) / 255))) / float(
      num_pixels)
    print("--- UACI %s %s ---" % (col, uaci * 100))
    axarr[0, i].plot(np.array(range(256)), histr, color=col)
    axarr[0, i].set_title(col)
    correlatin_matrix = np.corrcoef(encrypt_image[:, :, i])
    axarr[1, i].imshow(correlatin_matrix, interpolation='nearest')
  plt.draw()

  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  plt.pause(0.01)

# When everything done, release the capture
plt.ioff()
cap.release()
cv2.destroyAllWindows()
