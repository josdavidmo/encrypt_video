import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    heigth = np.random.randint(len(gray), size=len(gray))
    width = np.random.randint(len(gray[0]), size=len(gray[0]))

    gray = np.transpose(gray)
    for i in range(len(width)):
        gray[i] = np.roll(gray[i], width[i], axis=0)
    gray = np.transpose(gray)

    for i in range(len(heigth)):
        gray[i] = np.roll(gray[i], heigth[i], axis=0)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
