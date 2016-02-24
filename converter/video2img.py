import numpy as np
import cv2

videoFileName = 'forward.avi'
outFolder = 'imgs\\'

cap = cv2.VideoCapture(videoFileName)
imInd = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
#    if (~ret):
#        break
    imName = outFolder + ("%06d" % imInd) + '.bmp'
    cv2.imwrite(imName, frame)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    imInd = imInd+1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()