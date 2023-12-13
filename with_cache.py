from augmented_reality import find_and_warp
from imutils.video import VideoStream
from collections import deque
import imutils
import time
import cv2


input_image_path = "squirrel.jpg"

print("[INFO] initializing marker detector...")
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()


source = cv2.imread(input_image_path)


useCache = True


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

source_path = "squirrel.jpg" 

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    warped = find_and_warp(
        frame, source,
        cornerIDs=(923, 1001, 241, 1007),
        arucoDict=arucoDict,
        arucoParams=arucoParams,
        useCache=useCache)

    if warped is not None:
        frame = warped
        source = cv2.imread(source_path)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
