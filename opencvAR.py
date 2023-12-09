

import numpy as np
import imutils
import cv2


cap = cv2.VideoCapture(0)  


source_path = "/home/aahana/Pictures/squirrel.jpg"  
source = cv2.imread(source_path)

# Load the ArUCo dictionary and parameters
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from webcam")
        break

    # Resize the frame for faster processing (optional)
    frame = imutils.resize(frame, width=600)

    # Detect ArUCo markers in the webcam frame
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    output=frame
    # Check if 4 markers are found in the input frame
    if ids is not None and len(ids) == 4:
        ids = ids.flatten()
        refPts = []

        # Loop over the IDs of the ArUco markers
        for i in (923, 1001, 241, 1007):
            j = np.squeeze(np.where(ids == i))
            corner = np.squeeze(corners[j])
            refPts.append(corner)

        (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
        dstMat = np.array([refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]])

        # Perform augmentation
        (imgH, imgW) = frame.shape[:2]
        (srcH, srcW) = source.shape[:2]
        srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
        (H, _) = cv2.findHomography(srcMat, dstMat)
        warped = cv2.warpPerspective(source, H, (imgW, imgH))

        mask = np.zeros((imgH, imgW), dtype="uint8")
        cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)

        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, rect, iterations=2)

        maskScaled = mask.copy() / 255.0
        maskScaled = np.dstack([maskScaled] * 3)

        warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
        frameMultiplied = cv2.multiply(frame.astype(float), 1.0 - maskScaled)
        output = cv2.add(warpedMultiplied, frameMultiplied)
        output = output.astype("uint8")


    cv2.imshow("OpenCV AR Output", output)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
