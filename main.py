from collections import deque
from imutils.video import WebcamVideoStream
import imutils
import cv2
import time
import math

# A list to put moving trajectory
traceList = []
vs = WebcamVideoStream(src=0).start()

# Background frame
lastFrame = None
time.sleep(1)

# A distance to determine whether the two objects detected in two frames is the same object.
frameDistance = 100

# Window size
winWidth = 500
winHeight = 300

# Color filter (RED)
redLower = (120, 4, 24)
redUpper = (226, 58, 96)

# Standard line
inLine = int(winWidth/2 + 30)
outLine = int(winWidth/2 - 30)
# Indoor direction
inVector = (1,0) # x-axis

inCount = 0
outCount = 0


def dotproduct(v1, v2):
    '''
    Distance between 2 dots
    :param v1: dot1
    :param v2: dot2
    :return:
    '''
    return sum((a*b) for a, b in zip(v1, v2))


while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=winWidth, height=winHeight)
    # Tracing color
    rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.inRange(rbg, redLower, redUpper)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if lastFrame is None:
        lastFrame = gray
        continue

    frameDelta = cv2.absdiff(lastFrame, gray)
    dst = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    dst = cv2.dilate(dst, None, iterations=5)
    _, cnts, _ = cv2.findContours(dst.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Every detected rectangle
    for c in cnts:
        print(cv2.contourArea(c))
        c_check = False
        # Guess why ^v^
        if (cv2.contourArea(c)) < 1000 or (cv2.contourArea(c) > 30000):
            continue

        if cv2.contourArea(c) > 130000:
            lastFrame = gray
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        cv2.circle(frame, center, 1, (0, 0, 255), 5)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 经过离散优化后的代码逻辑，原逻辑是判断 在进出线范围内的循环 和 不在进出线内的循环
        # 原谅我这段英语说不来
        # Record the trajectory of each object
        for k,v in enumerate(traceList):
            if math.hypot(center[0] - v[0][0], center[1] - v[0][1]) < frameDistance:
                if (center[0] < inLine) and (center[0] > outLine):
                    v.appendleft(center)
                    cv2.putText(frame, str(k), center, cv2.FONT_ITALIC, 1.0, (255, 255, 255), 1)
                    c_check = True
                    break
                elif len(v) >= 2:
                    firstPoint = v[len(v)-1]
                    dx = v[0][0] - firstPoint[0]
                    dy = v[0][1] - firstPoint[1]
                    vt = (dx, dy)
                    if dotproduct(vt, inVector) > 0:
                        inCount += 1
                    else:
                        outCount += 1
                    traceList.remove(v)
                    break

        if (c_check is False) and (center[0] < inLine) and (center[0] > outLine):
            # Treat one dot as a new center point of one object
            traceList.append(deque([center]))

    cv2.putText(frame, "InCount: {}".format(str(inCount)), (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "OutCount: {}".format(str(outCount)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.line(frame, (outLine, 0), (outLine, winHeight), (250, 0, 1), 2)  # blue line
    cv2.line(frame, (inLine, 0), (inLine, winHeight), (0, 0, 255), 2)  # red line

    cv2.imshow("contours", dst)
    cv2.imshow("origin", frame)
    key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
vs.stop()
