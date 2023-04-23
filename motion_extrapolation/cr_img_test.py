import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, minimize
from scipy.optimize import NonlinearConstraint as NC
import cv2
import lk
from time import time

# program for testing curve fitting on real images. can safely ignore

def d(c):
  # optimize sum of squares of distances to mean circle
  r = np.sqrt((x-c[0])**2+(y-c[1])**2)
  return r - np.mean(r)

def f(c):
  return np.sum(d(c)**2)

def g(c):
  # constraint
  return np.mean(np.sqrt((x-c[0])**2+(y-c[1])**2)) - np.sqrt((x[-1]-c[0])**2+(y[-1]-c[1])**2)

points = []
imgs = []
x, y = np.zeros(0), np.zeros(0)
xc, yc = 0, 0
nstart = 5

def record(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        points.append([x, y])
        cv2.circle(frame,(x,y), 4, (255,255,255), 2)
        cv2.imshow("test", frame)

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

ret, frame = cam.read()
if not ret:
    print("Frame ", frame_id, " could not be read")
cv2.setMouseCallback("test", record)
print("Click feature points.")
cv2.imshow("test", frame)
cv2.waitKey(0)

mean_pt = (320, 240)
points = np.array(points)
lk.initTracker(frame, points)
snap_time = time()
while True:
    ret, frame = cam.read()
    points = lk.updateTracker(frame)
    lk.drawRegion(frame, points.T, (0, 0, 0))

    if len(x) < nstart:
        cv2.circle(frame, np.int32(mean_pt), 4, (255,255,255), 2)
    else:
        for i in range(nstart):
            cv2.circle(frame, (int(x[i]), int(y[i])), 4, (255,255,255), 2)
        cv2.circle(frame, np.int32(c), int(r), (255,0,0), 2)
    
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break

    if time() - snap_time > 0.1:
        mean_pt = np.mean(points.squeeze(), axis=0)
        x = np.concatenate((x, mean_pt[:1]))
        y = np.concatenate((y, mean_pt[1:]))
        if len(x) == nstart:
            xc = np.mean(x)
            yc = np.mean(y)
        if len(x) > nstart:
            xc += (x[-1] - x[0])/len(x)
            yc += (y[-1] - y[0])/len(y)
            x = x[1:]
            y = y[1:]
    
        c = minimize(f, [xc, yc], method="trust-constr").x
        r = np.mean(np.sqrt((x-c[0])**2+(y-c[1])**2))
        
        snap_time = time()
