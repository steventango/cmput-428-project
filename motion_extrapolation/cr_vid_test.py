import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, minimize
from scipy.optimize import NonlinearConstraint as NC
import cv2
import lk
from time import time
from scipy.stats import linregress

def d(c):
  # optimize sum of squares of distances to mean circle
  r = np.sqrt((x-c[0])**2+(y-c[1])**2)
  return r - np.mean(r)

def fc(c):
  return np.sum(w*d(c)**2)

def fl(l):
  # optimize sum of squared distances to line
  return np.sum((np.array([l[0], l[1], 1]) @ (np.sqrt(w)*np.array([x, y, np.ones_like(x)])))**2) / (np.linalg.norm(l)**2)

def g(c):
  # constraint
  return np.mean(np.sqrt((x-c[0])**2+(y-c[1])**2)) - np.sqrt((x[-1]-c[0])**2+(y[-1]-c[1])**2)

w = np.ones(4)
points = []
imgs = []
x, y = np.zeros(0), np.zeros(0)     # tracked points
xc, yc = 0, 0                       # mean of tracked points
nstart = len(w)                     # how many points before display
npreds = nstart                     # how many extrapolated points to show
b = None

def record(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        points.append([x, y])
        cv2.circle(frame,(x,y), 4, (255,255,255), 2)
        cv2.imshow("test", frame)

cam = cv2.VideoCapture('testvid1.mp4')
cv2.namedWindow("test")

ret, frame = cam.read()
frame = cv2.rotate(frame, 2)
if not ret:
    print("Frame ", frame_id, " could not be read")
cv2.setMouseCallback("test", record)
print("Click feature points.")
cv2.imshow("test", frame)
cv2.waitKey(0)

out = cv2.VideoWriter('output2.avi', cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280,720))
mean_pt = (360, 640)
points = np.array(points)
lk.initTracker(frame, points)
framenum = 0
while ret:
    points = lk.updateTracker(frame)
    lk.drawRegion(frame, points.T, (0, 0, 0))
    if b == None:
        cv2.circle(frame, np.int16(mean_pt), 4, (255,255,255), 2)
    else:
        for i in range(nstart):
            cv2.circle(frame, (int(x[i]), int(y[i])), 4, (255,255,255), 2)
        for p in ppred.T:
            cv2.circle(frame, p, 4, (0,255,0), 2)
        if fc(c) < fl(l):
            cv2.circle(frame, np.int16(c), int(r), (255,0,0), 2)
        else:
            cv2.line(frame, (0, int(-1/l[1])), (1280, int(-(1+1280*l[0])/l[1])), (255,0,0), 2)

    # out.write(frame)
    cv2.imwrite("frame%04i.png" % framenum, frame)
    framenum+=1
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break

    if framenum % 2 == 0:
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
        
            c = minimize(fc, [xc, yc], method="trust-constr").x
            
            xl = np.expand_dims(x, axis=0)      # to use in fl calc (concatenation)
            yl = np.expand_dims(y, axis=0)  
            ml = (y[-1] - y[0]) / (x[-1] - x[0])
            bl = y[-1] - ml * x[-1]
            l = minimize(fl, [ml/bl, -1/bl], method="trust-constr").x
        
            if fc(c) < fl(l):
              r = np.mean(np.sqrt((x-c[0])**2+(y-c[1])**2))
              tmeas = np.arctan((y - c[1])/(x - c[0])) + np.where(x < c[0], np.pi, 0)
              m, b, _, _, _ = linregress(np.arange(len(x)), tmeas)
              numpred = len(x) + np.arange(npreds)
              tpred = m * numpred + b
              ppred = np.array([np.int16(r*np.cos(tpred) + c[0]), np.int16(r * np.sin(tpred) + c[1])])
            else:
              # l[0] * x + l[1] * y + 1 = 0
              if x[-1] > x[0]:
                xstart, ystart, xend, yend = min(x), min(y), max(x), max(y)
              else:
                xend, yend, xstart, ystart = min(x), min(y), max(x), max(y)
              vecs = (np.array([x, y]) - np.array([[xstart], [ystart]])).T
              dr = np.array([xend - xstart, yend - ystart])
              dr = dr / np.linalg.norm(dr)
              dists = vecs @ dr
              incs = np.diff(dists)

              m, b, _, _, _ = linregress(np.arange(len(x)-1), incs)

              tpred = len(x) - 1 + np.arange(npreds)
              incpred = m * tpred + b
              distpred = dists[-1] + np.cumsum(incpred)
              xpred = xstart + distpred * dr[0]
              ppred = np.array([np.int16(xpred), np.int16(-(l[0]*xpred+1)/l[1])])
          
        snap_time = time()

    ret, frame = cam.read()
    frame = cv2.rotate(frame, 2)

print("fin")
out.release()
cv2.destroyAllWindows()
