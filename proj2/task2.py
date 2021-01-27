import cv2
import numpy as np


################################ Part 1 & 2 #####################################
def calculation(rho, theta):
    # Some of the codes are adapted from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    a,b = np.cos(theta),np.sin(theta)
    x0,y0 = a*rho,b*rho
    x1,y1,x2,y2 = int(x0 + 1000*(-b)),int(y0 + 1000*(a)),int(x0 - 1000*(-b)),int(y0 - 1000*(a))
    double_line = False
    if theta < 3:
        for i in temp_b:
            if abs(x1 - i) < 6:
                # print((x1,y1),(x2,y2))
                double_line = True
        if not double_line:
            temp_b.append(x1)
            return False
        return True
    else:
        for i in temp_r:
            if abs(x1 - i) < 6:
                # print((x1,y1),(x2,y2))
                double_line = True
        if not double_line:
            temp_r.append(x1)
            return False
        return True


def draw_lines(img,params,color):
    # Some of the codes are adapted from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    for theta, rho in params:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1, x2, y2 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)), int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(img,(x1,y1),(x2,y2),color,2)
    return img


def save_params(filename,params):
    with open(f'results/{filename}.txt', 'w') as f:
        for i in params:
            f.write(str(i)+'\n')


img = cv2.imread('Hough.png')

# detect edages
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,200)

# find lines
lines = cv2.HoughLines(edges,1,np.pi/90,150)

# get params
temp_r,temp_b = [0],[0]
blue_params = []
red_params = []

for i in range(len(lines)):
    for rho, theta in lines[i]:
        if theta < 3:
            double_line = calculation(rho, theta)
            if not double_line:
                blue_params.append([theta, rho])
        else:
            double_line = calculation(rho, theta)
            if not double_line:
                red_params.append([theta, rho])

# for undetected line, I do crop the image to small piece to detect the line
# Some of the codes are adapted from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
sub_img = img[500:,:130].copy()
sub_gray = cv2.cvtColor(sub_img,cv2.COLOR_BGR2GRAY)
sub_edges = cv2.Canny(sub_gray,100,200)
lines = cv2.HoughLines(sub_edges,0.36,np.pi/90,20)
for rho,theta in lines[0]:
    rho += 295
    blue_params.append([theta, rho])

# draw lines & save img
color = (255,0,0)
blue = draw_lines(img.copy(),blue_params,color)
cv2.imwrite('results/blue_lines.jpg', blue)
color = (0,0,255)
red = draw_lines(img.copy(),red_params,color)
cv2.imwrite('results/red_lines.jpg', red)


# save rho,theta
def angle_trans(params):
    for i in range(len(params)):
        theta = params[i][0]
        params[i][0] = int(theta / np.pi * 180)
    return params


save_params('blue_lines',angle_trans(blue_params))
save_params('red_lines',angle_trans(red_params))


################################ Part 3 #####################################
coins = img.copy()
cim = cv2.medianBlur(gray,5)
circles = cv2.HoughCircles(cim,cv2.HOUGH_GRADIENT,1,40
                           ,param1=50,param2=30,minRadius=20,maxRadius=50)
# Some of the codes are adapted from: https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html
circles = np.uint16(np.around(circles))
with open('results/coins.txt','w') as f:
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(coins,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(coins,(i[0],i[1]),2,(0,0,255),3)
        # save params
        f.write(str([i[0],i[1],i[2]])+'\n')
# save img
cv2.imwrite('results/coins.jpg', coins)