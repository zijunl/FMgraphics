## this script try a single image and transform a 3D color image to a graph(V,E)

import networkx as nx
import numpy as np
import cv2
import math
import cv2

K = 5


def calAllDist(pix, index,m, n):
    row = index//n
    col = index%n
    graphSize = m*n
    res = np.zeros(graphSize)
    for x in range(m):
        for y in range(n):
            pos = x*m +y
            deb = pix[row,col]
            a = np.array((row, col, pix[row,col]))
            b = np.array((x, y, pix[x,y]))
            temp = np.linalg.norm(a - b)
            res[x*n+y] = np.linalg.norm(np.array((row, col, pix[row,col])) - np.array((x, y, pix[x,y])))
    return res


def L2FM2(file):
    weight = 1
    raw = cv2.imread(file)
    grey = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    m, n = grey.shape
    pix = np.zeros(shape = (m,n))
    #normalize

    for i, p in enumerate(grey):
        for j, pj in enumerate(p):
            pix[i][j] = pj * weight * math.sqrt(m * n) / 255
            # pix[i] = p[i] * weight * math.sqrt(m*n)/255

    p = pix[0,0]
    graphSize = m*n
    e = 1
    C = (m*n)**2
    p = np.zeros((graphSize, K))
    for r in range(K):
        va = 0                       # randomly chosen
        vb = va
        for t in range(C):
            dai = calAllDist(pix, va, m, n)
            temp = np.zeros(graphSize)
            for vi in range(graphSize):
                sump = 0
                for j in range(r):
                    sump += ((p[va][j] - p[vi][j])**2)
                temp[vi] = dai[vi]**2 - sump
            vc = int(np.argmax(temp))
            if vc == vb:
                break
            else:
                vb = va
                va = vc
        dai = calAllDist(pix, va, m, n)
        dib = calAllDist(pix, vb, m, n)

        #visual
        x1 = va // n
        y1 = va % n
        x2 = vb // n
        y2 = vb % n
        tempimg = cv2.cvtColor(pix, cv2.COLOR_GRAY2BGR)
        tempimg[x1,y1] = [0,0,255]
        tempimg[x2, y2] = [255, 0, 0]
        resizeimage = cv2.resize(tempimg, (500,500))
        cv2.imshow('draw', resizeimage)
        cv2.waitKey(0)

        sump = 0
        for j in range(r):
            sump += ((p[va][j] - p[vb][j]) ** 2)
        dnodeab = dai[vb]**2 - sump
        print(dnodeab)
        if dnodeab < e:
            break
        for vi in range(graphSize):
            sump = 0
            for j in range(r):
                sump += ((p[va][j] - p[vi][j]) ** 2)
            dnodeai = dai[vi] ** 2 - sump
            sump = 0
            for j in range(r):
                sump += ((p[vi][j] - p[vb][j]) ** 2)
            dnodeib = dib[vi] ** 2 - sump
            p[vi][r] = (dnodeai + dnodeab - dnodeib)/(2 * math.sqrt(dnodeab))
    print(p.shape)
    return p

