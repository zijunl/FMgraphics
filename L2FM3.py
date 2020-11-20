## this script try a single image and transform a 3D color image to a graph(V,E)

import networkx as nx
import numpy as np
import cv2
import math
import cv2

K = 3


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


def L2FM3(file):
    weight = 1
    raw = cv2.imread(file)
    m, n, chan = raw.shape
    graphSize = m * n
    blue = np.zeros(shape=(m, n))
    red = np.zeros(shape=(m, n))
    green = np.zeros(shape=(m, n))

    # normalize

    for i, p in enumerate(raw):
        for j, pj in enumerate(p):
            blue[i][j] = pj[0] * weight * math.sqrt(m * n) / 255
            red[i][j] = pj[1] * weight * math.sqrt(m * n) / 255
            green[i][j] = pj[2] * weight * math.sqrt(m * n) / 255


    e = 1
    C = (m*n)**2


    pb = np.zeros((graphSize, K+1))
    pr = np.zeros((graphSize, K+1))
    pg = np.zeros((graphSize, K+1))



    for r in range(K):
        va = 0                       # randomly chosen
        vb = va
        for t in range(C):
            dai = calAllDist(blue, va, m, n)
            temp = np.zeros(graphSize)
            for vi in range(graphSize):
                sump = 0
                for j in range(r):
                    sump += ((pb[va][j] - pb[vi][j])**2)
                temp[vi] = dai[vi]**2 - sump
            vc = int(np.argmax(temp))
            if vc == vb:
                break
            else:
                vb = va
                va = vc
        dai = calAllDist(blue, va, m, n)
        dib = calAllDist(blue, vb, m, n)

        sump = 0
        for j in range(r):
            sump += ((pb[va][j] - pb[vb][j]) ** 2)
        dnodeab = dai[vb]**2 - sump
        print(dnodeab)
        if dnodeab < e:
            break
        for vi in range(graphSize):
            sump = 0
            for j in range(r):
                sump += ((pb[va][j] - pb[vi][j]) ** 2)
            dnodeai = dai[vi] ** 2 - sump
            sump = 0
            for j in range(r):
                sump += ((pb[vi][j] - pb[vb][j]) ** 2)
            dnodeib = dib[vi] ** 2 - sump
            pb[vi][r] = (dnodeai + dnodeab - dnodeib)/(2 * math.sqrt(dnodeab))

    for r in range(K):
        va = 0                       # randomly chosen
        vb = va
        for t in range(C):
            dai = calAllDist(red, va, m, n)
            temp = np.zeros(graphSize)
            for vi in range(graphSize):
                sump = 0
                for j in range(r):
                    sump += ((pr[va][j] - pr[vi][j])**2)
                temp[vi] = dai[vi]**2 - sump
            vc = int(np.argmax(temp))
            if vc == vb:
                break
            else:
                vb = va
                va = vc
        dai = calAllDist(red, va, m, n)
        dib = calAllDist(red, vb, m, n)

        sump = 0
        for j in range(r):
            sump += ((pr[va][j] - pr[vb][j]) ** 2)
        dnodeab = dai[vb]**2 - sump
        print(dnodeab)
        if dnodeab < e:
            break
        for vi in range(graphSize):
            sump = 0
            for j in range(r):
                sump += ((pr[va][j] - pr[vi][j]) ** 2)
            dnodeai = dai[vi] ** 2 - sump
            sump = 0
            for j in range(r):
                sump += ((pr[vi][j] - pr[vb][j]) ** 2)
            dnodeib = dib[vi] ** 2 - sump
            pr[vi][r] = (dnodeai + dnodeab - dnodeib)/(2 * math.sqrt(dnodeab))

    for r in range(K):
        va = 0                       # randomly chosen
        vb = va
        for t in range(C):
            dai = calAllDist(green, va, m, n)
            temp = np.zeros(graphSize)
            for vi in range(graphSize):
                sump = 0
                for j in range(r):
                    sump += ((pg[va][j] - pg[vi][j])**2)
                temp[vi] = dai[vi]**2 - sump
            vc = int(np.argmax(temp))
            if vc == vb:
                break
            else:
                vb = va
                va = vc
        dai = calAllDist(green, va, m, n)
        dib = calAllDist(green, vb, m, n)

        sump = 0
        for j in range(r):
            sump += ((pg[va][j] - pg[vb][j]) ** 2)
        dnodeab = dai[vb]**2 - sump
        print(dnodeab)
        if dnodeab < e:
            break
        for vi in range(graphSize):
            sump = 0
            for j in range(r):
                sump += ((pg[va][j] - pg[vi][j]) ** 2)
            dnodeai = dai[vi] ** 2 - sump
            sump = 0
            for j in range(r):
                sump += ((pg[vi][j] - pg[vb][j]) ** 2)
            dnodeib = dib[vi] ** 2 - sump
            pg[vi][r] = (dnodeai + dnodeab - dnodeib)/(2 * math.sqrt(dnodeab))

    for i in range(m):
        for j in range(n):
            pb[i*m+j][K] = blue[i][j]
            pr[i * m + j][K] = red[i][j]
            pg[i * m + j][K] = green[i][j]


    result = np.stack((pb, pr, pg))
    print(result.shape)
    return result

