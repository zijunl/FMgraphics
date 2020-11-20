## L2FM.py import the single 2D iamge graph and apply FastMap L2 to embeded it to higher dimension


import networkx as nx
import numpy as np
import math
import cv2


def image2graph(img_raw):
    m = img_raw.shape[0]
    n = img_raw.shape[1]
    size = m * n
    G = nx.Graph()
    #print(size)

    for vi in range(size):
        row = vi//n
        col = vi%n
        if row+1<m:
            pixel1 = img_raw[row][col]
            pixel2 = img_raw[row+1][col]
            diff = pixel1 - pixel2
            dist = math.sqrt(sum(diff*diff))
            G.add_edge(vi, vi+n, weight = dist)
        if col+1<n:
            pixel1 = img_raw[row][col]
            pixel2 = img_raw[row][col+1]
            diff = pixel1 - pixel2
            dist = math.sqrt(sum(diff*diff))
            G.add_edge(vi, vi+1, weight = dist)
    return G



def l2fm(input,K,e):
    G = image2graph(input)
    graphSize = len(G)
    #print(graphSize)
    C = G.size()

    p = np.zeros((graphSize,K))
    for r in range(K):
        va = 0                          # randomly chosen
        vb = va
        for t in range(C):
            dai = nx.single_source_dijkstra_path_length(G, va)
            temp = np.zeros(graphSize)
            for vi in range(graphSize):
                sump = sum((p[va][0:r] - p[vi][0:r]) * (p[va][0:r] - p[vi][0:r]))
                temp[vi] = dai[vi]**2 - sump
            vc = np.argmax(temp)
            if vc == vb:
                break
            else:
                vb = va
                va = vc
        dai = nx.single_source_dijkstra_path_length(G, va)
        dib = nx.single_source_dijkstra_path_length(G, vb)

        sump = sum((p[va][0:r] - p[vb][0:r]) * (p[va][0:r] - p[vb][0:r]))
        #dnodeab = dai[va][vb]**2 - sump
        dnodeab = dai[vb]**2 - sump
        #print(dnodeab)
        if dnodeab < e:
            break
        for vi in range(graphSize):
            sump = sum((p[va][0:r] - p[vi][0:r]) * (p[va][0:r] - p[vi][0:r]))
            dnodeai = dai[vi] ** 2 - sump
            sump = sum((p[vi][0:r] - p[vb][0:r]) * (p[vi][0:r] - p[vb][0:r]))
            dnodeib = dib[vi] ** 2 - sump
            p[vi][r] = (dnodeai + dnodeab - dnodeib)/(2 * math.sqrt(dnodeab))
    #print(p.shape)
    return p

if __name__=='__main__':
    #path_input = 'C:/Users/qingh/Desktop/USC/EE 599/final_project/img_align_celeba/img_align_celeba/000001.jpg'
    path_input = 'C:/Users/qingh/Desktop/USC/Research/FastMap/face100.jpg'
    K = 5
    e = 10
    p = l2fm(path_input,K,e)