# BuildLabelFor32.py divided the each original image with size 216 * 178 into 30 different 32*32 pieces, After that 
# we label each of the piece as face/Not face based the result of the bounding box of dlib front_face detector. If
# the a 32*32 piece have 75% of its pixel inside the bounding box, we say this piece is a face, vise versa.      

import os
import glob
import cv2
import dlib
import pickle
from dlib import get_frontal_face_detector
from utils import Config

# the findOverlap method return whether the 32*32 piece is face.
# input: bounding box of the front_face detector box; topleft coordinate of this 32 * 32 piece: (r,c)
# output: boolean represent whether this piece is a face

def findOverlap(box,r,c):
    num = 0
    for i in range(32):
        for j in range(32):
            curx = r + i
            cury = c + j
            if curx>=box.top() and curx<box.bottom() and cury>=box.left() and cury<box.right(): 
                num = num + 1
    if num>= int(32 * 32 * 3/4): return True
    return False



if __name__ == "__main__":
    path_input = Config['path_input']
    path_output = Config['path_output'] 
    detector = get_frontal_face_detector()
    index_file = 0
    labels = []
    for filename in glob.glob(os.path.join(path_input, '*.jpg')):
        if index_file%100 == 0:
            print(filename)
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = detector(img)
            if len(boxes)!= 0:
                box = boxes[0]
                r = 13
                c = 8
                index_image = 0
                for i in range(6):
                    for j in range(5):
                        img_crop = img[r+i*32:r+(i+1)*32,c+j*32:c+(j+1)*32]
                        print(img_crop.shape)
                        if findOverlap(box,r+i*32,c+j*32):
                            labels.append(1)
                            #print('1')
                        else:
                            labels.append(0)
                            #print('0')
                        index_image = index_image + 1
                        cv2.imwrite(path_output+filename[-10:-4]+ '_' + str(index_image)+'.jpg',img_crop) 
        index_file = index_file + 1
    pickle_out = open("labels.pickle","wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()

                                                                
