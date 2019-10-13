import numpy as np
import pandas as pd
import cv2
import os
import dlib
train_path="/Users/adminadmin/Documents/mywork/master/dataset2/original/train/"
public_path="/Users/adminadmin/Documents/mywork/master/dataset2/original/public/"
private_path="/Users/adminadmin/Documents/mywork/master/dataset2/original/private/"
#toimage
'''
A=pd.read_csv("fer2013.csv")#35887 dataset
N=35887
img_size=48*48
#images
for i in range(N):
    str1 = A.pixels[i]
    use=A.Usage[i]
    list1=str1.split( )
    #print list
    nparray=np.array(list1)
    img=nparray.reshape([48,48])
    img=img.astype(np.int16)
    #print img
    img3c=np.zeros([48,48,3])
    for i1 in range(48):
        for j1 in range(48):
            img3c[i1][j1][0]=img[i1][j1]
            img3c[i1][j1][1] = img[i1][j1]
            img3c[i1][j1][2] = img[i1][j1]
    if(use=="Training"):
        save_path = train_path
    if (use == "PublicTest"):
        save_path = public_path
    if (use == "PrivateTest"):
        save_path = private_path

    label=A.emotion[i]
    save_path=save_path+str(label)+"/"+str(i)+".png"
    cv2.imwrite(save_path, img3c)
    print (save_path)

#histrogram equliazion
def histoe(path1,path2):
    for i in range(7):
        path1_i = path1 + str(i) + "/"
        path2_i = path2 + str(i) + "/"
        for filename in os.listdir(path1_i):
            src = path1_i + filename
            dir = path2_i + filename
            img = cv2.imread(src, 0)
            e = cv2.equalizeHist(img)
            cv2.imwrite(dir, e)
            print(dir)

path1=train_path
path2="/Users/adminadmin/Documents/mywork/master/dataset2/histoe/train/"
histoe(path1,path2)
path1=public_path
path2="/Users/adminadmin/Documents/mywork/master/dataset2/histoe/public/"
histoe(path1,path2)

path1=private_path
path2="/Users/adminadmin/Documents/mywork/master/dataset2/histoe/private/"
histoe(path1,path2)
'''
#catch face
def catchfacet(path,path1):
    print("======================catch face======================")
    for i in range(7):
        path_i = path + str(i) + "/"
        for filename in os.listdir(path_i):
            image_path = path_i + filename
            try:
                img0 = cv2.imread(image_path)
                img = cv2.resize(img0, (256, 256))
                detector = dlib.get_frontal_face_detector()
                face = detector(img, 1)
            except:
                print(image_path + "is wrong")
            else:
                for k, d in enumerate(face):
                    height = d.bottom() - d.top()
                    width = d.right() - d.left()
                    try:
                        face_matrix = np.zeros([height, width, 3], np.uint8)
                        for m in range(height):
                            for n in range(width):
                                face_matrix[m][n][0] = img[d.top() + m][d.left() + n][0]
                                face_matrix[m][n][1] = img[d.top() + m][d.left() + n][1]
                                face_matrix[m][n][2] = img[d.top() + m][d.left() + n][2]
                    except:
                        print(image_path + "is wrong")

                    else:
                        save_path = path1 + str(i) + "/" + "c" + filename
                        print(save_path)
                        cv2.imwrite(save_path, cv2.resize(face_matrix, (48, 48)))

def copy1(path1,path2):
    print("======================copy======================")
    for i in range(7):
        path1_i = path1 + str(i) + "/"
        path2_i = path2 + str(i) + "/"
        for filename in os.listdir(path1_i):
            src = path1_i + filename
            dir = path2_i + filename
            img = cv2.imread(src)
            cv2.imwrite(dir, img)
            print(dir)
print ("train:")
path="/Users/adminadmin/Documents/mywork/master/dataset2/histoe/train/"
path1="/Users/adminadmin/Documents/mywork/master/dataset2/crop/train/"
copy1(path,path1)
catchfacet(path,path1)

print ("public:")
public_path="/Users/adminadmin/Documents/mywork/master/dataset2/histoe/public/"
public_path1="/Users/adminadmin/Documents/mywork/master/dataset2/crop/public/"
copy1(public_path,public_path1)
catchfacet(public_path,path1)

print ("private:")
private_path="/Users/adminadmin/Documents/mywork/master/dataset2/histoe/private/"
private_path1="/Users/adminadmin/Documents/mywork/master/dataset2/crop/private/"
copy1(private_path,private_path1)
catchfacet(public_path,path1)