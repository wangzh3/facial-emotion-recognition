import os

def totxt(path,path1):
    file = open(path1, "w")
    for i in range(7):
        src = path + str(i) + "/"
        for filename in os.listdir(src):
            img_path = src + filename
            print(img_path)
            file.write(img_path + " " + str(i) + "\n")

    file.close()

path="/Users/adminadmin/Documents/mywork/master/dataset2/crop/train/"
path1="/Users/adminadmin/Documents/mywork/master/dataset2/train.txt"
totxt(path,path1)

path="/Users/adminadmin/Documents/mywork/master/dataset2/crop/public/"
path1="/Users/adminadmin/Documents/mywork/master/dataset2/public.txt"
totxt(path,path1)

path="/Users/adminadmin/Documents/mywork/master/dataset2/crop/private/"
path1="/Users/adminadmin/Documents/mywork/master/dataset2/private.txt"
totxt(path,path1)