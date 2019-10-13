import cv2
import torch
import torch.utils.data
import numpy as np
import csv
path="/Users/adminadmin/Documents/mywork/master/dataset2/train.txt"
path1="/Users/adminadmin/Documents/mywork/master/dataset2/public.txt"
path2="/Users/adminadmin/Documents/mywork/master/dataset2/private.txt"
csvpath="/Users/adminadmin/Documents/mywork/master/dataset1/train.csv"
csvpath1="/Users/adminadmin/Documents/mywork/master/dataset1/test.csv"
savepath="/Users/adminadmin/Documents/mywork/master/cnn/accuracy.txt"
BS=256
class mydataset(torch.utils.data.Dataset):
    def __init__(self,txtpath):
        super(mydataset, self).__init__()
        data=np.loadtxt(txtpath,dtype=np.str)
        self.data=data

    def __getitem__(self, index):
        imgpath=self.data[index][0]
        img=cv2.imread(imgpath,0)
        #img=cv2.resize(img,(224,224))
        img = cv2.resize(img, (64, 64))
        label=self.data[index][1]
        label=label.astype(np.int)
        return img,label
    def __len__(self):
        return self.data.shape[0]

class mydataset1(torch.utils.data.Dataset):
    def __init__(self,txtpath,csvp):
        super(mydataset1, self).__init__()
        data=np.loadtxt(txtpath,dtype=np.str)
        self.data=data
        self.csvp=csvpath1

    def __getitem__(self, index):
        i=-1
        csv_file = open(self.csvp)
        csv_reader_lines = csv.reader(csv_file)
        for one_line in csv_reader_lines:
            img = np.array(one_line,np.int16)
            img1=np.zeros((224,224),np.int16)
            i = i + 1
            if (i == index):
                img1=cv2.resize(img,(224,224))
                break
        img=img1
        label=self.data[index][1]
        label = label.astype(np.int)
        return img,label
    def __len__(self):
        return self.data.shape[0]

traindataset = mydataset(txtpath=path)
#traindataset=mydataset1(txtpath=path,csvp=csvpath)
print (traindataset.__len__())
train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=BS, shuffle=True)

publicdataset=mydataset(txtpath=path1)
#testdataset=mydataset1(txtpath=path1,csvp=csvpath1)
print (publicdataset.__len__())
public_loader = torch.utils.data.DataLoader(dataset=publicdataset, batch_size=BS, shuffle=True)

privatedataset=mydataset(txtpath=path2)
print (privatedataset.__len__())
private_loader = torch.utils.data.DataLoader(dataset=privatedataset, batch_size=BS, shuffle=True)


class cnn(torch.nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer2=torch.nn.Sequential(
            torch.nn.Conv2d(64,128,3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer3=torch.nn.Sequential(
            torch.nn.Conv2d(128,256,3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer4=torch.nn.Sequential(
            torch.nn.Conv2d(256,512,3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,3,1,1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer5=torch.nn.Sequential(
            torch.nn.Conv2d(512,512,3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer6=torch.nn.Sequential(
            torch.nn.Linear(25088,4096),
            torch.nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Linear(4096, 1000),
            torch.nn.ReLU()
        )
        self.out=torch.nn.Linear(1000,7)

    def forward(self, x):
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x=x.view(x.size(0),-1)
        x = self.layer6(x)
        x = self.layer7(x)
        output=self.out(x)
        return output

class cnn1(torch.nn.Module):
    def __init__(self):
        super(cnn1,self).__init__()
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(1,6,5,1,0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer2=torch.nn.Sequential(
            torch.nn.Conv2d(6,6,5,1,0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer3=torch.nn.Sequential(
            torch.nn.Conv2d(6,16,5,1,0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer4=torch.nn.Sequential(
            torch.nn.Conv2d(16,120,4,1,0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.layer5=torch.nn.Sequential(
            torch.nn.Linear(120,84),
            torch.nn.ReLU()
        )
        self.out=torch.nn.Linear(84,7)

    def forward(self, x):
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=x.view(x.size(0),-1)
        x = self.layer5(x)
        output=self.out(x)
        return output

vgg16=cnn1()
print (vgg16)

optimizer=torch.optim.Adam(vgg16.parameters(),lr=0.0001)
loss_func=torch.nn.CrossEntropyLoss()
times=0
file=open(savepath,"w")
print("epoch i trainacc pubtacc priacc loss")
file.write("epoch i trainacc pubtacc priacc loss\n")
for epoch in range(10):
    i=0
    for input, label in train_loader:
        #input1=torch.autograd.Variable(input.view((-1,1,224,224))).float()
        input1 = torch.autograd.Variable(input.view((-1, 1, 64, 64))).float()
        output=vgg16(input1)
        loss=loss_func(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predict = torch.max(output,1)[1].data.squeeze()
        accuracy = float((predict.numpy() == label.numpy()).sum()) / BS
        i = i + 1

        avg=0
        count=0
        for publicinput, publiclabel in public_loader:
            count = count + 1
            print (count, end=" ")
            #publicinput1 = torch.autograd.Variable(publicinput.view((-1, 1, 224, 224))).float()
            publicinput1 = torch.autograd.Variable(publicinput.view((-1, 1, 64, 64))).float()
            publicoutput = vgg16(publicinput1)
            publicpredict = torch.max(publicoutput,1)[1].data.squeeze()
            publicaccuracy = float((publicpredict.numpy() == publiclabel.numpy()).sum()) /BS
            avg=(avg+publicaccuracy)
            if(count==10):
                break
        avg=avg/count

        avg1 = 0
        count = 0
        for privateinput, privatelabel in public_loader:
            count = count + 1
            print(count, end=" ")
            #privateinput1 = torch.autograd.Variable(privateinput.view((-1, 1, 224, 224))).float()
            privateinput1 = torch.autograd.Variable(privateinput.view((-1, 1, 64, 64))).float()
            privateoutput = vgg16(privateinput1)
            privatepredict = torch.max(privateoutput, 1)[1].data.squeeze()
            privateccuracy = float((privatepredict.numpy() == privatelabel.numpy()).sum()) / BS
            avg1 = (avg1 + privateccuracy)
            if (count == 10):
                break
        avg1 = avg1 / count
        print(" ")
        print(str(epoch) + " " + str(i) + " " + str(accuracy) + " " + " " + str(avg)+" "+str(avg1)+" "+str(loss.detach().numpy()))
        file.write(str(epoch) + " " + str(i) + " " + str(accuracy) + " " + " " + str(avg)+" "+str(avg1)+" "+str(loss.detach().numpy())+"\n")
file.close()
torch.save(vgg16,'net.pkl')





