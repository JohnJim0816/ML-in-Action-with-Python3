'''
@Author: Jiang Ji
@Email: hellojiangji@gmail.com
@Date: 2019-11-20 13:11:20
@LastEditor: Jiang Ji
@LastEditTime: 2020-04-23 17:17:31
@Discription: 
@Environment: python 3.7.7
'''
# coding=utf-8
import logRegress


def load_dataset():
    dataMat=[];labelMat=[]
    fr=open('test_set.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat #dataMat[100*3], labeMat[100*1]

def plotBestFit(weights):
    import matplotlib.pyplot as plt 
    import numpy as np
    #weights=weights.getA()
    dataMat,labelMat=load_dataset()
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def gradAscent(): 
    dataMat,labelMat=load_dataset()
    weights=logRegress.gradAscent(dataMat,labelMat)
    plotBestFit(weights.getA())

def stocGradAscent0():
    dataMat,labelMat=load_dataset()
    weights=logRegress.stocGradAscent0(dataMat,labelMat)
    plotBestFit(weights)

def stocGradAscent1():
    dataMat,labelMat=load_dataset()
    weights=logRegress.stocGradAscent1(dataMat,labelMat)
    plotBestFit(weights)
def colicTest():
    logRegress.colic_load_dataset()

def default():
    print("Invalid Operation!")

switch={
    1:gradAscent,
    2:stocGradAscent0,
    3:stocGradAscent1,
    4:colicTest
}

if __name__ == "__main__":
    task = 3
    switch.get(task,default)()