'''
@Author: Jiang Ji
@Email: hellojiangji@gmail.com
@Date: 2019-11-20 13:11:19
@LastEditor: Jiang Ji
@LastEditTime: 2020-04-24 00:38:29
@Discription: 
@Environment: python 3.7.7
'''
# coding=utf-8 
import numpy as np



def sigmoid(inX):
    '''sigmoid函数
    '''
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMat,labelMat):
    '''批量梯度上升
    '''
    dataMat=np.mat(dataMat)
    labelMat=np.mat(labelMat).transpose()
    m,n=np.shape(dataMat) # m cannot be deleted
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1)) #weights[3*1]
    for k in range(maxCycles):
        h=sigmoid(dataMat*weights)
        error=(labelMat-h) #error[100*1]
        weights=weights+alpha*dataMat.transpose()*error
    print(weights)
    return weights

def stocGradAscent0(dataMat,labelMat):
    '''随机梯度上升
    '''
    m,n=np.shape(dataMat) #dataMat[100*3]
    alpha=0.01
    weights=np.ones(n) #weights[1*3]
    dataArr=np.array(dataMat)
    for i in range(m):
        h=sigmoid(sum(dataMat[i]*weights)) # dataMat[i]表示一个样本
        error=labelMat[i]-h
        weights=weights+alpha*error*dataArr[i]
    return weights

def stocGradAscent1(dataMat,labelMat,iteration=150):
    '''改进的随机梯度上升
    '''
    m,n=np.shape(dataMat)
    weights=np.ones(n)
    dataArr=np.array(dataMat)
    for j in range(iteration):
        dataIndex=dataMat.copy() # initially dataIndex=range(m),but a range cannot be operated by del() at line 75
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01 #apha decreases with iteration, does not
            randIndex=int(np.random.uniform(0,len(dataIndex))) #go to 0 because of the constant
            h=sigmoid(sum(dataMat[randIndex]*weights))
            error=labelMat[randIndex]-h
            weights=weights+alpha*error*dataArr[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5: return 1.0
    else: return 0.0

def colic_load_dataset():
    frTrain=open('horseColicTraining.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    frTest=open('horseColicTest.txt')

def colicTest():
    
    trainingWeights=stocGradAscent1(trainingSet,trainingLabels,500)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainingWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" %errorRate)
    return errorRate

def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))






    
