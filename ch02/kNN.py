# -*- coding: UTF-8 -*-
#上面一行表示支持中文/The above line indicates support for Chinese
from numpy import * #引入科学计算包numpy
import operator #引入运算符模块
"""
   实例1：kNN算法小测试
   Example 1: kNN alghrithm test
   简介：给定已分为A、B两类的四个点，对待测试的点进行分类
"""
#定义函数生成数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) #生成四个二维坐标点
    labels = ['A','A','B','B'] #分别代表A、A、B、B类
    return group, labels
#定义分类函数(kNN分类算法)
#参数:inX--待分类点，dataSet--上面生成的group数据集，
#     labels--对应的分类标签，k--选择最临近点的数目
def classify0(inX, dataSet, labels, k):
    #距离计算
    dataSetSize = dataSet.shape[0]  #得出数据集大小=4
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #算出各点与待分类点inX的距离差
    sqDiffMat = diffMat**2 #距离差的平方
    sqDistances = sqDiffMat.sum(axis=1) #矩阵的每个行向量相加合并成一 列
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #从小到大排序    
    classCount={} 
    #选择距离最小的k个点         
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
#打印实例结果
'''group,label=createDataSet()
result=classify0([0,0],group,label,3)
print(result)'''

"""
   实例2：改进约会网站的配对结果
   简介:根据男性的某三个特在来预测是海伦很喜欢、一般、不喜欢的人
       原始数据存于"datingTestSet.txt"中，每一行一个样本数据，总共1000行
       每行分别代表每年飞行里程数、玩游戏所占时间比、每周消费冰淇淋公升数以及结果(很喜欢largeDoses、一般smallDoses、不喜didntLike)
       并且用3、2、1分别代表很喜欢、一般和不喜欢处理得到文本datingTestSet2.txt
"""
#函数功能：将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines() #得到文件行数
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#利用file2matrix函数将文本数据转出为数据矩阵以及分类标签
#注：每个定义的函数都可以在终端运行，具体参见书上P18页，这里是为了更加简便，只需要终端运行python kNN.py即可
#datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')

#制作原始数据的散点图，这里使用的是第二、三列数据，即玩游戏所占时间比和每周消费冰淇淋公升数
'''import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15*array(datingLabels),15*array(datingLabels))
plt.xlabel("time ratio of playing game") 
plt.ylabel("liter of consuming icecream per week")
plt.show()'''

#定义归一化函数
#原理:newValue=(oldValue-min)/(max-min),归一化为0～1之间
def autoNorm(dataSet):
    minVals=dataSet.min(0) #.min()所有的最小值；.min(0)每列的最小值;.min(1)每行的最小值
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

#利用file2matrix函数将文本数据转出为数据矩阵以及分类标签
#datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
#将生成的数据矩阵归一化输出归一化后的数据矩阵，原有矩阵最大最小值之差和最小值
#normMat,ranges,minVals=autoNorm(datingDataMat)

#函数功能：测试分类器效果
#返回值：处理数据集的错误率
#原理：以数据集后半部分(如当hoRatio=0.5时就是后面一半，具体参见100行)作为参考点，利用kNN算法推测前半部分的数据并与实际数据比较，计算错误率
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0] 
    #print(m)
    print(normMat[1,:])
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    #print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    #print errorCount
#运行测试分类器效果的函数
#datingClassTest()

#函数名称：约会网站预测函数
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input("percentage of time spent playing video games?"))
    ffMiles=float(raw_input("frequent flier miles earned per year?"))
    iceCream=float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person:" , resultList[classifierResult-1]
#运行约会网站预测函数
#classifyPerson()\

#函数功能：将图像格式处理为一个向量
def img2vector(filename):
    words_vect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            words_vect[0,32*i+j] = int(lineStr[j])
    return words_vect

'''
testVector=img2vector('testDigits/0_13.txt')
print(testVector) #直接打印不能显示全部，可以执行下面两步一部分一部分看
print(testVector[0,0:31])
'''
from os import listdir #引用os模块中的listdir函数
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
handwritingClassTest()