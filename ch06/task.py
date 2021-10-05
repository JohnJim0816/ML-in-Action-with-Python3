import svmMLiA

task = 1

def loadDataSet(): 
    dataArr,labelArr=svmMLiA.loadDataSet('testSet.txt')
    print(labelArr)

def stocGradAscent0():
  

def stocGradAscent1():
    
def colicTest():
    

def default():
    print("Invalid Operation!")

switch={
    1:loadDataSet,
    2:stocGradAscent0,
    3:stocGradAscent1,
    4:colicTest
}

switch.get(task,default)()