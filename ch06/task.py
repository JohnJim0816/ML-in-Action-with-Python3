import svmMLiA

task = 1

def load_dataset(): 
    dataArr,labelArr=svmMLiA.load_dataset('testSet.txt')
    print(labelArr)

def stocGradAscent0():
  

def stocGradAscent1():
    
def colicTest():
    

def default():
    print("Invalid Operation!")

switch={
    1:load_dataset,
    2:stocGradAscent0,
    3:stocGradAscent1,
    4:colicTest
}

switch.get(task,default)()