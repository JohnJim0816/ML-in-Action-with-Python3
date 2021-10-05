import numpy as np
def load_posting_list():
    posting_list=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classes_list=[0,1,0,1,0,1] # 1代表侮辱性文字，0代表正常言论
    return posting_list,classes_list


def createVocabList(posting_list):
    '''createVocabList 创建一个包含在所有文档中出现的不重复词的列表
    Args:
        posting_list [type]: [description]
    Returns:
        [type]: [description]
    '''
    vocab_set=set([])
    for doc in posting_list:
        vocab_set=vocab_set|set(doc)
    return list(vocab_set)

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else: print("the word: %s is not in my Vocabulary!") 
    return returnVec

def trainNB0(train_mat,trainCategory):
    numWords = len(train_mat[0])
    pAbusive = sum(trainCategory)/float(len(train_mat))
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)      #change to ones() 
    p0Denom = 0.0; p1Denom = 0.0                        #change to 2.0
    for i in range(len(train_mat)):
        if trainCategory[i] == 1:
            p1Num += train_mat[i]
            p1Denom += sum(train_mat[i])
        else:
            p0Num += train_mat[i]
            p0Denom += sum(train_mat[i])
    p1Vect = p1Num/p1Denom          
    p0Vect = p0Num/p0Denom         
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
if __name__ == '__main__':

    posting_list,classes_list=load_posting_list()
    vocab_list=createVocabList(posting_list)
    train_mat= []
    for posting_doc in posting_list:
        train_mat.append(setOfWords2Vec(vocab_list,posting_doc))
    print(train_mat)
    p0V,p1V,pAb=trainNB0(train_mat,classes_list)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(vocab_list, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(vocab_list, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    