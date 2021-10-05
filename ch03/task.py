# -*- coding: UTF-8 -*-
#上面一行表示支持中文/The above line indicates support for Chinese

#计算信息熵
import trees
my_data,labels=trees.createDataSet()
trees.chooseBestFeatureToSplit(my_data)
