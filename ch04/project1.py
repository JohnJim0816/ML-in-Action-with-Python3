import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加父路径到系统路径sys.path

import bayes
posts_list,labels_list = bayes.load_dataset()
vocab_list = bayes.create_vocab_list(posts_list)
train_mat = []
for post in posts_list:
    train_mat.append(bayes.words2vec(vocab_list,post))
p0V,p1V,PAb = bayes.trainNB0(train_mat,labels_list)
print(p0V,p1V,PAb)
