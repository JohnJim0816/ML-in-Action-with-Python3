import numpy as np
from math import sqrt, pi, exp
means={}
means['Beagle']=np.array([41,37,10])
means['Corgi']=np.array([53,27,12])
means['Husky']=np.array([66,55,22])
means['Poodle']=np.array([61,52,26])

stds={}
stds['Beagle']=np.array([6,4,2])
stds['Corgi']=np.array([9,3,2])
stds['Husky']=np.array([10,6,6])
stds['Poodle']=np.array([9,7,8])

possible_labels = ['Beagle','Corgi','Husky','Poodle']
prior_probs={}
prior_probs['Beagle']=0.30
prior_probs['Corgi']=0.21
prior_probs['Husky']=0.14
prior_probs['Poodle']=0.35
def gaussian_prob(obs, mu, sig):
    num = (obs - mu) ** 2
    denum = 2 * sig ** 2
    norm = 1 / sqrt(2 * pi * sig ** 2)
    return norm * exp(-num / denum)

def predict_probs(obs):
    """
    Private method used to assign a probability to each class.
    """
    probs = {}
    for label in possible_labels:
        product = 1.00
        for feature in range(3):
            product *= gaussian_prob(obs[feature], means[label][feature], stds[label][feature])
        product *= prior_probs[label]    
        probs[label] = product
    """probs is a dictionary containing the probabilities of the observation for each class.
    The class labels are the keys with the values being the respective probabilities for that class."""
    data = np.array(list(probs.values()))
    probs= data/np.sum(data)
    return probs

print(predict_probs([59, 32, 17]))