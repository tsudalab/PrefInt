#This is the accuracy evaluation method of gaussian process preference learning 

import numpy as np

def ndcg(mu, test, printall=0):
    """
    This is to evaluate the ranking accuracy of the test data using 
    Normalized Discounted Cumulative Gain.
    """
    newtest = np.c_[test[:,:-1],mu]
    n = len(test)

    datasort = np.c_[newtest[:,-1],test]
    datasort = datasort[np.lexsort(-datasort.T)]
    ra = np.arange(1,n+1)
    datasort = np.c_[datasort,ra]
    datasort = np.c_[datasort,datasort[:,0]]
    datasort = datasort[np.lexsort(-datasort.T)]
    predrank = datasort[:,-2]
    c = len(predrank)
    idcg = 0
    dcg = 0

    for i in range (1,c+1):
        idcg += (c - i)/(np.log10(i+1))
        dcg += (c - predrank[i-1])/(np.log10(i+1))
    error = dcg/idcg



    if printall==1:
        print 'predicted ranking is', predrank

        print 'Normalized Discounted Cumulative Gain is', error

    return error