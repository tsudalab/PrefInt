# Copyright (C) 2020 by Xiaolin Sun
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
        print ('predicted ranking is', predrank)

        print ('Normalized Discounted Cumulative Gain is', error)

    return error