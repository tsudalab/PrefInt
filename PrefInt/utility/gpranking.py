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

import numpy as np

from PrefInt.preference_generate import generate_pair
from PrefInt.ibo.gaussianprocess import PrefGaussianProcess
from PrefInt.ibo.gaussianprocess.kernel import GaussianKernel_iso
from PrefInt.candidate import predict_old
from PrefInt.ibo.acquisition import EI

from PrefInt.visuallization import gperrboxplot,boitrsucrate


import random
import time

def GPranking(expdata,simdata,test,musig=0):
    #generate pairwise preference of simulated wavelengths
    time1=time.time()
    GN = generate_pair(expdata)
    prefs = GN.firstgen()
    #genereate pairwise preference of HOMO-LUMO bandgap
    GN = generate_pair(simdata)
    simprefs = GN.firstgen()
    #if conflicts happen, trun d = 1 to d = 0
    newsimprefs = simprefs[:]
    for i in range(len(prefs)):
        for j in range(len(simprefs)):
            mi = prefs[i]
            ni = simprefs[j]
            if all(mi[0] == ni[1]) and all(mi[1] == ni[0]):
                newsimprefs[j] = (mi[0],mi[1],0)
    prefs.extend(newsimprefs)
    print len(prefs),"pairs generated"
    print "start training......"
    #Gaussian Process
    kernel = GaussianKernel_iso(np.array([20.0]))
    GP = PrefGaussianProcess(kernel)
    GP.addPreferences(prefs)
    #predict test data with surrogate mean values for ranking
    print "predicting......"
    mu,sig2 = GP.posteriors(test)
    newtest = np.c_[test,mu]
    time2=time.time()
    print "taking time:",time2 - time1,"s"
    if musig == 0:
        return newtest 
    if musig == 1:
        return newtest,mu,sig2
    
