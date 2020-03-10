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
    