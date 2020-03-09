# Copyright (C) 2020 Tsuda Laboratory
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
import pandas as pd

from PrefInt.preference_generate import generate_pair
from PrefInt.ibo.gaussianprocess import PrefGaussianProcess
from PrefInt.ibo.gaussianprocess.kernel import GaussianKernel_iso
from PrefInt.candidate import predict_old
from PrefInt.ibo.acquisition import EI

import random
import time

class BOpred ():
    def __init__(self, candidates,simdata):
        self.candidates = candidates
        self.simdata = simdata
        self.Besty = list()
        self.CurrentBesty = list()
        self.Bestact = list()
        self.CurrentBestact = list()
    def predict(self,maxitr=100,stopwhenbest=0):

        #Randomly select 2 data in the candidate dataset as initial data
        #for testing, we here avoid selecting the known maximum point
        timest=time.time()
        expdata = self.candidates.copy()
        etrind = random.sample(range(len(expdata)),2)
        explist = expdata[:,-1].tolist()
        for i in range(100):
            if etrind[0] != explist.index(max(explist)):
                if etrind[1] != explist.index(max(explist)):
        #            print 'good sampling',i
                    break
            else:
        #        print etrind, explist.index(max(explist))
                etrind = random.sample(range(len(expdata)),2) 

        #print 'start iteration...'
        def dataformat(data,trindex):
            n = len(trindex)
            array = np.zeros(shape=(n,data.shape[1]))
            j = 0
            for i in trindex:
                array[j] = data[i,:]
                j+=1
            tx = array[:,:-1]
            ty = array[:,-1]
            return array
        etraindata= dataformat(expdata,etrind)
        predata = np.delete(expdata,etrind,axis=0)
        maxy = max(predata[:,-1])

        #Start generating pairwise preference
        GN = generate_pair(etraindata)
        prefs = GN.firstgen()

        GN = generate_pair(self.simdata)
        simprefs = GN.firstgen()
        #Avoid conflicting pairs
        newsimprefs = simprefs[:]
        for i in range(len(prefs)):
            for j in range(len(simprefs)):
                mi = prefs[i]
                ni = simprefs[j]
                if all(mi[0] == ni[1]) and all(mi[1] == ni[0]):
                    newsimprefs[j] = (mi[0],mi[1],0)
        prefs.extend(newsimprefs)
        print len(prefs),"pairs generated"

        #Start train the model
        print "start training......"
        kernel = GaussianKernel_iso(np.array([18.0]))
        GP = PrefGaussianProcess(kernel)
        GP.addPreferences(prefs)

        iteration = maxitr
        besty = float('-inf')
        selectedind = etrind[:]
        indi = None
        for itr in range(iteration):
            print "start iteration......"
            ei = EI(GP,xi=0.05)
            newdata, bestind = predict_old(ei,expdata,selectedind)
            if newdata[-1] > besty:
                bestcan = newdata[:-1]
                besty = newdata[-1]
                bestact = bestind
            #print itr+1,newdata[-1],bestind
            print "select y is", newdata[-1],"current best y is", besty, "the indicator is", bestact
            selectedind.append(bestind)
            self.Besty.append(newdata[-1])
            self.CurrentBesty.append(besty)
            self.Bestact.append(bestind)
            self.CurrentBestact.append(bestact)
            lenprefs = list()
            fittime = list()    
            if len(selectedind) == len(expdata):
                break
            if stopwhenbest == 1:  
                if newdata[-1] == maxy:
                    self.indi = itr + 1
                    break
            print "try next point......"
            newprefs = GN.addnew(newdata)
            GP.addPreferences(newprefs)
            #print '---DEBUG---len(GP.Preferences)',len(GP.preferences)

            #print '---DEBUG---',apend - apstart, 's'
            lenprefs.append(len(GP.preferences))
        #error = max(expdata[:,-1]) - besty
        timeed=time.time()
        print "taking time", timeed-timest,"s"
        return bestcan,besty
    def showall(self):
        print "history function values:"
        print self.Besty
        print "history best function values:"
        print self.CurrentBesty
        print "history act:"
        print self.Bestact
        print "history of best act"
        print self.CurrentBestact
