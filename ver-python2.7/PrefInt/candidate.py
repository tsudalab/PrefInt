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


def fitcan(bestX, allcandidates, DEBUG=False):

    #Calculate Distance (Not used)

    distmatrix = list()
    data = allcandidates

    for i in range(0,len(data)):

        dist = np.linalg.norm(bestX-data[i,:-1])
        distmatrix.append(dist)

    bestind = distmatrix.index(min(distmatrix))
    newdata = data[bestind,:]


       # if dist == min(distmatrix):
       #     newdata = data[i]
       #     newbestX = data[i,:-1]
       #     bestindex = i
       # if DEBUG: print (dist,data[i,:-1],newbestX)

    return newdata

def predict(model,expcandidates,simcandidates):

    def _predict(model,allcandidates):

        eires = list()
        data = allcandidates

        for i in range(0,len(data)):

            y = model.f(data[i,:-1])
            eires.append(y)


         #   if y == max(eires):
         #       newdata = data[i]
         #       bestX = data[i,:-1]
         #       bestei = y
    #
        #if showresult == True:
         #       print ('bestX is:', bestX, 'Its EI result is:', bestei)

        bestind = eires.index(max(eires))
        newdata = data[bestind,:]

        return newdata,bestind

    bestexp,bestexpind = _predict(model,expcandidates)
    bestsim,bestsimind = _predict(model,simcandidates)

    if bestexp[-1] >= bestsim[-1]:
        newdata = bestexp 
        bestind = bestexpind
        datatype = 0
    else:
        newdata = bestsim
        bestind = bestsimind
        datatype = 1


    return newdata, bestind, datatype

def predict_old(model,allcandidates,selectedind):
    """
    selectedind means the data has already been used and should not appear again. list type
    """

    eires = list()
    data = allcandidates

    for i in range(0,len(data)):

        y = model.f(data[i,:-1])
        eires.append(y)

    for j in selectedind:
        eires[j] = float('-inf')
     #   if y == max(eires):
     #       newdata = data[i]
     #       bestX = data[i,:-1]
     #       bestei = y
#
    #if showresult == True:
     #       print ('bestX is:', bestX, 'Its EI result is:', bestei)

    bestind = eires.index(max(eires))
    newdata = data[bestind,:]

    return newdata,bestind
    






