# This is to format the bestX predicted by the EI aquisition. For now, this can be done in two ways.
# 1. all candidates exhaustion. This can only be used when the data set is really small.
# 2. caculate the Euclidean distance to the predicted bestX and find the candidate with the smallest distance.

import numpy as np


def fitcan(bestX, allcandidates, DEBUG=False):

    """ This is used to find a candidate which has the smallest distance with the bestX predicted by IBO maximizeEI"""
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
    






