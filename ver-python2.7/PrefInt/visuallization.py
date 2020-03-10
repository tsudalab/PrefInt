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

"""
This visuallization is only used for example project for now.
"""

from __future__ import division

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt



def gperrboxplot(egp,esgp,ngp=None,Element=None):

    if ngp is None:
        data = [egp,esgp]
    else:
        data = [egp,esgp,ngp]
    data = np.array(data)
    data = data.T

    if ngp is None:
        df = pd.DataFrame(data,columns=['EXP','EXP&SIM '])
    else:
        df = pd.DataFrame(data,columns=['ExpGP','Exp&Sim GP','Normal GP'])
    print 'The prediction accuracy of',Element
    plt.figure(figsize=(7, 5), dpi=300)
    plt.title('Predicted Ranking Accuracy (Molecular Wavelength 14/80)',fontsize=14)
    plt.tick_params(labelsize=15)   #size control of tick
    f = df.boxplot(return_type='dict',patch_artist =True)
    colors = ['cornflowerblue','lightcoral']
    #for bplot in df:
    #    for patch, color in zip(bplot['boxes'],colors):
    #        patch.set_facecolor(color)
    for box, colors in zip(f['boxes'],colors):
        box.set(color ='dimgrey',linewidth=1)
        box.set(facecolor = colors,alpha=0.99)
        
    for median in f['medians']:
        median.set(color='dimgrey', linewidth=1.5)    
    for cap in f['caps']:
        cap.set(color='black', linewidth=2)
    plt.savefig('./result/molwlpg_80.png')
    plt.show()

def gperrvioplot(egp,esgp,ngp=None,Element=None):
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    if ngp is None:
        data = [egp,esgp]
    else:
        data = [egp,esgp,ngp]
    data = np.array(data)
    data = data.T
    #print data
    #if ngp is None:
    #    df = pd.DataFrame(data,columns=['EXP','EXP&SIM'])
    #else:
    #    df = pd.DataFrame(data,columns=['ExpGP','Exp&Sim GP','Normal GP'])
    print 'The prediction accuracy of',Element
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    plt.title('Predicted Ranking Accuracy (Molecular Wavelength)',fontsize=14)
    plt.tick_params(labelsize=15)   #size control of tick
    f = ax.violinplot(data,showmeans=False,showmedians=False,showextrema=False)
    #print f
    colors = ['cornflowerblue','lightcoral']
    #for bplot in df:
    #    for patch, color in zip(bplot['boxes'],colors):
    #        patch.set_facecolor(color)
    for box, colors in zip(f['bodies'],colors):
        box.set(color ='dimgrey',linewidth=1)
        box.set(facecolor = colors,alpha=0.99)
    
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1,len(medians)+1)
    #print medians
    
    ax.scatter(inds, medians, marker='^', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    #ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    labels =['EXP','EXP&SIM']
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.yaxis.grid(True)
    #ax.set_ylim()
    ax.set_xticklabels(labels)
    plt.savefig('./result/moleculargp_vio.png')
    plt.show()

def boitrboxplot(ebo,esbo,nbo=None,Element=None):

    if nbo is None:
        data = [ebo,esbo]
    else:
        data = [ebo,esbo,nbo]
    data = np.array(data)
    data = data.T

    if nbo is None:
        df = pd.DataFrame(data,columns=['ExpBO','Exp&Sim BO'])
    else:
        df = df = pd.DataFrame(data,columns=['ExpBO','Exp&Sim BO','Normal BO'])
    print 'The Bayesian Optimization of ',Element
    plt.figure(figsize=(12, 9), dpi=80)
    plt.title('Iterations of getting the maximum',fontsize=20)
    plt.tick_params(labelsize=15)   #size control of tick
    f = df.boxplot(return_type='dict',patch_artist =True)
    for box in f['boxes']:
        box.set(color ='b',linewidth=2)
        box.set(facecolor = 'b',alpha=0.2)
        
    for median in f['medians']:
        median.set(color='DarkGreen', linewidth=2)    
    for cap in f['caps']:
        cap.set(color='black', linewidth=2)
    plt.show()

def boitrsucrate(ebo,esbo,itr_interval,n):

    #itr_interval = 1
    #n = 10
    esucrate = list()
    essucrate = list()
    ecount = 0

    for i in range(n):
        for sn in ebo:
            if itr_interval*i+1 <= sn < itr_interval*(i+1)+1:
                ecount+=1
        #print ecount
        esucrate.append(ecount/len(ebo)*100)
    #print esucrate

    ecount = 0

    for i in range(n):
        for sn in esbo:
            if itr_interval*i+1 <= sn < itr_interval*(i+1)+1:
                ecount+=1
        #print ecount
        essucrate.append(ecount/len(esbo)*100)
    #print essucrate

    xplot = np.arange(itr_interval,itr_interval*n+1,step=itr_interval)
    #xplot = np.arange(24,102, step=4)
    plt.figure(figsize=(7,5),dpi=300)
    plt.plot(xplot,esucrate[-20:],label='EXP',color='cornflowerblue',marker='^')
    plt.plot(xplot,essucrate[-20:],label='EXP&SIM',color='lightcoral',marker ='s')
    plt.title('Success rate after n iterations(Molecular Wavelength 14/80',fontsize=14)
    plt.tick_params(labelsize=15)
    plt.xlabel('the number of iterations',fontsize=16)
    plt.ylabel('Success rate %',fontsize=16)
    plt.legend(loc='lower right')
    #xplot = np.arange(24,102,step=8)
    plt.xticks(xplot)
    plt.savefig('./result/molwlbo_80.png')
    plt.show()









