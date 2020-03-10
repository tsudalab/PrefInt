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

from numpy import array, log, zeros, exp, sqrt, sum, vstack, ones, arange, clip
from numpy.linalg import norm

    
class Kernel(object):
    """
    base class for kernels.
    """
    def __init__(self, hyperparams):
        self._hyperparams = array(hyperparams)
        self._hyperparams.setflags(write=False)

    # this needs to be read-only, or bad things will happen
    def getHyperparams(self):
        return self._hyperparams

    hyperparams = property(getHyperparams)

    def cov(self, x1, x2):
        raise NotImplementedError('kernel-derived class does not have cov method')
    
    
    def covMatrix(self, X):
        NX, _ = vstack(X).shape
        K = ones((NX, NX))
        for i in xrange(NX):
            for j in xrange(i+1):
                K[i, j] = K[j, i] = self.cov(X[i], X[j])
        
        return K
        
        
    def derivative(self, X, hp):
        raise NotImplementedError('kernel-derived class does not have derivative method')
        
class GaussianKernel_iso(Kernel):
    """
    Isotropic Gaussian (aka "squared exponential") kernel.  Has 2
    non-negative hyperparameters:
    
        hyperparams[0]      kernel width parameter
        hyperparams[1]      noise magnitude parameter
    """
    
    def __init__(self, hyperparams, **kwargs):
        super(GaussianKernel_iso, self).__init__(hyperparams)
        self._itheta2 = 1 / hyperparams[0]**2
        # self._magnitude = hyperparams[1]
        # self._sf2 = exp(2.0*log(self._magnitude))  # signal variance
        

    def cov(self, x1, x2):
        
        return exp(-.5 * norm(x1-x2)**2 * self._itheta2)
        

    def derivative(self, X, hp):
        
        NX, _ = vstack(X).shape
        K = self.covMatrix(X)
        
        if hp == 0:
            C = zeros(K.shape)
            for i in xrange(NX):
                for j in xrange(i):
                    C[i, j] = C[j, i] = sum((X[i]-X[j])**2) * self._itheta2
            return K * C
        # elif hp == 1:
        #     return 2.0 * K
        else:
            raise ValueError



        
