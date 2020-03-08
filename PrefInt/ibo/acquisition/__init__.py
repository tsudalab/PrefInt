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

# check https://github.com/misterwindupbird/IBO for more details

from numpy import array, sqrt, max, arange, log, nan, linalg, isscalar, pi
from code.ibo.gaussianprocess import CDF, PDF


class EI(object):
    
    def __init__(self, GP, xi=.01, **kwargs):
        super(EI, self).__init__()
        self.GP = GP
        self.ymax = max(self.GP.Y)
        self.xi = xi
        #self.kappa = kappa

        assert isscalar(self.ymax)
        assert isscalar(self.xi)
        

    def negf(self, x):
        
        mu, sig2 = self.GP.posterior(x)
        assert isscalar(mu)
        assert isscalar(sig2)
        
        ydiff = mu - self.ymax - self.xi
        s = sqrt(sig2)
        Z = float(ydiff / s)

        EI = (ydiff * CDF(Z)) + (s * PDF(Z))
        if EI is nan:
            return 0.
        # print '[python] EI =', EI
        return -EI
    

    def f(self, x):
        
        return -self.negf(x)

        