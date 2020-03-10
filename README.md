# PrefInt
The PrefInt package provides a way of integrating data via preference learning. For more details, please check our online manuscript: https://arxiv.org/abs/1910.11516

Python Version: 2.7
## Example
For generating pairwise preference via traindata and add preference with new point:      
This is for adding single new datapoint during iteration in optimization. 
For adding more datapoints in one step, please extend the traindata to regenerate the pairs.

```python
from PrefInt.preference_generate import generate_pair
GN = generate_pair(traindata)
prefs = GN.firstgen()

newprefs = GN.addnew(newdata)
```
For training the pairwise preference:

```python
from PrefInt.ibo.gaussianprocess import PrefGaussianProcess
from PrefInt.ibo.gaussianprocess.kernel import GaussianKernel_iso

kernel = GaussianKernel_iso(np.array([18.0]))
GP = PrefGaussianProcess(kernel)
GP.addPreferences(prefs)
```
For more examples, please check [examples](https://github.com/tsudalab/PrefInt/tree/master/Examples) and [utility](https://github.com/tsudalab/PrefInt/tree/master/PrefInt/utility).
## License
The PrefInt package is licensed under the MIT "Expat" License
