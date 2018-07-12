import numpy as np

import fmj

### Generate some data
X = np.random.rand(1000,10) - 0.5
z_func = lambda x: -0.1 + 0.4*x[1] - 0.6*x[3] + 0.2*x[5] - 0.4*x[7] + \
    0.1*x[9] - 0.2*x[0]*x[2] + 0.4*x[4]*x[6] - 0.3*x[8]**2
z = np.apply_along_axis(z_func, axis=1, arr=X)
p = 1 / (1 + np.exp(-z))
y = np.array([0]*len(p))
y[p>0.5] = 1

### Run classifer
junior = fmj.FlashMobJunior()
junior.fit(X,y)
f = junior.predict(X, threshold=0.5)