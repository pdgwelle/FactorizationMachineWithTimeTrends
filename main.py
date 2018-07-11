import numpy as np

import fmj

### Generate some data
X = np.random.rand(1000,10)
y = np.random.binomial(n=1, p=0.5, size=1000)

### Run classifer
junior = fmj.FlashMobJunior()
junior.fit(X,y)
junior.predict(X)