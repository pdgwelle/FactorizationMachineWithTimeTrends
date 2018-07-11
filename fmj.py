import numpy as np
import scipy.optimize

class FlashMobJunior:

    def __init__(self, k=4):
        self.k = k
        pass

    def _compute_low_rank_interactions_slow(self, X, Vs):
        W = np.matmul(Vs,np.transpose(Vs))

        running_sum = 0
        for i in range(len(W)):
            for j in range(len(W)):
                running_sum+=W[i,j]*X[:,i]*X[:,j]

        return running_sum

    def _model_equation(self, X, X2, theta):
        n_features = X.shape[1]

        # unpack theta into model paramaeters
        B0 = theta[0]
        Bs = theta[1: n_features+1]
        Vs_unformed = theta[n_features+1:]
        Vs = Vs_unformed.reshape(n_features, self.k)

        # calculate y_hat and p
        y_hat = B0 + np.sum(np.multiply(Bs ,X), axis=1) + self._compute_low_rank_interactions_slow(X,Vs)
        p = 1 / (1 + np.exp(-y_hat))

        return p

    def _obj_function(self, theta, *args):
        # unpack data
        X = args[0]
        X2 = args[1]
        y = args[2]
        
        # predict p given theta
        p = self._model_equation(X,X2,theta)

        # cross entropy loss
        loss = np.mean(-y*np.log(p) + (1-y)*np.log(1-p))
        
        return loss

    def fit(self, X, y):
        # precompute X squared for speed
        X2 = X**2

        n_features = X.shape[1]

        # generate random initalizations
        Bs = np.random.rand(n_features+1)
        Vs = np.random.rand(n_features, self.k)
        theta0 = np.append(Bs, Vs)

        # minimize!
        self.solution = scipy.optimize.minimize(self._obj_function, theta0, args=(X,X2,y))

    def predict(self, X, threshold = 0.5):
        X2 = X**2
        theta = self.solution.x
        
        p = self._model_equation(X, X2, theta)
        
        return p
        # y = np.array([0]*len(p))
        # y[p>0.5] = 1
        # return y
