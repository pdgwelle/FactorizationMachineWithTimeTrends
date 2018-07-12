import numpy as np
import scipy.optimize

class FlashMobJunior:

    def __init__(self, time_index, k=4):
        self.k = k
        self.time_index = time_index
        self.theta0 = None
        self.solution = None
        pass

    def _compute_low_rank_interactions_slow(self, X, Vs):
        W = np.matmul(Vs,np.transpose(Vs))

        running_sum = 0
        for i in range(len(W)):
            for j in range(i+1,len(W)):
                running_sum+=W[i,j]*X[:,i]*X[:,j]

        return running_sum

    def _model_equation(self, X, theta):
        n_features = X.shape[1]

        # unpack theta into model paramaeters
        B0 = theta[0]
        Bs = theta[1: self.time_index+1]
        Vs_unformed = theta[self.time_index+1:(self.time_index+1+self.time_index*self.k)]
        Vs = Vs_unformed.reshape(self.time_index, self.k)
        Vst_unformed = theta[(self.time_index+1+self.time_index*self.k):]
        Vst = Vst_unformed.reshape(n_features, self.k)

        # calculate y_hat and p
        y_hat = B0 + np.sum(np.multiply(Bs ,X), axis=1) + self._compute_low_rank_interactions_slow(X,Vs)
        probs = 1 / (1 + np.exp(-y_hat))

        probs[probs==0.0]+=0.001
        probs[probs==1.0]-=0.001

        return probs

    def _obj_function(self, theta, *args):
        # unpack data
        X = args[0]
        y = args[1]
        
        # predict p given theta
        p = self._model_equation(X,theta)

        # cross entropy loss
        loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        
        return loss

    def fit(self, X, y):
        n_features = X.shape[1]

        # generate random initalizations
        Bs = np.random.rand(self.time_index+1) - 0.5
        Vs = np.random.rand(self.time_index, self.k) - 0.5
        Vst = np.random.rand(n_features, self.k) - 0.5
        theta0 = np.concatenate((Bs, np.ravel(Vs), np.ravel(Vst)))
        self.theta0 = theta0

        # minimize!
        self.solution = scipy.optimize.minimize(self._obj_function, theta0, args=(X,y))

    def predict_proba(self, X):
        theta = self.solution.x
        p = self._model_equation(X, theta)
        return p

    def predict(self, X, threshold):
        p = self.predict_proba(X)
        y = np.array([0]*len(p))
        y[p>0.5] = 1
        return y