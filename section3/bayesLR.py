import numpy as np
import matplotlib.pyplot as plt

class BayesianLR():
    def __init__(self,sigma2_y,sigma2_w,K):
        self.sigma2_y = sigma2_y
        self.sigma2_w = sigma2_w
        self.K = K
    
    def fit(self,X,Y,seqential):
        if seqential:
            pre_mu = self.mu_hat.flatten()
            pre_sigma = self.sigma_hat
        else:
            pre_sigma = np.eye(self.K+1) / self.sigma2_w
            pre_mu = np.array(np.zeros(self.K+1)).T
        self.sigma_hat = np.linalg.inv(np.sum( [ self.Phi(x) @ self.Phi(x).T  for x in X] ,axis=0) / self.sigma2_y + np.linalg.inv(pre_sigma))
        #import pdb; pdb.set_trace()
        self.mu_hat = self.sigma_hat @ ( (np.sum([float(y)*self.Phi(x) for x,y in zip(X,Y)],axis=0)) / self.sigma2_y + (np.linalg.inv(pre_sigma) @ pre_mu).reshape(-1,1))

    def predict(self,x_new):
        mu_pred = np.array([self.mu_hat.T @ self.Phi(x) for x in x_new]).flatten()

        phi_ = np.array([self.Phi(x) for x in x_new])
        phi_ = np.array(np.squeeze(phi_)).T
        sigma2_pred = self.sigma2_y + np.diag(phi_.T@ self.sigma_hat @ phi_)

        return mu_pred,sigma2_pred

    def get_sample_from_posterior(self,x_lin):
        w_ = np.random.multivariate_normal(mean=self.mu_hat.flatten(), cov=self.sigma_hat)
        
        phi_ = np.array([self.Phi(x) for x in x_lin])
        phi_ = np.array(np.squeeze(phi_)).T
        sample_mean_ = w_ @ phi_
        
        return np.array(sample_mean_).flatten()
    
    def Phi(self,x):
        return self.calc_basic_function(x)

    def calc_basic_function(self,x):
        return np.array([x**k for k in range(self.K+1)])


N_train = 6
x_train = 2*np.pi*np.random.rand(N_train).reshape(-1,1)
y_train =  np.sin(x_train) + np.random.normal(0,0.1,N_train).reshape(-1,1)
blr = BayesianLR(0.5,1,3)
blr.fit(x_train,y_train,False)
x_lin = np.linspace(-1, 7, 30).reshape(-1,1)
mu_pred, sigma2_pred = blr.predict(x_lin)
upper = mu_pred + np.sqrt(sigma2_pred)
lower = mu_pred - np.sqrt(sigma2_pred)

N_added = 2
x_added = np.array([[-0.8],[7]])
y_added = np.sin(x_added) + np.random.normal(0,2,N_added).reshape(-1,1)
blr.fit(x_added,y_added,True)
mu_pred, sigma2_pred = blr.predict(x_lin)
upper = mu_pred + np.sqrt(sigma2_pred)
lower = mu_pred - np.sqrt(sigma2_pred)
# Get five samples from posterior
for i in range(5):
    y_sample = blr.get_sample_from_posterior(x_lin)
    if i==0:
        plt.plot(x_lin, y_sample, color='green', linewidth = 1.2, label='sample from posterior')
    else:
        plt.plot(x_lin, y_sample, color='green', linewidth = 1.2)

plt.plot(x_lin, mu_pred, color='orangered', linewidth = 2.5, linestyle='dashed', label='predictive mean')
plt.fill_between(x_lin[:,0], upper, lower, color='lightgrey', label='predictive variance')
plt.scatter(x_train, y_train, color='black', marker='o')
plt.scatter(x_added, y_added, color='black', marker='o')

plt.legend(ncol=2)
plt.ylim(-3,3)
plt.show()
