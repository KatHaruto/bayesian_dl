import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal

def generate_data():
    size = 100
    mean_1 = np.array([4, 4])
    cov_1 = np.array([[1, 0], [0, 1]])

    mean_2 = np.array([-1, -1])
    cov_2 = np.array([[1, 0], [0, 1]])
    # numpy を用いた生成
    data_1 = np.random.multivariate_normal(mean_1, cov_1, size=size)
    data_2 = np.random.multivariate_normal(mean_2, cov_2, size=size)

    X = np.concatenate([data_1,data_2])
    Y = np.concatenate([-1*np.ones(100),np.ones(100)])
    return X,Y
class MomentMatchingProbit():
    def __init__(self):
        self.liklihood = []
        self.mus = []
        self.vs = []
        self.mu = np.array([0,0])
        self.v = np.array([[100,0],[0,100]])
    def calc_Z(self,x,y):
        a = y*np.dot(x,self.mu) / np.sqrt(1+np.dot(self.v,x**2)+1e-6)
        return a,norm.cdf(a)
    def calc_dln_Z_du(self,a,Z,x,y):
        return (1/Z) * norm.pdf(a) * (y*x / np.sqrt(1+np.dot(self.v,x**2)+1e-6))
    def calc_dln_Z_dv(self,a,Z,x,y):
        return (1/Z) * norm.pdf(a) * a* (-(x**2)/(2*(1+np.dot(self.v,x**2))))
    def update_mu(self,x,y):
        a,Z = self.calc_Z(x,y)
        d_lnZ_du = self.calc_dln_Z_du(a,Z,x,y)
        self.mu = self.mu + np.dot(self.v,d_lnZ_du)
    def update_v(self,x,y):
        a,Z = self.calc_Z(x,y)
        d_lnZ_du = self.calc_dln_Z_du(a,Z,x,y)
        d_lnZ_dv = self.calc_dln_Z_dv(a,Z,x,y)
        
        self.v = self.v-(self.v)**2 * (d_lnZ_du**2 - 2*d_lnZ_dv)
    def update_param(self,x,y):
        self.update_mu(x,y)
        self.update_v(x,y)
    def fit(self,X,Y):
        for x,y in zip(X,Y):
            self.update_param(x,y)
            self.mus.append(self.mu)
            self.vs.append(self.v)
            if self.calc_liklihood(X,Y) > -1e3:
                self.liklihood.append(self.calc_liklihood(X,Y))
    def calc_liklihood(self,X,Y):
        return np.sum([np.log(norm.cdf(Y[i]*np.dot(self.mu,X[i]))+1e-6) for i in range(len(X))])

    def plot(self):
        plt.subplot(1,2,1)
        plt.plot(list(range(len(self.liklihood))),self.liklihood)
        plt.subplot(1,2,2)
        #plt.plot(range(len(self.mus)),self.mus)
        #plt.fill_between(range(len(self.vs)), self.mus+np.sqrt(self.vs), self.mus-np.sqrt(self.vs), color='lightgrey', label='predictive variance')
        plt.show()

X,Y = generate_data()

pb = MomentMatchingProbit()
pb.fit(X,Y)
pb.plot()
print(pb.mu,pb.v)
print(sum(norm.cdf(np.dot(X[:100],pb.mu)) < 0.5))
print(sum(norm.cdf(np.dot(X[100:],pb.mu)) > 0.5))
plt.show()