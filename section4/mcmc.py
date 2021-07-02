import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

class MCMC():
    def __init__(self,mu,sigma,K,sample_size):
        self.mu=mu
        self.sigma=sigma
        self.K = K
        self.sample_size = sample_size
    
    def MH(self):
        def p_hat(x,mu,sigma):
            return np.exp(-((x - mu).T @ np.linalg.inv(sigma) @ (x - mu))[0][0]/2.0)

        def q(x,x_,):
            #return q(x | x_)
            return np.exp(-((x - x_).T @ (x - x_))[0][0]/2.0) / (np.sqrt((2 * np.pi) ** self.K))
        
        t = 0
        delta = 1
        #x = np.zeros(self.K)
        x = np.array([2,3]).reshape(-1,1)
        x_acc = np.array(x).reshape(1,self.K)
        x_rej = np.array(x).reshape(1,self.K)
        while t < self.sample_size:
            
            x_ = np.random.multivariate_normal(x.flatten(), np.eye(self.K)).reshape(-1,1)
            #print(((x - x_).T@(x - x_))[0][0])
            r = p_hat(x_,self.mu,self.sigma)*q(x,x_) / (p_hat(x,self.mu,self.sigma)*q(x_,x))
            if np.random.uniform() < min(1,r):
                x = x_
                x_acc= np.append(x_acc,np.array(x).reshape(1,self.K),axis=0)
                t += 1
            else:
                x_rej = np.append(x_rej,np.array(x_).reshape(1,self.K),axis=0)

        return x_acc[1:],x_rej[1:]

    def HMC(self,L=100,epsilon=0.5):
        def d_log_prob(self,x):
            return - np.linalg.inv(self.sigma) @ (x - self.mu)

        def d_potential(self,x):
            return - d_log_prob(self,x)

        def join_prob(x,p):
            u = (x - self.mu).T @ np.linalg.inv(self.sigma) @ (x - self.mu) / 2
            k = p.T @ p / 2
            return np.exp(-(u+k))


        def leap_frog(self,x,p):
            for _ in range(L):
                p = p - (epsilon / 2 ) * d_potential(self,x)
                x = x + epsilon*p
                p = p - (epsilon / 2 )* d_potential(self,x)

            return x,p

        t = 0
        x = np.array([2,3]).reshape(-1,1)
        x_acc = np.array(x).reshape(1,self.K)
        x_rej = np.array(x).reshape(1,self.K)
        while t < self.sample_size:
            #import pdb;pdb.set_trace()
            p = np.random.multivariate_normal(np.zeros(self.K), np.eye(self.K)).reshape(-1,1)
            x_,p_ = leap_frog(self,x,p)

            r = (join_prob(x_,p_) / join_prob(x,p))[0][0]
            if np.random.uniform() < min(1,r):
                x = x_
                x_acc= np.append(x_acc,np.array(x).reshape(1,self.K),axis=0)
                t += 1
            else:
                x_rej = np.append(x_rej,np.array(x_).reshape(1,self.K),axis=0)

        return x_acc[1:],x_rej[1:]
    
    def gibbs(self):
        def sample_x1_from_x2(self,x):
            mu = self.mu[0]+ self.sigma[0][1] / self.sigma[1][1] * (x[1]-self.mu[1])
            sigma = (self.sigma[0][0]*self.sigma[1][1] -self.sigma[0][1]*self.sigma[1][0]) / self.sigma[1][1]

            return np.random.normal(mu, np.sqrt(sigma))
        def sample_x2_from_x1(self,x):
            mu = self.mu[1]+ self.sigma[1][0] / self.sigma[0][0] * (x[0]-self.mu[0])
            sigma = (self.sigma[0][0]*self.sigma[1][1] -self.sigma[0][1]*self.sigma[1][0]) / self.sigma[0][0]

            return np.random.normal(mu, np.sqrt(sigma))
        
        t = 0
        x = np.array([0,0]).reshape(-1,1)
        x_acc = np.array(x).reshape(1,self.K)

        for i in range(self.sample_size):
            if i %2 ==0:
                x1 = sample_x1_from_x2(self,x)
                x = np.array([x1,x[1]])
            else:
                x2 = sample_x2_from_x1(self,x)
                x = np.array([x[0],x2])
            
            x_acc= np.append(x_acc,np.array(x).reshape(1,self.K),axis=0)
        return x_acc,None





K = 2
mu = np.zeros(K)
sigma = np.array([[2,-1],[-1,3]])
xy = np.random.multivariate_normal(mu,sigma,300)
fig, axs = plt.subplots(1,3, figsize=(10, 5))
mu = mu.reshape(-1,1)
mcmc = MCMC(mu,sigma,K,300)
for i, ax in enumerate(axs):
    ax.set_aspect('equal', adjustable='box')
    confidence_ellipse(xy[:,0], xy[:,1], ax, edgecolor='red')

    if i == 0:
        acc,rej = mcmc.MH()
        ax.set_title("MH")
        ax.plot(rej[:,0],rej[:,1],c='r',linewidth=0.3,linestyle="dashed",marker="o",markersize=3)
    elif i ==1:
        acc,rej = mcmc.HMC()
        ax.set_title("HMC")
        ax.plot(rej[:,0],rej[:,1],c='r',linewidth=0.3,linestyle="dashed",marker="o",markersize=3)
    else:
        acc,rej = mcmc.gibbs()
        ax.set_title("Gibbs")
        
    ax.plot(acc[:,0],acc[:,1],linewidth=1,linestyle="dashed",marker="o",markersize=3)
    if not rej is None: 
        ax.plot(rej[:,0],rej[:,1],c='r',linewidth=0.3,linestyle="dashed",marker="o",markersize=3)
plt.savefig("./result_mcmc.png")
plt.show()
