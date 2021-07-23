import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sympy
from sympy import symbols,plot_implicit,Eq


def generate_data(neg_size,pos_size):
    mean_1 = np.array([1, 1])
    cov_1 = np.array([[1, 0], [0, 1]])

    mean_2 = np.array([0, -1])
    cov_2 = np.array([[1, 0], [0, 1]])
    # numpy を用いた生成
    data_1 = np.array([np.hstack(([1],d)) for d in np.random.multivariate_normal(mean_1, cov_1, size=neg_size)])
    data_2 = np.array([np.hstack(([1],d)) for d in np.random.multivariate_normal(mean_2, cov_2, size=pos_size)])

    X = np.concatenate([data_1,data_2])
    Y = np.concatenate([-1*np.ones(neg_size),np.ones(pos_size)])
    return X,Y
    #学習順序による影響を確認
    p = list(zip(X,Y))
    np.random.shuffle(p)
    X,Y = zip(*p)
    X = np.array([np.array(x) for x in X])
    Y = np.array([np.array(y) for y in Y])
    return X,Y
class EP():
    class g():
        def __init__(self,prior=False):
            if prior:
                self.mu = np.array([0,0,0])
                self.v = np.array([[1,0,0],[0,1,0],[0,0,1]])
                self.s = 1
            else:
                self.mu = np.array([0,0,0])
                self.v = np.array([[1000,0,0],[0,1000,0],[0,0,1000]])
                self.s = 1

        def update_param(self,q_mu,q_v,q_mu_i,q_v_i,Z):
            self.v = np.diag(1/ (1 / np.diag(q_v) - 1 / np.diag(q_v_i)))
            self.mu = np.diag(self.v) * (( 1/np.diag(q_v) )*q_mu - (1/np.diag(q_v_i)) * q_mu_i)
            self.s = Z *np.prod(np.sqrt(np.diag(self.v)**2 * np.diag(q_v_i) / ((2*np.pi)**3*np.diag(q_v))) \
                    * np.exp(-(1/2)*(q_mu**2 * 1/np.diag(q_v_i)-q_mu_i**2 *(1/np.diag(q_v_i)) - self.mu**2 * 1/np.diag(self.v))))
            
    def __init__(self,data_num):
        self.likelihood = []
        self.mu = np.array([0,0,0])
        self.v = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.mus= []
        self.vs= []

        self.gs = [self.g(prior=True)] + [self.g()]*(data_num)
    
    def calc_Z(self,x,y):
        a = y*np.dot(x,self.mu) / np.sqrt(1+np.sum(np.dot(self.v,x**2))+1e-6)
        return a,norm.cdf(a)
    def calc_dln_Z_du(self,a,Z,x,y):
        # retun vector [dlnZ / du1, dlnZ / du2, ... ]
        return (1/Z) * norm.pdf(a) * (y*x / np.sqrt(1+np.sum(np.dot(self.v,x**2))+1e-6))
    def calc_dln_Z_dv(self,a,Z,x,y):
        # retun vector [dlnZ / dv1, dlnZ / dv2, ... ] (v1,v2... = diag(V))
        return (1/Z) * norm.pdf(a) * a* (-(x**2)/(2*(1+np.sum(np.dot(self.v,x**2)))))
    def update_mu(self,x,y):
        a,Z = self.calc_Z(x,y)
        d_lnZ_du = self.calc_dln_Z_du(a,Z,x,y)
        self.mu = self.mu + np.dot(self.v,d_lnZ_du)
    
    def remove_factor_of(self,g_i):
        self.prev_v = copy.deepcopy(self.v)
        self.prev_mu = copy.deepcopy(self.mu)

        self.v = np.diag(1/ (1 / np.diag(self.v) - 1 / np.diag(g_i.v) + 1e-5 ))
        self.mu = np.diag(self.v) * (( 1/np.diag(self.prev_v) )*self.mu - (1/np.diag(g_i.v)) * g_i.mu)
    def update_v(self,x,y):
        a,Z = self.calc_Z(x,y)
        d_lnZ_du = self.calc_dln_Z_du(a,Z,x,y)
        d_lnZ_dv = self.calc_dln_Z_dv(a,Z,x,y)
        ''' 
                [ v1 ] - [v1^2] * ( d_lnZ-du1 ^2 - 2*d_lnZ_dv1)
        v_new = [ v2 ] - [v2^2] * ( d_lnZ-du2 ^2 - 2*d_lnZ_dv2)
                  ︙
        ⇔ diag(V) - diag(V) ○ (d_lnZ-du ^2 - 2 * d_lnZ-dv)
                        (アダマール積)
        '''
        self.v = np.diag(np.diag(self.v)-np.diag((self.v))**2 * (d_lnZ_du**2 - 2*d_lnZ_dv))
    def update_q(self,x,y):
        self.update_mu(x,y)
        self.update_v(x,y)
    
    def update_g_i(self,g_i,x,y):
        Z = norm.cdf(y*np.dot(x,self.prev_mu) / np.sqrt(1+np.sum(np.dot(self.prev_v,x**2))+1e-6))
        g_i.update_param(self.mu,self.v,self.prev_mu,self.prev_v,Z)

    def fit(self,X,Y):
        while True:
            if self.is_converge():
                print(self.mu,self.v)
                return
            for i, (x,y) in enumerate(zip(X,Y), start=1):
                self.remove_factor_of(self.gs[i])
                self.update_q(x,y)
                self.update_g_i(self.gs[i],x,y)
                self.mus.append(self.mu)
                self.vs.append(self.v)
            self.likelihood.append(self.calc_likelihood(X,Y))
            if len(self.likelihood) >= 2:
                print(abs(self.likelihood[-1] - self.likelihood[-2]))
            print(self.gs[1].s)
        
    def is_converge(self):
        if len(self.likelihood) >= 2 and abs(self.likelihood[-1] - self.likelihood[-2]) < 1e-2:
            return True
        return False
    def calc_likelihood(self,X,Y):
        return np.sum([np.log(norm.cdf(Y[i]*np.dot(self.mu,X[i]))+1e-6) for i in range(len(X))])

def animate(i):

    ax.scatter(draw_x1[:,1],draw_x1[:,2],color='r')
    ax.scatter(draw_x2[:,1],draw_x2[:,2],color='b')
    
    mu,v = list(zip(ep.mus,ep.vs))[i]
    ax.set_xlim(int(min(X[:,1]))-1,int(max(X[:,1]))+1)
    ax.set_ylim(int(min(X[:,2]))-1,int(max(X[:,2]))+1)
    x = np.array(range(int(min(X[:,1]))-1,int(max(X[:,1]))+1))
    ax.plot(x,(-(mu[0]+mu[1]*x) / (mu[2])))
neg_size = 100
pos_size = 100
X,Y = generate_data(neg_size,pos_size)
draw_x1 = np.array([list(x) for x,y in zip(X,Y) if y == -1])
draw_x2 = np.array([list(x) for x,y in zip(X,Y) if y == 1])
ep = EP(200)

ep.fit(X,Y)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ani = FuncAnimation(fig,animate,frames=np.linspace(0, len(ep.mus)-1,100,dtype=int),interval=100)
#ani.save("./result_EP.gif",writer='pillow')