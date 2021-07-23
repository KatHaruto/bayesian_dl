import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sympy
from sympy import symbols,plot_implicit,Eq


def generate_data(neg_size,pos_size):
    mean_1 = np.array([2, 1])
    cov_1 = np.array([[1, 0], [0, 1]])

    mean_2 = np.array([2, 1])
    cov_2 = np.array([[1, 0], [0, 1]])
    # numpy を用いた生成
    data_1 = np.array([np.hstack(([1],d)) for d in np.random.multivariate_normal(mean_1, cov_1, size=neg_size)])
    data_2 = np.array([np.hstack(([1],d)) for d in np.random.multivariate_normal(mean_2, cov_2, size=pos_size)])

    X = np.concatenate([data_1,data_2])
    Y = np.concatenate([-1*np.ones(neg_size),np.ones(pos_size)])
    #return X,Y
    #学習順序による影響を確認
    p = list(zip(X,Y))
    np.random.shuffle(p)
    X,Y = zip(*p)
    X = np.array([np.array(x) for x in X])
    Y = np.array([np.array(y) for y in Y])
    return X,Y
class MomentMatchingProbit():
    def __init__(self):
        self.likelihood = []
        self.mu = np.array([0,0,0])
        self.v = np.array([[10000,0,0],[0,10000,0],[0,0,10000]])
        self.mus= []
        self.vs= []
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
    def update_param(self,x,y):
        self.update_mu(x,y)
        self.update_v(x,y)
    def fit(self,X,Y):
        for x,y in zip(X,Y):
            self.update_param(x,y)
            self.mus.append(self.mu)
            self.vs.append(self.v)
            self.likelihood.append(self.calc_likelihood(X,Y))
        
    def calc_likelihood(self,X,Y):
        return np.sum([np.log(norm.cdf(Y[i]*np.dot(self.mu,X[i]))+1e-6) for i in range(len(X))])

    def predict(self,X):
        X = np.hstack(([1],X)) 
        print(X)
        return norm.cdf([np.dot(self.mu - np.sqrt(np.diag(self.v)),X),np.dot(self.mu + np.sqrt(np.diag(self.v)),X)])


def animate(i):

    ax.scatter(draw_x1[:,1],draw_x1[:,2],color='r')
    ax.scatter(draw_x2[:,1],draw_x2[:,2],color='b')
    
    mu,v = list(zip(pb.mus,pb.vs))[i]
    ax.set_xlim(int(min(X[:,1]))-1,int(max(X[:,1]))+1)
    ax.set_ylim(int(min(X[:,2]))-1,int(max(X[:,2]))+1)
    x = np.array(range(int(min(X[:,1]))-1,int(max(X[:,1]))+1))
    ax.plot(x,(-(mu[0]+mu[1]*x) / (mu[2])))


neg_size = 100
pos_size = 100
X,Y = generate_data(neg_size,pos_size)
draw_x1 = np.array([list(x) for x,y in zip(X,Y) if y == -1])
draw_x2 = np.array([list(x) for x,y in zip(X,Y) if y == 1])
pb = MomentMatchingProbit()

pb.fit(X,Y)
print(pb.predict([1,0]))
fig = plt.figure()
ax = fig.add_subplot(111)
ani = FuncAnimation(fig,animate,frames=np.linspace(0, len(X)-1, 10, dtype=int),interval=1000)
ani.save("./result_moment_matching.gif",writer='pillow')

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.scatter(draw_x1[:,1],draw_x1[:,2],color='r')
ax2.scatter(draw_x2[:,1],draw_x2[:,2],color='b')

mu,v = list(zip(pb.mus,pb.vs))[-1]
print(mu)
print(v)
ax2.set_xlim(int(min(X[:,1]))-1,int(max(X[:,1]))+1)
ax2.set_ylim(int(min(X[:,2]))-1,int(max(X[:,2]))+1)
color = ['black','yellow','green']

for i,lam in enumerate([0,1.96,2.32]):
    x,y = symbols("x y")
    a = mu[1]**2-lam**2*v[1][1]
    b = mu[2]**2-lam**2*v[2][2]
    c = 2*mu[1]*mu[2]
    d = 2*mu[0]*mu[1]
    e = 2*mu[0]*mu[2]
    f = lam**2*(1+v[0][0]) - mu[0]**2
    eq = Eq(a * x ** 2 + b * y ** 2 + c*x*y + d*x + e*y ,f)
    pli = plot_implicit(eq ,(x, int(min(X[:,1]))-1,int(max(X[:,1]))+1), (y, int(min(X[:,2]))-1,int(max(X[:,2]))+1),show=False)
    data, _ = pli[0].get_points()
    data = np.array([(x_int.mid, y_int.mid) for x_int, y_int in data])
    ax2.scatter(data[:,0],data[:,1],s=0.1,c=color[i],alpha=0.1)
plt.savefig("./result_moment_matching.png")
