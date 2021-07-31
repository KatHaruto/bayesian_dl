import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sympy
from sympy import symbols,plot_implicit,Eq
from statistics import NormalDist

def generate_data(neg_size,pos_size):
    mean_neg = np.array([2, 2])
    cov_neg = np.array([[1, 0], [0, 1]])

    mean_pos = np.array([-2, -1])
    cov_pos = np.array([[1, 0], [0, 1]])
    # numpy を用いた生成
    x_neg = np.array([np.hstack(([1],d)).reshape(3,1) for d in np.random.multivariate_normal(mean_neg, cov_neg, size=neg_size)])
    x_pos = np.array([np.hstack(([1],d)).reshape(3,1) for d in np.random.multivariate_normal(mean_pos, cov_pos, size=pos_size)])

    x_pos[-1] = np.hstack(([1],np.random.multivariate_normal([5,5], [[1,0],[0,1]]))).reshape(3,1)
    X = np.concatenate([x_neg,x_pos])
    Y = np.concatenate([-1*np.ones(neg_size),np.ones(pos_size)])
    return X,Y
    #学習順序による影響を確認
    p = list(zip(X,Y))
    np.random.shuffle(p)
    X,Y = zip(*p)
    X = np.array([np.array(x) for x in X])
    Y = np.array([np.array(y) for y in Y])
    return X,Y
class ADFforProbitRegression():
    def __init__(self):
        self.mu = np.array([0,0,0]).reshape(3,1)
        self.v = np.array([[100,0,0],[0,100,0],[0,0,100]])
        self.mus= []
        self.vs= []
    def fit(self,X,Y):
        for x,y in zip(X,Y):
            self.update_param(x,y)
            self.mus.append(self.mu)
            self.vs.append(self.v)
    def update_param(self,x,y):
        self.update_mu(x,y)
        self.update_v(x,y)

    def update_mu(self,x,y):
        a = self.calc_a(x,y)
        Z = norm.cdf(a)
        dlnZ_du = self.calc_dlnZ_du(a,Z,x,y)
        
        self.mu = self.mu + self.v @ dlnZ_du

    def update_v(self,x,y):
        a = self.calc_a(x,y)
        Z = norm.cdf(a)
        dlnZ_du = self.calc_dlnZ_du(a,Z,x,y)
        dlnZ_dv = self.calc_dlnZ_dv(a,Z,x,y)
        
        self.v = self.v- self.v @ (dlnZ_du @ dlnZ_du.T - 2*dlnZ_dv) @ self.v

    def calc_dlnZ_du(self,a,Z,x,y):
        return (1/Z) * norm.pdf(a) * (y*x / np.sqrt(1+(x.T @ self.v @ x)[0][0]))
    def calc_dlnZ_dv(self,a,Z,x,y):
        return (1/Z) * norm.pdf(a) * a* (-(x @ x.T))/(2*(1+(x.T @ self.v @ x)[0][0]))
    def calc_a(self,x,y):
        a =y*(self.mu.T @ x)[0][0] / np.sqrt(1+(x.T @ self.v @ x)[0][0])
        return a

    def predict(self,X):
        X = np.hstack(([1],X)).reshape(3,1)
        return norm.cdf([((self.mu - np.sqrt(np.diag(self.v).reshape(3,1))).T @ X)[0][0],((self.mu + np.sqrt(np.diag(self.v).reshape(3,1))).T @ X)[0][0]])


def animate(i):

    ax.scatter(draw_x1[:,1],draw_x1[:,2],color='r')
    ax.scatter(draw_x2[:,1],draw_x2[:,2],color='b')
    
    mu,v = list(zip(pb.mus,pb.vs))[i]
    ax.set_xlim(int(min(X[:,1]))-1,int(max(X[:,1]))+1)
    ax.set_ylim(int(min(X[:,2]))-1,int(max(X[:,2]))+1)
    x = np.array(range(int(min(X[:,1]))-1,int(max(X[:,1]))+1))
    ax.plot(x,((-mu[0]-mu[1]*x) / (mu[2])))


neg_size = 100
pos_size = 100
X,Y = generate_data(neg_size,pos_size)
draw_x1 = np.array([list(x) for x,y in zip(X,Y) if y == -1])
draw_x2 = np.array([list(x) for x,y in zip(X,Y) if y == 1])
plt.scatter(draw_x1[:,1],draw_x1[:,2],color='r')
plt.scatter(draw_x2[:,1],draw_x2[:,2],color='b')
plt.savefig("./generated_data.png")
pb = ADFforProbitRegression()

pb.fit(X,Y)
#print(pb.predict([2.5,2.4]))
fig = plt.figure()
ax = fig.add_subplot(111)
ani = FuncAnimation(fig,animate,frames=np.linspace(0, len(X)-1, 50, dtype=int),interval=100)
ani.save("./result_adf.gif",writer='pillow')

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.scatter(draw_x1[:,1],draw_x1[:,2],color='r')
ax2.scatter(draw_x2[:,1],draw_x2[:,2],color='b')

mu,v = list(zip(pb.mus,pb.vs))[-1]
mu = mu.flatten()
print(mu)
print(v)
ax2.set_xlim(int(min(X[:,1]))-1,int(max(X[:,1]))+1)
ax2.set_ylim(int(min(X[:,2]))-1,int(max(X[:,2]))+1)
color = ['black','yellow','green']
lams = [NormalDist().inv_cdf(0.5),NormalDist().inv_cdf(0.95),NormalDist().inv_cdf(0.99)]
for i,lam in enumerate(lams):
    x,y = symbols("x y")
    a = mu[1]**2-lam**2*v[1][1]
    b = mu[2]**2-lam**2*v[2][2]
    c = 2*mu[1]*mu[2] - 2* lam**2 * v[1][2]
    d = 2*mu[0]*mu[1] - 2* lam**2 * v[0][1]
    e = 2*mu[0]*mu[2] - 2* lam**2 * v[2][0]
    f = lam**2*(1+v[0][0]) - mu[0]**2
    eq = Eq(a * x ** 2 + b * y ** 2 + c*x*y + d*x + e*y ,f)
    pli = plot_implicit(eq ,(x, int(min(X[:,1]))-1,int(max(X[:,1]))+1), (y, int(min(X[:,2]))-1,int(max(X[:,2]))+1),show=False)
    data, _ = pli[0].get_points()
    data = np.array([(x_int.mid, y_int.mid) for x_int, y_int in data]) 
    ax2.scatter(data[:,0],data[:,1],s=0.1,c=color[i],alpha=0.1)
plt.savefig("./result_adf.png")


fig = plt.figure()
ax3 = fig.add_subplot(111)
ax3.scatter(draw_x1[:,1],draw_x1[:,2],color='r')
ax3.scatter(draw_x2[:,1],draw_x2[:,2],color='b')
ax3.set_xlim(int(min(X[:,1]))-1,int(max(X[:,1]))+1)
ax3.set_ylim(int(min(X[:,2]))-1,int(max(X[:,2]))+1)
for i in range(5):
    sample_w = np.random.multivariate_normal(mu,v)
    print(sample_w)
    x = np.array(range(int(min(X[:,1]))-1,int(max(X[:,1]))+2))
    ax3.plot(x,((-sample_w[0]-sample_w[1]*x) / (sample_w[2])))
plt.savefig("./random_w.png")