import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
def confidence_ellipse(mu,cov, n_std=3.0, facecolor='none', **kwargs):

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
    mean_x = mu[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ellipse


def generate_data():
    pos_size = 100
    neg_size = 100
    mean_1 = np.array([1, 1])
    cov_1 = np.array([[1, 0], [0, 1]])

    mean_2 = np.array([3, -2])
    cov_2 = np.array([[1, 0], [0, 1]])
    # numpy を用いた生成
    data_1 = np.random.multivariate_normal(mean_1, cov_1, size=neg_size)
    data_2 = np.random.multivariate_normal(mean_2, cov_2, size=pos_size)

    X = np.concatenate([data_1,data_2])
    Y = np.concatenate([-1*np.ones(neg_size),np.ones(pos_size)])
    return X,Y
class MomentMatchingProbit():
    def __init__(self):
        self.likelihood = []
        self.mu = np.array([0,0])
        self.v = np.array([[10,0],[0,10]])
        self.mus= []
        self.vs= []
        self.images = []
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
        self.evaluation(X,Y)
        for x,y in zip(X,Y):
            self.update_param(x,y)
            self.mus.append(self.mu)
            self.vs.append(self.v)
            self.evaluation(X,Y)
            self.likelihood.append(self.calc_likelihood(X,Y))

    def evaluation(self,X,Y):
        color = np.concatenate([np.where(norm.cdf(np.dot(X[:100],self.mu)) < 0.5,'r','b'),
                                np.where(norm.cdf(np.dot(X[100:],self.mu)) >= 0.5,'g','y')])
        im1 = ax.scatter(X[:,0],X[:,1],color=color)
        images.append([im1])
        

    def calc_likelihood(self,X,Y):
        return np.sum([np.log(norm.cdf(Y[i]*np.dot(self.mu,X[i]))+1e-6) for i in range(len(X))])



X,Y = generate_data()
pb = MomentMatchingProbit()
fig = plt.figure()
ax = fig.add_subplot(111)
images = []
pb.fit(X,Y)
ani = animation.ArtistAnimation(fig, images)
ani.save("./section4/result_moment_matching.gif",writer='pillow')
fig = plt.figure()
ax = fig.add_subplot(111)



def update_anim(i):
    ax.cla()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    im = ax.scatter(pb.mus[i][0],pb.mus[i][1])
    ax.add_patch(confidence_ellipse(pb.mus[i], pb.vs[i], edgecolor='red'))

ani = FuncAnimation(fig, update_anim, frames=range(len(X)))

ani.save("./section4/result_moment_matchin_w.gif",writer="pillow")