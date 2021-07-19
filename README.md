# bayesian_dl
implementation of the book 'Bayesian Deep Learning' in python from scratch using NumPy

## Bayesian Linear Regression 
This model can be updated sequentially.

![result_blr](https://user-images.githubusercontent.com/74958594/124242836-22a86b80-db58-11eb-9b76-a09aa762bfc5.png)

## MCMC 
- Metropolis-Hastings method
- Hamiltonian Monte Carlo method
- Gibbs sampling

Blue dots are adopted.
Red dots are rejected.

![result_mcmc](https://user-images.githubusercontent.com/74958594/124243411-d3af0600-db58-11eb-8369-717d673acf6c.png)

## Moment Matching and Assumed Density Filtering
Application of Assumed Density Filtering to Binary Classification by Probit Model.

In moment matching, the posterior distribution of parameters is updated sequentially. 
This GIF image shows the classification results for each parameter update.
 - Red points are **negative** examples that were **correctl**y  classified
 - Blue points are **negative** examples that **failed** to be classified 
 - Green points are **positive** examples that were **correctly** classified
 - Yellow points are **positive** examples that **failed** to be classified

Classification fails at first, but succeeds as learning progresses.
![result_moment_matching](https://user-images.githubusercontent.com/74958594/126166765-5ba49f3f-b122-4376-b58c-8a39c4d3e296.gif)

