# bayesian_dl
implementation of the book 'Bayesian Deep Learning' in python from scratch using NumPy

Numpyから「ベイズ深層学習」の実装

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

## Moment Matching, Assumed Density Filtering and Expectation Propagation
Application of Assumed Density Filtering and Expectation Propagation to Binary Classification by Probit Model.

refer https://qiita.com/tr-author/items/9ff85ff87f388e230ec4 for detail, in Japanese.

In moment matching, the posterior distribution of parameters is updated sequentially. 

The image shows the boundaries of the classification at each update of the parameters.
(On this boundary, the value of the  cumulative distribution function of the standard normal distribution will be 0.5)

![the boundaries of the classification](https://user-images.githubusercontent.com/74958594/126290607-f6af1190-cfcf-4a75-8a66-72eb2b143989.gif)

This image similarly shows the boundaries where the classification probabilities are 50% 95% and 99%
![the boundaries of the classification](https://user-images.githubusercontent.com/74958594/126291202-36a5f0f7-22a7-405c-a6f1-edf85be4e3d0.png)




