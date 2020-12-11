---
layout: post
title: "Maximum Likelihood Estimation"
---

## Introduction
The goal of this topic is to introduce what does Maximum Likelihood Estimation (MLE) in statistics mean and the application of this method as well as the merits & demerits of this approach.

* Regarding the first part, the aim is to explain what is the difference between a probability density function and a likelihood function and some live illustrations for likelihood functions.

* About the second part, it is mainly focused on the most intuitive way to maximize the likelihood function, that is, equating the derivative of the likelihood function to 0 and solving it for the parameters of interest.

* The third part is about evaluating whether the variance of the estimator for the parameters of interest could achieve the Cramer-Rao lower bound. The motivation behind this task is that the lower variance of the estimator could have, the more likely that the estimated value is closer to the true value).

* Finally, it is essential to mention MLE for Normal Linear Model since this type of model has a wide variaty of application and it is constituted as the baseline-predictive model before applying other ML algorithms.

## Probability Density Function vs Likelihood Function

Suppose that a random variable $$X$$ generated from a probability density function (pdf) denoted as $$f(x;\boldsymbol\theta)$$, where $$\boldsymbol\theta$$ is the parameters of the distribution and everything after the semicolon ";" is treated as known or constant. Therefore, $$f(x;\boldsymbol\theta)$$ is a function of $$x$$ and everything else is known/fixed. <br>
For instance, if $$f(x;\boldsymbol\theta) = \frac{1}{\sqrt{2\pi\sigma^2}}exp\{-\frac{(x-\mu)^2}{2\sigma^2}\}$$, then $$X$$ is a random variable generated from the Normal distribution with the corresponding vector of parameters $$\boldsymbol\theta = \begin{pmatrix} \mu \\ \sigma^2 \end{pmatrix}$$. Note that, only $$x$$ is the variable for the function $$f(x;\boldsymbol\theta)$$ in this example.

Conversely, the likelihood function treats parameters of a distribution to be unknown variables while the outcome/realization of $$X$$ is known. Specifically, the likelihood function is denoted as $$L(\boldsymbol\theta; x)$$, which has exactly the same form of $$f(x;\boldsymbol\theta)$$. Semantically, the likelihood function tells how "likely" the given data $$x$$ is from the distribution with the pdf $$f(x;\boldsymbol\theta)$$ and this is often used in the context for the need of determining the hidden/latent distribution (e.g. try computing many pdfs of different distributions to see from which the data is more "likely"). 
That's enough for the definition, it's demo time!

Suppose a random vector $$\mathbf{X} = (X_1, X_2, ..., X_n)^T \sim Beta(\alpha, \beta)$$, where $$\alpha$$ and $$\beta$$ are the parameters for the Beta distribution.

Assume that $$\alpha = 1000$$ and $$\beta = 200$$ are the true parameters but unknown of Beta distribution for generating the above random vector and the only information given is that the some realizations from $$\mathbf{X}$$, which is $$\mathbf{x} = (x_1, x_2, ..., x_{100})^T$$ and the data is actually from a Beta distribution. Our job is thus to determine the underlying distribution for the given data, or finding the true values for $$\alpha$$ and $$\beta$$.

$$\Rightarrow$$ The likelihood function for $$\mathbf{x}$$ is of the form:<br>
<center>$$L(\alpha, \beta; \mathbf{x}) = \prod_{i=1}^{100}\frac{x^{\alpha-1}_i(1-x_i)^{\beta-1}}{B(\alpha, \beta)} \tag{1}$$</center>
where,
<center>$$B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}\space\space \text{is Beta function}$$</center>
However, the value of $$L(\alpha, \beta; \mathbf{x})$$ is extremely small since it is the product of values between 0 and 1 and as the number of observations increases, the value of $$L(\alpha, \beta; \mathbf{x})$$ is no longer accurate due to the precision of a float number in computer is 32 bits at maximum. Hence, it is advised to transform the product to the summation by taking the log of $$L(\alpha, \beta; \mathbf{x})$$ because $$log$$ is a monotonic function in which the increase/decrease in $$L(\alpha, \beta; \mathbf{x})$$ also leads to the increase/decrease in $$log(L(\alpha, \beta; \mathbf{x}))$$ and vice-versa. Commonly, $$log(L(\alpha, \beta; \mathbf{x}))$$ is called log likelihood function and denoted as $$l(\alpha, \beta; \mathbf{x})$$.

```python
# Import necessary packages
from scipy.stats import beta
import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
import matplotlib.animation as mpl_animation
from scipy import special

writer = mpl_animation.ImageMagickWriter(fps=1) # For animation use

np.random.seed(456)
```

```python
'''Generate 100 observations from Beta(1000, 200)'''
true_alpha = 1000
true_beta = 200
data = np.random.beta(true_alpha, true_beta, 100)

'''Try many different values of alpha & beta for the pdf 
to see which one is the most likely distribution of the given data'''
plt.figure(figsize=(10, 6))
for i in range(4):
    sample_alpha = int(np.ceil(np.random.rand() * 2000)) # Randomly pick a value for alpha starting from 0 to 2000
    sample_beta = int(np.ceil(np.random.rand() * 500)) # Randomly pick a value for beta starting from 0 to 500
    x = np.linspace(0, 1, 1000)
    plt.plot(x, beta.pdf(x, sample_alpha, sample_beta), label='alpha={} and beta={}'.format(sample_alpha, sample_beta))
    
    
plt.plot(data, np.zeros(data.shape) - 1, 'o', markersize=5, clip_on=False, label='data point')
plt.xlabel('x')
plt.ylabel('L(alpha, beta; x)')
plt.legend()
plt.title('4 different pdfs for the data')
plt.show()
```
![MLE example](/blog/assets/mle_example.png)

## Maximizing likelihood function

As can be shown in the figure above, the densities of the data points evaluated by the red pdf are generally higher compared to that of other pdf. So it makes more sense to say that those data points are more likely generated from the Beta distribution with $$\alpha = 1929.51$$ and $$\beta = 386.99$$. Therefore, the approach of MLE is to adjust parameters of a distribution such that it could maximize the likelihood function. Mathematically,
<center>$$\boldsymbol\theta = \underset{\boldsymbol\theta}{argmax}\space L(\boldsymbol\theta;\mathbf{x})$$ </center>
which is equivalent to
<center>$$\boldsymbol\theta = \underset{\boldsymbol\theta}{argmax}\space l(\boldsymbol\theta;\mathbf{x})$$</center>

Back to the above example, the target is to find out the true parameters for Beta distribution. And based on the given data, the best guess we can make is that the data points should be from the distribution such that their likelihood densities are as large as possible. Consequently, searching for the true parameters for Beta distribution is somehow equivalently to finding out the values that could maximize the likelihood function of Beta distribution.

Recall that the likelihood function of the data has the form of the Equation $$\text{(1)} \Rightarrow$$ the log likelihood function of the data should be <br>
<center>
    $$
    \begin{split}
        l(\alpha, \beta; \mathbf{x}) & = \sum_{i=1}^{100}[(\alpha-1)log(x_i) + (\beta-1)log(1-x_i) - log(B(\alpha, \beta))]\\
        & = \sum_{i=1}^{100}[(\alpha-1)log(x_i) + (\beta-1)log(1-x_i) - log(\Gamma(\alpha)) - log(\Gamma(\beta)) + log(\Gamma(\alpha + \beta))]
    \end{split}
    $$
</center>

The remaining task is to find the value of $$\alpha$$ and $$\beta$$ that maximize $$l(\alpha, \beta; \mathbf{x})$$ so the most straightforward way is by equating the gradient of $$l(\alpha, \beta; \mathbf{x})$$ to the vector $$\mathbf{0}$$ and solving for $$\alpha$$ and $$\beta$$. More precisely,<br>
<center>
    $$\nabla_{\boldsymbol\theta}l(\boldsymbol\theta; \mathbf{x}) = \mathbf{0}$$
</center>
which is equivalent to
<center>
    $$
    \begin{cases}
        \frac{\partial}{\alpha}=\sum_{i=1}^{100}log(x_i) - 100\frac{\Gamma'(\alpha)}{\Gamma(\alpha)} + 100\frac{\Gamma'(\alpha + \beta)}{\Gamma(\alpha + \beta)} = 0\\
        \frac{\partial}{\beta}=\sum_{i=1}^{100}log(1-x_i) - 100\frac{\Gamma'(\beta)}{\Gamma(\beta)} + 100\frac{\Gamma'(\alpha + \beta)}{\Gamma(\alpha + \beta)} = 0
    \end{cases}
    \Leftrightarrow
    \begin{cases}
        \frac{\partial}{\alpha}=\sum_{i=1}^{100}log(x_i) - 100\psi(\alpha) + 100\psi(\alpha + \beta) = 0 \hskip{2em}(2)\\
        \frac{\partial}{\beta}=\sum_{i=1}^{100}log(1 - x_i) - 100\psi(\beta) + 100\psi(\alpha + \beta) = 0 \hskip{2em}(3)
    \end{cases}
    $$
</center>
where,
<center>$$\psi(z) = \frac{\Gamma'(z)}{\Gamma(z)}$$ is the <a href="https://en.wikipedia.org/wiki/Digamma_function">digamma function</a></center>

Up to this stage, directly solving the Equations $$\text{(2)}$$ & $$\text{(3)}$$ for $$\alpha$$ & $$\beta$$ is infeasible due to the fact that these are nonlinear equations. Hence, one of the possible way to find the roots of these equations is using numerical methods. To be specific, 
Step <b>1</b>. Declare the smart-initial values of $$\alpha$$ & $$\beta$$, which could be derived from the Method of Moment (MoM), that is, <br>
<center>
    $$
    \begin{cases}
        E[X] = \frac{\alpha}{\alpha + \beta}\\
        E[X^2] = Var(X) + E[X]^2
    \end{cases}\\
    $$
    $$
    (\text{Here, 1st & 2nd methods of moment are required because we have 2 variables } \alpha \text{ & } \beta)
    $$
    $$
    \Leftrightarrow
    \begin{cases}
        \frac{\alpha}{\alpha + \beta} = E[X] \\
        \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)} + (\frac{\alpha}{\alpha + \beta})^2 = E[X^2]
    \end{cases}\\
    \Leftrightarrow
    \begin{cases}
        \frac{\alpha}{E[X]} = \alpha + \beta\\
        \frac{\alpha(\frac{\alpha}{E[X]} - \alpha)}{(\frac{\alpha^2}{E[X]^2}) (\frac{\alpha}{E[X]} + 1)} + E[X]^2 = E[X^2]
    \end{cases}\\
    \Leftrightarrow
    \begin{cases}
        \frac{\alpha}{E[X]} = \alpha + \beta\\
        \frac{E[X] - E[X]^2}{\frac{\alpha}{E[X]} + 1} = E[X^2] - E[X]^2
    \end{cases}\\
    \Leftrightarrow
    \begin{cases}
        \frac{\alpha}{E[X]} = \alpha + \beta\\
        E[X](1 - E[X])= (\frac{\alpha}{E[X]} + 1)(E[X^2] - E[X]^2)
    \end{cases}\\
    \Leftrightarrow
    \begin{cases}
        \frac{\alpha}{E[X]} = \alpha + \beta\\
        \frac{E[X](1 - E[X])}{E[X^2] - E[X]^2} - \frac{E[X^2] - E[X]^2}{E[X^2] - E[X]^2}= \frac{\alpha}{E[X]}
    \end{cases}\\
    \Leftrightarrow
    \begin{cases}
        \frac{\alpha}{E[X]} = \alpha + \beta\\
        \frac{E[X] - E[X^2]}{E[X^2] - E[X]^2}= \frac{\alpha}{E[X]}
    \end{cases}\\
    \Leftrightarrow
    \begin{cases}
        \frac{\alpha}{E[X]} = \alpha + \beta\\
        \frac{E[X]^2 - E[X^2]E[X]}{E[X^2] - E[X]^2}= \alpha
    \end{cases}\\
    \Leftrightarrow
    \begin{cases}
        \frac{\alpha}{E[X]} = \alpha + \beta\\
        \frac{E[X]^2 - E[X]^3 + E[X]^3 - E[X^2]E[X]}{E[X^2] - E[X]^2}= \alpha
    \end{cases}\\
    \Leftrightarrow
    \begin{cases}
        \frac{\alpha}{E[X]} = \alpha + \beta\\
        \frac{E[X](1 - E[X]^2)}{\underbrace{E[X^2] - E[X]^2}_{V[X]}} + E[X]= \alpha
    \end{cases}\\
    $$
    $$
    \text{(Note that V[X] is the sample variance NOT the population variance, the reason for this will be discussed shortly)}
    $$
    $$
    \Rightarrow
    \begin{cases}
        \alpha = \frac{E[X](1 - E[X]^2)}{V[X]} + E[X] \hskip{2em} (4)\\
        \beta = (\frac{E[X](1 - E[X])}{V[X]} - 1)(1 - E[X]) \hskip{2em} (5)
    \end{cases}\\
    $$
</center>
Based on the Law of Large Number (LLN): <br>
$$
\begin{cases}
    \frac{1}{N}\sum_{i=1}^{N}X_i \to E[X] \\
    \frac{1}{N}\sum_{i=1}^{N}X_i^2 \to E[X^2]
\end{cases} \\
\hskip{2em} \text{(as N } \to \infty \text{)} \\
\Rightarrow V[X] \text{ in (4) & (5) }= \frac{1}{N}\sum_{i=1}^{N}X_i^2 - (\frac{1}{N}\sum_{i=1}^{N}X_i)^2
$$<br><br>
$$\rightarrow$$ Use $$(4)$$ & $$(5)$$ to estimate the initial value of $$\alpha$$ & $$\beta$$ we get:
<center>
    $$
    \begin{cases}
        \alpha_0 = \frac{\bar{x}^2(1-\bar{x})}{s^2} - \bar{x}\\
        \beta_0 = (\frac{\bar{x}(1-\bar{x})}{s^2} - 1)(1-\bar{x})
    \end{cases}
    $$
</center>
where,
<center>
    $$
        \bar{x} = \frac{\sum_{i=1}^{100}x_i}{100} \\
        s^2 = \frac{\sum_{i=1}^{100}(x_i - \bar{x})^2}{100}
    $$
</center>

Step <b>2</b>. Choose a numerical method to estimate $$\alpha$$ & $$\beta$$.<br><br>
Since taking the 2nd derivative of $$l(\alpha, \beta; \mathbf{x})$$ is feasible, it is therefore a reasonable choice to choose $$\text{Newton-Raphson}$$ method as a way to find the roots of the Equations $$(4)$$ & $$(5)$$ (More detail about why this method works could be found <a href="#">here</a>).
Step <b>3</b>. Evaluate the 2nd order gradient of $$l(\alpha, \beta; \mathbf{x})$$:
<center>
    $$
    \begin{split}
    \nabla^2_{\boldsymbol\theta}l(\boldsymbol\theta; \mathbf{x}) & = H(\boldsymbol\theta) = 
    \begin{pmatrix}
        \frac{\partial^2}{\alpha^2} & \frac{\partial^2}{\alpha\beta} \\
        \frac{\partial^2}{\beta\alpha} & \frac{\partial^2}{\beta^2}
    \end{pmatrix}\\
    & = 
    \begin{pmatrix}
        100\psi_1(\alpha + \beta) - 100\psi_1(\alpha) & 100\psi_1(\alpha + \beta) \\
        100\psi_1(\alpha + \beta) & 100\psi_1(\alpha + \beta) - 100\psi_1(\beta)
    \end{pmatrix}\\
    \end{split}
    $$
</center>
where,
<center>$$\psi_1(z) = \frac{d^2}{z^2}ln\Gamma(z)$$ is the <a href="https://en.wikipedia.org/wiki/Trigamma_function">trigamma function</a></center>

Step <b>4</b>. Define the way to update parameters:

<center>$$\boldsymbol\theta_{t+1} = \boldsymbol\theta_t - H(\boldsymbol\theta_t)^{-1}S(\boldsymbol\theta_t)$$</center>
where,
<center>$$S(\boldsymbol\theta) = \nabla_{\boldsymbol\theta}l(\boldsymbol\theta;\mathbf{x})$$</center>

```python
def plot_mle(ax1, ax2, camera, data, theta, iteration):
    '''
        Plot the Equations (2) & (3) given above to demonstrate how Newton-Raphson method helps to find the roots of these functions
        
        Parameters
        ----------
        ax1: The first axes for sketching the Equation (2)
        
        ax2: The second axes for sketching the Equation (3)
        
        camera: An object used to record ax1 and ax2 results
        
        data: Input data used to compute the Equations (2) & (3)
        
        theta: The current estimated alpha & beta, which could be referred via the index 0 & 1 of this variable respectively.
        
        iteration: The current iteration of parameter-updating procedure
        
        Return
        ------
        NoneType
        
        Notes
        -----
        Due to limited computational resource, both Equations (2) & (3) can only be plotted in 2D space in which
        The Equation (2) is treated as a one-variable function w.r.t alpha, while beta is assigned to the current estimated value of beta
        Similarly, Equation (3) is treated as a one-variable function w.r.t beta, while alpha is assigned to the current estimated value of alpha
    '''
    partial_l_wrt_alpha = lambda alpha, beta: np.sum(np.log(data)) - (len(data) * special.polygamma(0, alpha)) + (len(data) * special.polygamma(0, alpha + beta))
    partial_l_wrt_beta = lambda alpha, beta: np.sum(np.log(1 - data)) - (len(data) * special.polygamma(0, beta)) + (len(data) * special.polygamma(0, alpha + beta))
    alphas = np.linspace(750, 1500, 1000)
    betas = np.linspace(100, 300, 1000)
    z1 = [partial_l_wrt_alpha(a, theta[1]) for a in alphas]
    z1_at_current_alpha = partial_l_wrt_alpha(theta[0], theta[1])
    z2 = [partial_l_wrt_beta(theta[0], b) for b in betas]
    z2_at_current_beta = partial_l_wrt_beta(theta[0], theta[1])
    ax1.plot(alphas, z1, color='blue')
    ax1.plot(theta[0], z1_at_current_alpha, marker='o', clip_on=False, color='red')
    ax2.plot(betas, z2, color='blue')
    ax2.plot(theta[1], z2_at_current_beta, marker='o', clip_on=False, color='red')
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('Equation (2)')
    ax2.set_xlabel('beta')
    ax2.set_ylabel('Equation (3)')
    ax1.legend(['beta={}, iteration:{}'.format(np.round(theta[1], 2), iteration)])
    ax2.legend(['alpha={}, iteration:{}'.format(np.round(theta[0], 2), iteration)])
    ax1.set_title('Partial derivative of log likelihood w.r.t alpha given beta')
    ax2.set_title('Partial derivative of log likelihood w.r.t beta given alpha')
    camera.snap()
    return
```
```python
'''Perform MLE to maximize the log likelihood function of Beta distribution'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
camera = Camera(fig)
sample_mean = np.mean(data)
sample_variance = np.var(data)
# Initialize values for alpha & beta
# current_theta[0] is the value of current alpha && current_theta[1] is the value of current beta
current_theta = np.array([((sample_mean ** 2) * (1 - sample_mean) / sample_variance) - sample_mean, \
                          ((sample_mean * (1 - sample_mean) / sample_variance) - 1) * (1 - sample_mean)])
threshold = np.array([1e-5, 1e-5]) # Define the stop condition for updating parameters
# Perform Newton-Raphson method for updating alpha & beta
i = 0 # Memorize the number of iterations
while True:
    plot_mle(ax1, ax2, camera, data, current_theta, i)
    H11 = (len(data) * special.polygamma(1, current_theta[0] + current_theta[1])) - (len(data) * special.polygamma(1, current_theta[0]))
    H12 = len(data) * special.polygamma(1, current_theta[0] + current_theta[1])
    H22 = (len(data) * special.polygamma(1, current_theta[0] + current_theta[1])) - (len(data) * special.polygamma(1, current_theta[1]))
    S1 = (len(data) * special.polygamma(0, current_theta[0] + current_theta[1])) - (len(data) * special.polygamma(0, current_theta[0])) + np.sum(np.log(data))
    S2 = (len(data) * special.polygamma(0, current_theta[0] + current_theta[1])) - (len(data) * special.polygamma(0, current_theta[1])) + np.sum(np.log(1 - data))
    H = np.array([[H11, H12], [H12, H22]])
    S = np.array([S1, S2])
    next_theta = current_theta - np.dot(np.linalg.inv(H), S)
    if (next_theta - current_theta > threshold).all():
        current_theta = next_theta
    else: # If the changes in updated parameters are less than or equal to the threshold, stop the loop
        break
    i += 1
plt.tight_layout()
animation = camera.animate(interval=200)
animation.save('mle_update_parameters.gif', writer=writer)
print('Estimated values of alpha={}, beta={}'.format(int(current_theta[0]), int(current_theta[1])))
```
> Estimated values of alpha=1022, beta=206

![MLE demo](/blog/assets/mle_demo.gif)

```python
'''Plot the pdf of the estimated parameters vs the pdf of the actual parameters'''
x = np.linspace(0, 1, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, beta.pdf(x, current_theta[0], current_theta[1]), color='red', label='MLE pdf with alpha={} and beta={}'.format(int(current_theta[0]), int(current_theta[1])))
plt.plot(x, beta.pdf(x, true_alpha, true_beta), color='blue', label='true pdf with alpha={} and beta={}'.format(true_alpha, true_beta))    
plt.plot(data, np.zeros(data.shape) - 1, 'o', markersize=5, clip_on=False, label='data point')
plt.xlabel('x')
plt.ylabel('f(x; alpha, beta)')
plt.legend()
_ = plt.title('Actual pdf vs pdf with parameters derived from MLE')
```
![MLE pdf comparison](/blog/assets/mle_true_pdf_vs_estimated_pdf.png)

With only 2 iterations from Newton-Raphson algorithm, the near-optimal solution for the log likelihood function can be easily obtained. This is due to the fact that the initial values of $$\alpha$$ & $$\beta$$ could be evaluated analytically using MoM; as a result, the initial values are almost close to the MLE solution.<br><br>
So far, the parameters derived from MLE reflects the pdf (in red) which is almost analogous to the true pdf (in blue) as shown in the figure above. Hence, the more data could be obtained, the more accurate MLE method could achieve. <br><br>
It is noteworthy that, some likelihood functions are intractable to take derivative or to solve for the parameters of interest, but having said that, those problems could be tackled by using some numerical methods which shall be further discussed <a href="#">here</a>. 

## Cramér-Rao Information Inequality

The purpose of the C-R inequality is to evaluate whether the variance of the estimator could achieve the C-R lower bound (CRLB). This is particularly useful if we want to compare the performance of many estimators (e.g. the closer of the variance's estimator to the CRLB, the better the estimator is). More reasons could be found <a href="https://cnx.org/contents/g433V4cH@4/The-Cramer-Rao-Lower-Bound">here</a>. <br><br>
In this section, we shall not discuss the way to assess whether the MLE estimator of Beta distribution achieve CRLB since there is no closed-form solution for estimating $$\alpha$$ & $$\beta$$ as shown above; in other words, one of the numerical methods was used to compute MLE estimation of $$\alpha$$ & $$\beta$$. Even the estimated values of $$\alpha$$ & $$\beta$$ have the analytical solution obtained via MoM, evaluating the variance and the expectation of these estimations are still intractable.
Therefore, it is better to set another example to demonstrate how CRLB works.

Suppose a random vector $$\mathbf{X} = (X_1, X_2, ..., X_n)^T \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$$ <br>
Given the fact that the vector $$\mathbf{x} = (x_1, x_2, ..., x_{n})^T$$ is one realization of the vector $$\mathbf{X}$$ <br>
$$\Rightarrow$$ The MLE estimator of $$\mu$$, which could be obtained by solving the derivative of $$l(\mu, \sigma^2; \mathbf{x})$$ w.r.t $$\mu$$, is going to be:<br>
<center>$$\hat{\mu} = \bar{x} = \frac{\sum_{i=1}^{n}x_i}{n}$$</center>
Similarly, the MLE estimator of $$\sigma^2$$ is of the form:<br>
<center>$$\hat{\sigma}^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}$$</center>
However, the $$\hat{\sigma}^2$$ above is a biased estimator, that is, $$E[\hat{\sigma}^2] \neq \sigma^2$$ <br>
Proof:
<center>
    $$
    \begin{split}
        E[\hat{\sigma}^2] & = E[\frac{\sum_{i=1}^{n}(X_i - \bar{X})^2}{n}] \\
        & = E[\frac{\sum_{i=1}^{n}X_i^2 - 2\overbrace{X_i\bar{X}}^{(6)} + \overbrace{\bar{X}^2}^{(7)}}{n}]
    \end{split}
    $$
</center>
Consider $$(6)$$:
<center>$$X_i\bar{X} = \frac{X_1X_i + X_2X_i + ... + X_i^2 + ... + X_nX_i}{n} = \frac{X_i^2}{n} + \frac{(n-1)X_iX_j}{n} \hskip{2em} (\forall i \neq j)  \tag{*}$$</center>
Consider $$(7)$$:
<center>$$\bar{X}^2 = \underbrace{\frac{X_1^2 + X_2^2 + ... + X_n^2}{n^2}}_{\frac{nX_i^2}{n^2}} + \frac{(n^2 - n)X_iX_j}{n^2} \hskip{2em} (\forall i \neq j) \tag{**}$$</center>
<center>
    $$ 
    \begin{split}
        (*), (**) \Rightarrow E[\hat{\sigma}^2] & = E\frac{\sum_{i=1}^{n}\frac{(n^2-n)X_i^2}{n^2} - \frac{(n^2 - n)X_iX_j}{n^2}}{n} \\
        & = \frac{\sum_{i=1}^{n}\frac{(n^2-n)E[X_i^2]}{n^2} - \frac{(n^2 - n)E[X_iX_j]}{n^2}}{n} \hskip{2em} \text{(Linear property of expectation)}\\
        & = \frac{\sum_{i=1}^{n}\frac{(n^2-n)(\sigma^2 + \mu^2)}{n^2} - \frac{(n^2 - n)E[X_i]E[X_j]}{n^2}}{n} \hskip{2em} \text{(Because } i \neq j, \forall i, j \text{ and } X_i \text{ & } X_j \text{ are independent)} \\
        & = \frac{\sum_{i=1}^{n}\frac{(n^2-n)(\sigma^2 + \mu^2)}{n^2} - \frac{(n^2 - n)\mu^2}{n^2}}{n} \\
        & = \frac{\sum_{i=1}^{n}\frac{(n^2-n)\sigma^2}{n^2}}{n} \\
        & = \frac{(n-1)\sigma^2}{n} \neq \sigma^2
    \end{split}\\
    \begin{split}
    & \Rightarrow \text{To make the sample variance estimator unbiased, dividing } \sum_{i=1}^{n}(X_i - \bar{X})^2 \text{ by } n-1 \text{ instead of } n \\
    & \Rightarrow \text{Unbiased sample variance estimator: }\hat{\sigma}^2 = \frac{\sum_{i=1}^{n}(X_i - \bar{X})^2}{n-1}
    \end{split}
    $$
</center>

Now, let us examine whether the variance of these estimators could achieve the CRLB.

Regarding the definition of Cramér-Rao Information Inequality, it states that the variance of an estimator $$Var(T(\mathbf{X}))$$ for estimating the parameter $$\boldsymbol\theta$$ is greater than or equal to $$\nabla g(\boldsymbol\theta)^T I(\boldsymbol\theta)^{-1} \nabla g(\boldsymbol\theta)$$. Formally,
<center>$$Var(T(\mathbf{X})) \geqslant \nabla g(\boldsymbol\theta)^T I(\boldsymbol\theta)^{-1} \nabla g(\boldsymbol\theta)$$</center>
where, 
<center>$$g(\boldsymbol\theta) = E[T(\mathbf{X})]$$</center>
<center>
    $$\text{(Expected value of the estimation estimated by the estimator, which is a function of } \boldsymbol\theta \text{)}$$
</center><br>
<center>$$I(\boldsymbol\theta) = E[-\frac{d^2l(\boldsymbol\theta; \mathbf{x})}{d\boldsymbol\theta^2}]$$</center>
<center>$$\text{(Fisher Information matrix of } \boldsymbol\theta \text{)}$$</center>
The proof of CRLB is quite straightforward. It is thus easier to mention about the idea of the proof only as well as a good way to remember this definition. <br><br>
Proof (Sketch):<br>
Consider $$\theta$$ in 1-D case, denote the derivative of the log likelihood function as $$S(\theta) = d\frac{l(\theta; \mathbf{x})}{d\theta}$$, aka the score function, and $$T(\mathbf{x})$$ as the estimator of the parameter $$\theta$$.<br>
Knowing the fact that the Pearson correlation of $$S(\theta)$$ & $$T(\mathbf{x})$$ is between -1 and 1. Specifically,
<center>
    $$
    \begin{split}
    & -1 \leqslant r(S, T) \leqslant 1 \\
    \Leftrightarrow & -1 \leqslant \frac{Cov(S, T)}{\sqrt{Var(S)Var(T)}} \leqslant 1 \\
    & \Rightarrow \frac{Cov(S, T)^2}{Var(S)Var(T)} \leqslant 1 \\
    & \Leftrightarrow Var(T) \geqslant \frac{Cov(S, T)^2}{Var(S)} \\
    & \Leftrightarrow Var(T) \geqslant \frac{Cov(S, T)^2}{I(\theta)} \\
    & \text{(By definition of Information Matrix)}
    \end{split}
    $$
</center>
It remains to show that $$Cov(S, T)$$ is indeed the derivative of $$g(\theta) = E[T(\mathbf{x})]$$ w.r.t $$\theta$$ under regularity condition (Hint: $$Cov(S, T) = E[ST] - \underbrace{E[S]}_{0}E[T]$$). Consequently, 
<center>$$Cov(S, T)^2 = g'(\theta)^2$$</center>

### CRLB for the estimator of $$\mu$$: $$\hat{\mu} = \frac{\sum_{i=1}^{n}x_i}{n}$$

Since $$\mathbf{x} \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2) \Rightarrow l(\mu, \sigma^2; \mathbf{x}) = -\frac{n}{2}ln(2\pi) -\frac{n}{2}ln(\sigma^2) - \sum_{i=1}^{n}\frac{(x_i - \mu)^2}{2\sigma^2}$$
<center>
    $$
    \Rightarrow \begin{cases}
        \frac{dl(\mu, \sigma^2; \mathbf{x})}{d\mu} = \frac{\sum_{i=1}^{n}(x_i - \mu)}{\sigma^2} \\
        \frac{d^2l(\mu, \sigma^2; \mathbf{x})}{d\mu^2} = -\frac{n}{\sigma^2}
    \end{cases}\\
    $$
    $$
    \Rightarrow I(\mu) = E[-\frac{d^2l(\mu, \sigma^2; \mathbf{x})}{d\mu^2}] = \frac{n}{\sigma^2} \tag{8}
    $$
</center>
$$\rightarrow$$ Taking the expectation of $$\hat{\mu} = T(\mathbf{x})$$ w.r.t $$\mu$$ yields: <br>
<center>
    $$
    \begin{split}
        g(\mu) & = E_{\mu}[T(\mathbf{x})] \\
        & \overset{def}{=} \frac{1}{n}\sum_{i=1}^{n} \underbrace{\int_{-\infty}^{+\infty} x_i f(x_i; \mu, \sigma^2)dx_i}_{(\star)} \\
        & = \frac{1}{n}\sum_{i=1}^{n} \mu \\ 
        & \text{(The expected value of a variable } x \sim \mathcal{N}(\mu, \sigma^2) \text{ is equal to } \mu \text{)} = \mu 
    \end{split} \tag{9}
    $$
</center>
$$\Rightarrow T(\mathbf{x}) \text{ is an unbiased estimator of } \mu$$

* Proof for $$(\star) == \mu$$: <br>
$$\rightarrow$$ Using the Moment Generating Function (MGF) of $$X$$ gives:<br>
<center>
    $$
    \begin{split}
        E[e^{tX}] & \overset{def}{=} \int_{-\infty}^{+\infty} e^{tx}\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}dx \\
        & =  \int_{-\infty}^{+\infty} \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{x^2 - 2x(\mu + t\sigma^2) + \mu^2}{2\sigma^2})dx \\
        & =  exp(\frac{-\mu^2 + (\mu + t\sigma^2)^2}{2\sigma^2})\underbrace{\int_{-\infty}^{+\infty} \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{x^2 - 2x(\mu + t\sigma^2) + (\mu + t\sigma^2)^2}{2\sigma^2})dx}_{1} \\
        & = exp(\frac{-\mu^2 + (\mu + t\sigma^2)^2}{2\sigma^2}) = exp(\frac{t^2\sigma^4 + 2t\mu\sigma^2}{2\sigma^2}) = exp(\frac{t^2\sigma^2}{2} + t\mu)\\
        \Rightarrow \frac{dE[e^{tX}]}{dt} & = M^{(1)}(t) = 2t\sigma^2 + \mu \\
        \Rightarrow M^{(1)}(0) & = \mu == E[X]\\
        \text{(Since } E[X] & = \frac{dE[e^{tX}]}{dt} = E[\frac{de^{tX}}{dt}] = E[Xe^{tX}] \\
        & \Rightarrow E[Xe^{tX}]\rvert_{0} = E[X] \text{)}
    \end{split}
    $$
</center>

$$(8), (9) \Rightarrow$$ 
<center>
    $$
    \begin{split}
        Var(T(\mathbf{x})) & \geqslant \frac{g'(\mu)^2}{I(\mu)} \\
        \Leftrightarrow Var(\frac{\sum_{i=1}^{n}x_i}{n}) & \geqslant \frac{1}{\frac{n}{\sigma^2}} \\
        \Leftrightarrow \frac{1}{n^2}Var(\sum_{i=1}^{n}x_i) & \geqslant \frac{\sigma^2}{n} \\
        \Leftrightarrow \frac{\sigma^2}{n} & \geqslant \frac{\sigma^2}{n} \\
    \end{split} 
    $$
</center>
$$\Rightarrow$$ The estimator $$\hat{\mu}$$ of $$\mu$$ attains CRLB.

### CRLB for the estimator of $$\sigma^2$$: $$\hat{\sigma}^2 = s^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n-1}$$

$$\cdot
    \begin{cases}
    \frac{dl(\mu, \sigma^2; \mathbf{x})}{d\sigma^2} = -\frac{n}{2\sigma^2} + \frac{\sum_{i=1}^{n}(x_i - \mu)^2}{2(\sigma^2)^2} \\
    \frac{d^2l(\mu, \sigma^2; \mathbf{x})}{d(\sigma^2)^2} = \frac{n}{2(\sigma^2)^2} - \frac{\sum_{i=1}^{n}(x_i-\mu)^2}{(\sigma^2)^3} \\
    E[\hat{\sigma}^2] = g(\sigma^2) = \sigma^2
    \end{cases}
  $$
<center>
    $$
    \begin{split}
    \Rightarrow I(\sigma^2) & = E[-\frac{d^2l(\mu, \sigma^2; \mathbf{x})}{d(\sigma^2)^2}] = -E[\frac{n}{2(\sigma^2)^2} - \frac{\sum_{i=1}^{n}(x_i-\mu)^2}{(\sigma^2)^3}] = -\frac{n}{2(\sigma^2)^2} + \frac{E[\sum_{i=1}^{n}(x_i-\mu)^2]}{(\sigma^2)^3} \\
    & = -\frac{n}{2(\sigma^2)^2} + \frac{n\sigma^2}{(\sigma^2)^3} = \frac{n}{2(\sigma^2)^2}
    \end{split}
    $$
</center>
According to CRLB definition:<br>
<center>
    $$
    \begin{split}
        Var(s^2) \geqslant \frac{(g'(\sigma^2))^2}{I(\sigma^2)} \\
        \Leftrightarrow Var(s^2) \geqslant \frac{2(\sigma^2)^2}{n}
    \end{split}\tag{10}
    $$
</center>
* Note that the term $$\frac{(n-1)s^2}{\sigma^2} \sim \chi_{n-1} \Rightarrow Var(\frac{(n-1)s^2}{\sigma^2}) = \frac{(n-1)^2}{(\sigma^2)^2}Var(s^2) = 2(n-1)$$<br>
<center>$$\Rightarrow Var(s^2) = \frac{2(\sigma^2)^2}{n-1} \tag{11}$$</center>
$$(10), (11) \Rightarrow$$
<center>
    $$
    \frac{2(\sigma^2)^2}{n-1} \gt \frac{2(\sigma^2)^2}{n}
    $$
</center><br>
Hence, $$\hat{\sigma}^2 = s^2$$ does NOT attain CRLB even it is the unbiased estimator.

## MLE for Normal Linear Model

### Introduction to Linear Model

When it comes to Linear Model, it is not only about representing a linear combination of linear functions (e.g. $$f(x_1, x_2, x_3) = ax_1 + bx_2 + cx_3 + d$$), but also introducing a model with multiple nonlinear functions with associated adjustable parameters (e.g. $$f(x_1, x_2, x_3) = ax_1 + bsin(x_2) + cx_3^5 + d$$). <br><br>
The big plus of Linear Model is the fact that it could utilize many useful class of functions by taking linear combinations of fixed set of nonlinear functions of the input variables/predictors, aka <i>basis functions</i>. What is more, since the model itself explicitly restricts the relationship of indepedent variables to be linear, it is hence easy to interpret the influence of predictors as well as giving some simple but nice analytical properties such as finding optimal parameters for the model could be addressed analytically by Least Squares approach, and relationship between predictors could be transformed from nonlinear into linear by nonlinear functions. <br><br>
However, the number of parameters in Linear Model is fixed and manually defined, which could be insufficient to capture some real world problems requiring thounds of parameters or more.

Mathematically, the form of Linear Model could be defined as
<center>
    $$
    \mathbf{y} = \mathbf{X}\boldsymbol\beta
    $$
</center>
where,
<center>$$\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n\end{pmatrix}_{nx1}$$</center><br>
<center>
        $$\mathbf{X} = 
        \begin{pmatrix} 
            1 & \phi_1(\mathbf{x}_1) & \phi_2(\mathbf{x}_1) & \cdots & \phi_d(\mathbf{x}_1) \\ 
            1 & \phi_1(\mathbf{x}_2) & \phi_2(\mathbf{x}_2) & \cdots & \phi_d(\mathbf{x}_2) \\
            \vdots & &  & \ddots \\
            1 & \phi_1(\mathbf{x}_n) & \phi_2(\mathbf{x}_n) & \cdots & \phi_d(\mathbf{x}_n)
        \end{pmatrix}_{nx(d+1)}
        $$
</center><br>
<center>$$\boldsymbol\beta = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_d \end{pmatrix}_{(d+1)x1}$$</center><br>

$$n$$ is the number of observations and $$d$$ is the number of parameters/predictors of this model.<br>
$$\phi_j(\mathbf{x}_i)$$ is the jth basis function which takes the ith observation as the input.<br>
For every $$x_{i\cdot}$$, it could be understood as the ith observation.<br><br>
* A typical example of Linear Model is Polynomial regression. For instance, $$y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 + \ldots + \beta_dx^d$$. Note that this formula is only accountable for one realization at a time<br>
$$\rightarrow$$ The formula for Polynomial regression of multiple records could be represented in matrix form as follows
<center>
    $$
    \underbrace{\begin{pmatrix}
        y_1 \\
        y_2 \\
        \vdots \\
        y_n
    \end{pmatrix}}_{\mathbf{y}}
    =
    \underbrace{\begin{pmatrix}
        1 & x_1 & x_1^2 & \cdots & x_1^d \\
        1 & x_2 & x_2^2 & \cdots & x_2^d \\
        \vdots & & & \ddots \\
        1 & x_n & x_n^2 & \cdots & x_n^d
    \end{pmatrix}}_{\mathbf{X}}
    \underbrace{\begin{pmatrix}
        \beta_0 \\
        \beta_1 \\
        \beta_2 \\
        \vdots \\
        \beta_d
    \end{pmatrix}}_{\boldsymbol\beta}
    $$
</center>

### Normal Linear Model

Regarding Normal Linear Model, it is nothing but of the form of Linear Model and assumes that the outcomes have the noise follows Normal distribution with the mean $$\boldsymbol\mu_{\text{noise}} = \mathbf{0}$$ and the covariance matrix $$\Sigma$$. Mathematically,
<center>$$\mathbf{y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\epsilon$$</center>
where,
<center>
    $$
    \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \Sigma)
    $$
</center>
For simplicity, we shall treat the noise of all records follows an isotropic Gaussian distribution. That is, $$\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$$ (e.g. every record has the same variance $$\sigma^2$$ and independent from each other).<br>

Moving onto probabilistic approach, given a training data set comprising N observations {$$x_n$$} associated with the outcomes {$$y_n$$}. The goal is to build a predictive model given predictors $$\mathbf{x}$$ for observations only; in other words, we would like to construct the predictive conditional distribution<br><br>
<center>$$f(y_{new}|\mathbf{x}_{new}, \mathbf{X}_{train}, \mathbf{y}_{train})$$</center><br>
It is worth mentioning that the noise of data follows a multivariate Normal distribution based on the definition of Normal Linear Model, the outcome of observations $$\mathbf{Y}$$ therefore a linear transformation of $$\mathcal{N}(\mathbf{0}, \Sigma)$$ which is also a multivatiate Normal distribution with mean $$\mathbf{X}\boldsymbol\beta$$ and covariance matrix $$\sigma^2\mathbf{I}$$. Hence, the conditional distribution of $$\mathbf{y}$$ given $$\mathbf{X}$$ & $$\boldsymbol\beta$$ could be defined as<br><br>
<center>
    $$
    f(\mathbf{y}|\boldsymbol\beta, \mathbf{X}, \sigma^2) = (2\pi\sigma^2)^{-\frac{n}{2}}exp\{-\frac{(\mathbf{y} - \mathbf{X}\boldsymbol\beta)^{T}(\mathbf{y} - \mathbf{X}\boldsymbol\beta)}{2\sigma^2}\}
    $$
</center><br>
With regards to MLE setting, the objective of constructing the predictive conditional distribution $$f(y_{new}|\mathbf{x}_{new}, \mathbf{X}_{train}, \mathbf{y}_{train})$$ is analogous to maximizing the conditional distribution of $$f(\mathbf{y}|\boldsymbol\beta, \mathbf{X}, \sigma^2)$$. Mathematically speaking,
<center>
    $$
    \begin{split}
    \underset{\boldsymbol\beta}{argmax}f(\mathbf{y}|\boldsymbol\beta, \mathbf{X}) & = \underset{\boldsymbol\beta}{argmax} \space (2\pi\sigma^2)^{-\frac{n}{2}}exp\{-\frac{(\mathbf{y} - \mathbf{X}\boldsymbol\beta)^{T}(\mathbf{y} - \mathbf{X}\boldsymbol\beta)}{2\sigma^2}\} \\
    & = \underset{\boldsymbol\beta}{argmin}\frac{(\mathbf{y} - \mathbf{X}\boldsymbol\beta)^{T}(\mathbf{y} - \mathbf{X}\boldsymbol\beta)}{2\sigma^2} \\
    & = \underbrace{\underset{\boldsymbol\beta}{argmin} \frac{(\mathbf{y} - \mathbf{X}\boldsymbol\beta)^{T}(\mathbf{y} - \mathbf{X}\boldsymbol\beta)}{2}}_{(12)}
    \end{split}
    $$
</center><br>
To find the value of $$\boldsymbol\beta$$ that minimizes $$(12)$$, setting the derivative of $$(12)$$ w.r.t $$\boldsymbol\beta$$ to $$\mathbf{0}$$ and solving for $$\boldsymbol\beta$$ is one of the most intuitive way. More precisely, define
<br><br>
<center>
    $$
        L_D(\boldsymbol\beta) = \frac{(\mathbf{y} - \mathbf{X}\boldsymbol\beta)^{T}(\mathbf{y} - \mathbf{X}\boldsymbol\beta)}{2} = \frac{\mathbf{y}^T\mathbf{y} - 2(\mathbf{X}\boldsymbol\beta)^T\mathbf{y} + (\mathbf{X}\boldsymbol\beta)^T(\mathbf{X}\boldsymbol\beta)}{2} \\
        \Rightarrow \nabla L_D(\boldsymbol\beta) = - \mathbf{X}^T\mathbf{y} + \mathbf{X}^T\mathbf{X}\boldsymbol\beta = \mathbf{0} \\
        \Rightarrow \boldsymbol\beta_{\text{MLE}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
    $$
</center>
where, <br>
$$\hskip{5em} L_D(\boldsymbol\beta) = \frac{1}{2}\sum_{i=1}^{n}(y_i - \boldsymbol\beta^T\boldsymbol\phi(\mathbf{x}_i))^2$$ <br><br>
$$\hskip{5em}$$ is called the sum-of-squares error function which is the summation of squares error between $$\mathbf{y}$$ & $$\mathbf{X}\boldsymbol\beta$$ over all given data points in $$D$$.
<br><br>
$$\hskip{5em} \boldsymbol\phi(\mathbf{x}_i) = \begin{pmatrix} \phi_1(\mathbf{x}_i) \\ \phi_2(\mathbf{x}_i) \\ \vdots \\ \phi_d(\mathbf{x}_i) \end{pmatrix} \space$$ is the vector of basis functions
<br><br>
Note that the function $$L_D(\boldsymbol\beta)$$ is of the form of the least squares function. Therefore, it is equivalent to say that searching for optimal parameters by MLE approach is similar to optimizing parameters by least squares method. 

Another way to obtain optimal $$\boldsymbol\beta$$ for $$L_D(\boldsymbol\beta)$$ is to tackle this problem by geometry argument. Specifically, we want to find $$\boldsymbol\beta$$ such that $$\mathbf{X}\boldsymbol\beta$$ is as close to $$\mathbf{y}$$ as possible with fixed $$\mathbf{X}$$. In geometrical perspective, the closest $$\mathbf{X}\boldsymbol\beta$$ to $$\mathbf{y}$$ is indeed the the projection of $$\mathbf{y}$$ onto the vector space $$\mathbf{X}$$. For the ease of interpretation, the image below illustrates why $$\mathbf{X}\boldsymbol\beta = \text{proj}_{\mathbf{X}}(\mathbf{y})$$ is the closest distance between $$\mathbf{y}$$ and $$\mathbf{X}\boldsymbol\beta$$.

$$\hskip{10em}$$![projection-of-y-onto-X](/blog/assets/mle_projection_of_y_onto_X.png)<br>
<div style="display:block; text-align:center">
<i>Figure 1.</i> $$\mathbf{Xh}$$ is the projection of $$\mathbf{y}$$ onto vector space $$\mathbf{X}$$, and $$\mathbf{Xi}$$ or $$\mathbf{Xj}$$ are other candidates of $$\mathbf{X}\boldsymbol\beta$$
</div>
<br>
Consequently, the solution for $$\mathbf{X}\boldsymbol\beta$$, where $$L_D(\boldsymbol\beta)$$ is minimized, should be $$\mathbf{Xh}$$.<br> To be specific, <br>
<center>
    $$
    \mathbf{X}^T(\mathbf{y} - \text{proj}_{\mathbf{X}}(\mathbf{y})) = \mathbf{0} \hskip{2em} \text{(Null space property)}\\
    \Leftrightarrow \mathbf{X}^T(\mathbf{y} - \mathbf{Xh}) = \mathbf{0} \\
    \Leftrightarrow \mathbf{h} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
    $$
</center><br>
$$\rightarrow$$ If $$\mathbf{h}$$ is used as the solution for $$\boldsymbol\beta$$, this result is equivalent to $$\boldsymbol\beta_{\text{MLE}}$$ acquired via solving $$\nabla L_D(\boldsymbol\beta) = \mathbf{0}$$ for $$\boldsymbol\beta$$.

### Regularized least squares

Even the solution obtained by least squares method is an easy-intuitive way, there is still a pitfall in estimating $$\boldsymbol\beta$$ by MLE approach, that is, the model could be overfitting if the it is trained with relatively small data set since the objective is to maximize the likelihood of every observation, which could possibly be heavily influenced by the outliers and adapted to the noise as well. Thereby, one of the most sensible manner is to add another term to the sum-of-squares error function to control overfitting issue, which is so-called <i>regularization term</i>. In particular, the new objective function now becomes <br><br>
<center>
    $$
    \underset{\boldsymbol\beta}{argmin} \space L_D(\boldsymbol\beta) + \lambda L_{\boldsymbol\beta}(\boldsymbol\beta) \tag{13}
    $$
</center>
where <br>
$$\hskip{5em} L_D(\boldsymbol\beta)$$ is a data-dependent error function already defined above.<br><br>
$$\hskip{5em} L_{\boldsymbol\beta}(\boldsymbol\beta)$$ is the regularization term, which serves as the purpose of shrinking model's parameters towards zero unless supported by the data.<br><br>
Generally, the regularizer takes the form<br>
<center>$$\lambda L_{\boldsymbol\beta}(\boldsymbol\beta) = \frac{\lambda}{2}\sum_{i=1}^{d}|\beta_i|^{q} \hskip{2em} \text{(} q \text{ is some positive number)}$$</center><br>
$$\rightarrow$$ Since the regularizer depends on the value of $$\lambda$$; hence, minimizing $$(13)$$ is identical to finding values for $$\boldsymbol\beta$$ to reduce the influence of $$\lambda$$ in order to make the whole expression as minimal as possible. For example, the loss function could be <br><br>
<center>
    $$
    \boldsymbol\beta = \underset{\boldsymbol\beta}{argmin} \space \frac{1}{2}\sum_{i=1}^{n}(y_i - [\beta_0 + \beta_1\phi_1(\mathbf{x_i}) + \beta_2\phi_2(\mathbf{x_i})])^2 + \underbrace{10^4(\beta_0^2 + \beta_1^2 + \beta_2^2)}_{(\ast)} \\
    {\text{(Need to work on choosing } \boldsymbol\beta \text{ to make } (\ast) \text{ as small as possible)}}
    $$
</center><br>
However, the values for hyperparameters $$\lambda$$ and $$q$$ are manually defined, the issue thus becomes tuning hyperparameters $$\lambda$$ and $$q$$. The way of selecting values for those hyperparameters to construct a good-fit model would be shortly discussed in the following part.

For the case of Normal Linear Model, the sum-of-squares error function $$L_D(\boldsymbol\beta)$$ is a quadratic function of $$\boldsymbol\beta$$; hence, choosing $$q=2$$ for the regularizer would make the total error function remain a quadratic function of $$\boldsymbol\beta$$ as same as the previous example. To be exact, the total error function with $$q=2$$ has the following form <br>
<center>
    $$
    \frac{1}{2}(\mathbf{y} - \mathbf{X}\boldsymbol\beta)^T(\mathbf{y} - \mathbf{X}\boldsymbol\beta) + \frac{\lambda}{2}\boldsymbol\beta^T \boldsymbol\beta \tag{14}
    $$
</center><br>
$$\rightarrow$$ Setting the the gradient of the equation $$(14)$$ to $$\mathbf{0}$$ and solving for $$\boldsymbol\beta$$ gives <br><br>
<center>
     $$
     -\mathbf{X}^T\mathbf{y} + \mathbf{X}^T\mathbf{X}\boldsymbol\beta + \lambda\boldsymbol\beta = \mathbf{0} \\
     \Rightarrow \boldsymbol\beta_{\text{MLE}} = (\lambda\mathbf{I} + \mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
     $$
</center><br>
In fact, the method of searching for the optimal solution for $$(14)$$ is similar to using Lagrange multipliers for maximizing/minimizing the function $$L_D(\boldsymbol\beta)$$ subject to the constraint $$\sum_{i=1}^{n}|\beta_i|^q \leqslant \eta$$ for an appropriate value of the parameter $$\eta$$. <br><br>
That's enough for maths. Let's work on an example to see how regularization helps MLE approach avoid overfitting problem!

```python
''' Generate some data from a 2nd-degree polynomial model with the noise followed a Normal distribution '''
train_sample_size = 100
test_sample_size = 20
noise_std = 100
true_beta_0 = 2.5
true_beta_1 = 7.89
true_beta_2 = 4.56
true_beta = np.array([true_beta_0, true_beta_1])

data_x = np.random.rand(train_sample_size + test_sample_size)
data_y = true_beta_0 + (true_beta_1 * data_x) + np.random.normal(loc=0, scale=noise_std, \
                                                                size=train_sample_size + test_sample_size)
data_x_train, data_x_test = data_x[:train_sample_size], data_x[train_sample_size:]
data_y_train, data_y_test = data_y[:train_sample_size], data_y[train_sample_size:]
```

```python
def lm_cross_validation(data_X, data_y, k_folds, lmbda):
    '''
        Perform k-fold cross validation for Linear Model with Ridge regression 
        and compute the average sum of squared errors throughout k scenarios
        
        Parameters
        ----------
        data_X: Input of data, which are also considered as the features of data
        
        data_y: Output of data, which is also known as the outcome of data
        
        k_folds: Number of folds for cross validation procedure
        
        lmda: The hyperparameter of regularization term
        
        Return
        ------
        FloatType: The average sum of squared errors for k subsamples
        
        Notes
        -----
        For more information why cross validation helps in finding good values for hyperparameters tuning, 
        refer to this link -> https://medium.com/datadriveninvestor/k-fold-cross-validation-for-parameter-tuning-75b6cb3214f 
    
    '''
    subsample_size = int(np.ceil(len(data_X) / k_folds))
    avg_total_squares_error = 0
    for k in range(k_folds):
        test_indices = [i + k*subsample_size for i in range(subsample_size)] if k + 1 < k_folds \
                        else [i + k*subsample_size for i in range(len(data_X) - (k*subsample_size))]
        
        X_train, y_train = np.delete(data_X, test_indices, axis=0), np.delete(data_y, test_indices, axis=0)
        
        X_test = data_X[k*subsample_size:(k*subsample_size + subsample_size)] \
                        if k + 1 < k_folds else data_X[k*subsample_size:]
        y_test = data_y[k*subsample_size:(k*subsample_size + subsample_size)] \
                        if k + 1 < k_folds else data_y[k*subsample_size:]
        # Compute the value for total error function
        beta_hat = np.dot(np.dot(np.linalg.inv((np.eye(data_X.shape[1]) * lmbda) + np.dot(X_train.T, X_train)), X_train.T), y_train)
        Xbeta_test = np.dot(X_test, beta_hat)
        avg_total_squares_error += ((1/2) * np.dot((y_test - Xbeta_test).T, (y_test - Xbeta_test))) + \
                                ((1/2) * np.dot(beta_hat.T, beta_hat))

    return avg_total_squares_error / k_folds
```
```python
''' MLE for Normal Linear Model for unregularized total error function '''
# Suppose that the hypothesized model is also a 2nd-degree polynomial model.
X = np.dstack((np.ones(train_sample_size), data_x_train))[0]
y = data_y_train
mle_unregularized_beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
```
```python
''' MLE for Normal Linear Model for regularized total error function '''
# Suppose that the hypothesized model is also a 2nd-degree polynomial model.
best_lmbda, best_score = 2, 9999999 # Initialize values for best_lmbda, best_score 
lmbda_space = np.logspace(-5, 5, 10000) # Domain of lmbda
n_iter = 460 # There is 99% chance that one candidate will be selected from the best 1% candidates in the given lmbda space
k_folds = 4 # Declare number of folds for cross validation procedure
# Perform randomized grid search method for picking good value of lmbda
for i in range(n_iter):
    selected_lmbda = np.random.choice(lmbda_space)
    score = lm_cross_validation(X, y, k_folds, selected_lmbda)
    if score < best_score:
        best_lmbda = selected_lmbda
        best_score = score

mle_regularized_beta = np.dot(np.dot(np.linalg.inv((best_lmbda*np.eye(X.shape[1])) + np.dot(X.T, X)), X.T), y)
```
```python
''' Plot the graph of Normal Linear Models with & without regularized total error function on training data '''
x_range = np.linspace(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1, 10000)
X_range = np.dstack((np.ones(10000), x_range))[0]
y_unregularized_range = np.dot(X_range, mle_unregularized_beta)
y_regularized_range = np.dot(X_range, mle_regularized_beta)
y_true_range = np.dot(X_range, true_beta)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_unregularized_range, color='blue', label='Model with regularization-free')
plt.plot(x_range, y_regularized_range, color='red', label='Model with regularization')
plt.plot(x_range, y_true_range, color='green', label='True linear model')
plt.plot(data_x_train, data_y_train, '+', markersize=5, clip_on=False, label='Data point')
plt.title('Visualization of unregularized linear model vs regularized linear model optimized by MLE')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Compute the total squared errors with testing data to see which model performs better
X_test = np.dstack((np.ones(len(data_x_test)), data_x_test))[0]
score_unregularized_model = np.dot((data_y_test - np.dot(X_test, mle_unregularized_beta)).T, \
                                           data_y_test - np.dot(X_test, mle_unregularized_beta))
score_regularized_model = np.dot((data_y_test - np.dot(X_test, mle_regularized_beta)).T, \
                                           data_y_test - np.dot(X_test, mle_regularized_beta))
print('Sum-of-squares value for unregularized model: {}'.format(score_unregularized_model))
print('Sum-of-squares value for regularized model: {}'.format(score_regularized_model))
```
![Comparison between regularized model vs unregularized model](/blog/assets/mle_unregularized_model_vs_regularized_model.png)

> Sum-of-squares value for unregularized model: 299335.89017885574 <br>
> Sum-of-squares value for regularized model: 289619.40085747186

As can be seen from the illustration above, the model with regularization term tends to approximate much better and have smaller sum-of-squares value compared to that of the model with unregularized cost function. <br><br>
For more information why minimizing regularized total error function results in better approximation. Let's look at the contour plot of unregularized total error function w.r.t $$\beta_0$$ & $$\beta_1$$ and the constraint defined by regularization term to see the feasible domain of solution.

```python
'''Contour plot for unregularized total error function subject to the constraint defined by regularization term'''
# Since the constraint defined by regularization term is of the form of a circle equation
# And the solution must be the tangent of both total error function and the constraint function
# Hence, the radius where the constraint function meets the total error function should be sqrt(beta_0^2 + beta_1^2) 
constraint_radius = np.sqrt((mle_regularized_beta[0] ** 2) + (mle_regularized_beta[1] ** 2))

# Initialize mesh points for drawing contour plot of unregularized total error function
mesh_size = 100
beta_0 = np.linspace(-50, 50, mesh_size)
beta_1 = np.linspace(-50, 50, mesh_size)
Beta_0, Beta_1 = np.meshgrid(beta_0, beta_1)

total_error_function = np.eye(mesh_size) # Initialize values for total_error_function
for i in range(mesh_size):
    for j in range(mesh_size):
        beta = np.array([Beta_0[i, j], Beta_1[i, j]])
        Xbeta = np.dot(X, beta)
        total_error_function[i, j] = (1/2) * np.dot((y - Xbeta).T, (y - Xbeta))

sse_for_regularized_beta = (1/2) * np.dot((y - np.dot(X, mle_regularized_beta)).T, (y - np.dot(X, mle_regularized_beta)))
sse_for_unregularized_beta = (1/2) * np.dot((y - np.dot(X, mle_unregularized_beta)).T, \
                                            (y - np.dot(X, mle_unregularized_beta))) + 100

fig, ax = plt.subplots(figsize=(10, 6))

circle = plt.Circle((0, 0), constraint_radius, color='blue', fill=False)
ax.add_artist(circle)

cost_function_contour = ax.contour(Beta_0, Beta_1, total_error_function, \
           [sse_for_unregularized_beta, sse_for_regularized_beta, 555000, 585000, 700000], colors='red')

optimal_beta, = ax.plot(mle_regularized_beta[0], mle_regularized_beta[1], 'o', markersize=5, color='green')

contour_legend, _ = cost_function_contour.legend_elements()

ax.set_xlim((-50, 50))
ax.set_ylim((-50, 50))
ax.set_xlabel('beta_0')
ax.set_ylabel('beta_1')
ax.legend((contour_legend[0], circle, optimal_beta), ('contour lines for total error function', \
                                                         'regularization constraint', \
                                                         'optimal beta'))
ax.set_title('Contour plot for unregularized total error function and regularization constraint')
plt.show()
```
![MLE contour plot](/blog/assets/mle_contour_plot.png)

So far, introducing regularization term is a great way to overcome severe overfitting problem by MLE approach, although choosing the sensible model (e.g selecting basis functions for Normal Linear Model) is generally more important since it reflects the overal behaviour of the model for a particular problem. For the next topic, we shall discuss the <a href="#">Bayesian treatment of linear regression</a> and the reasons for choosing this approach over MLE fashion.

## Reference

[1] Christopher M. Bishop. Pattern Recognition and Machine Learning, Section 3.1, Chapter 3. <br>
[2] D.P. Kroese and J.C.C. Chan. Statistical Modeling and Computation, Chapter 6. <br>
[3] <a href="https://people.missouristate.edu/songfengzheng/Teaching/MTH541/Lecture%20notes/Fisher_info.pdf"> Fisher Information and Cramér-Rao Bound. </a>
