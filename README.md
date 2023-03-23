# Background extrapolation for anomaly detection.

# Extrapolation methods

 - Density estimation with Normalizing Flow
 - Unbinned Rerighting MC $\to$ data
 - ABCD

## Density estimation with Normalizing Flow

### Toy example

Use variables $\alpha$ and $\beta$ to define SR.

$$
p(\alpha)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{\alpha^{2}}{2}}
$$

$$
p(\beta)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{\beta^{2}}{2}}
$$

Use variable $y$ which is a function of $\alpha$ and $\beta$ as to do the likelihood fit.

$$
P(y) = N(k(\sin{\theta}\alpha + \cos{\theta}\beta,) 1),
$$

where $k$ and $\theta$ can be set to decrease or increases the dependence of $\alpha$ and $\beta$.
