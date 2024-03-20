# Comments for the reparameterization one:

1) Like every machine learning algorithm, if you have a high beta(like 0.95) it will be steadlier but slower to arrive the optima. If you have a low beta(like 0.8) it will be faster to arrive the optima but after that it will fluctuate.

2) The slogan of differentiable rendering: "The integrand is not differentiable, but the integral is differentiable". But here we do not have any integral:(

3) A possible way to construct a integral is still REINFORCED: for two options (or more), we have a random variable for p and 1-p. Outcome is p*f(state 1) + (1-p)*f(state 2), then p can be optimized.
