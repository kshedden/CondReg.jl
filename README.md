# Overview

# CondReg.jl: Conditional regression models in Julia

# The CondReg.jl Julia package implements conditional logistic and conditional
# Poisson regression. These are techniques for regression analysis with grouped
# data, where each group has a distinct and unknown intercept.

# Usage:

First we generate grouped Bernoulli data with group-specific random
intercepts, and fit a logistic regression model using conditional
logistic regression.

````julia
using CondReg, DataFrames, StatsModels, StableRNGs, Distributions
import StatsBase

rng = StableRNG(123)

n = 500 # sample size
m = 50 # number of groups
id = StatsBase.sample(rng, 1:m, n, replace=true)
sort!(id)
u = 0.25*randn(rng, m) # group intercepts
x1 = randn(rng, n) # first covariate
x2 = randn(rng, n) # second covariate
````

A linear predictor, with group intercepts

````julia
lp = x1 + u[id]
````

A vector of probabilites

````julia
pr = 1 ./ (1 .+ exp.(-lp))
````

Vector of binary outcomes

````julia
y = rand.(rng, Binomial.(1, pr))
````

da is a DataFrame containing y, x1, x2, and id.

````julia
da = DataFrame(y=y, x1=x1, x2=x2, id=id)
````

Fit a conditional logit model

````julia
m1 = fit(ConditionalLogitModel, @formula(y ~ x1 + x2), da, da[:, :id])
````

Next we generate data from a Poisson model with group-specific
intercepts and fit a model using conditional Poisson regression.

A vector of expected values

````julia
ev = exp.(lp)
y = rand.(rng, Poisson.(ev))
da[:, :y] = y

m2 = fit(ConditionalPoissonModel, @formula(y ~ x1 + x2), da, da[:, :id])
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

