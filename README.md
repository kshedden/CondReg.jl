# Overview

CondReg.jl: Conditional regression models in Julia
--

The CondReg.jl Julia package implements conditional logistic and conditional
Poisson regression. These are techniques for regression analysis with grouped
data, where each group has a distinct and unknown intercept.

## Usage:

```julia
using CondReg, DataFrames, StatsModels

# da is a DataFrame containing y, x1, x2, and id.
m = clogit(@formula(y ~ x1 + x2), da, da[:, id])

m = cpoisson(@formula(y ~ x1 + x2), da, da[:, id])
```
