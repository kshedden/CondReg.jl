# Overview

CondReg.jl: Conditional logistic regression in Julia
--

The CondReg.jl Julia package implements conditional logistic regression.
This is a technique for regression analysis of a binary outcome with grouped 
data, where each group has a distinct and unknown intercept.

## Usage:

```julia
using CondReg, DataFrames, StatsModels

# da is a DataFrame containing y, x1, x2, and id.
m = clogit(@formula(y ~ x1 + x2), da, da[:, id])
```
