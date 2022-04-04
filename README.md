# Overview

CondReg.jl: Conditional logistic regression in Julia
--

## Usage:

```julia
# da is a DataFrame containing y, x1, x2, and id.
m = clogit(@formula(y ~ x1 + x2), da, da[:, id])
```
