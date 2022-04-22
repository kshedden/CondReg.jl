module CondReg

using Printf
using StatsBase, StatsModels
using GLM: LinPredModel, LinPred
using LinearAlgebra, Optim, FiniteDifferences, Distributions, Missings

import StatsBase: coef, coeftable, vcov, stderr, fit

export fit,
    fit!, ConditionalModel, ConditionalLogitModel, ConditionalPoissonModel, clogit, cpoisson
export loglike, hessian, score, coef, stderr, vcov, coeftable, drop_intercept

include("defs.jl")
include("clogit.jl")
include("cpoisson.jl")

end
