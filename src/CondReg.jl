module CondReg

using Printf
using DataFrames
using StatsModels
using StatsBase: CoefTable
using LinearAlgebra, Optim, Distributions
using FiniteDifferences

import StatsAPI: coef, coeftable, vcov, stderr, fit

export fit, fit!, ConditionalModel, ConditionalLogitModel, ConditionalPoissonModel
export loglike, hessian, score, coef, stderr, vcov, coeftable, drop_intercept

include("defs.jl")
include("clogit.jl")
include("cpoisson.jl")

end
