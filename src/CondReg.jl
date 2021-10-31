module CondReg

using StatsBase
using GLM: LinPredModel, LinPred

import StatsBase: coef, coeftable, vcov, stderr, fit

export fit, fit!, ConditionalModel, ConditionalLogitModel, loglike, score
export coef, stderr, vcov

include("condmod.jl")

end
