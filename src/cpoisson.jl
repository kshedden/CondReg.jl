struct ConditionalPoissonModel <: AbstractConditionalModel
    cm::ConditionalModel
end

coef(m::ConditionalPoissonModel) = coef(m.cm)
vcov(m::ConditionalPoissonModel) = vcov(m.cm)

"""
    fit(ConditionalPoissonModel, X, y, g; <keyword arguments>)

Fit a conditional Poisson regression model to the response vector `y`
given the covariates in the columns of `X`.  Individuals belong to groups
as given in `g`.  The values in `g` must be sorted.

# Keyword Arguments
- `offset::Vector`: If present, an offset that is added to the linear predictor.
- `dofit::Bool`: If true, fit the model, otherwise return an unfit model.
"""
function fit(
    ::Type{M},
    X::Matrix,
    y::AbstractVector,
    g::AbstractVector;
    offset = zeros(0),
    dofit::Bool = true,
    fitargs...,
) where {M<:ConditionalPoissonModel}

    if !(size(X, 1) == length(y) == length(g))
        n1, n2, n3 = size(X, 1), length(y), length(g)
        msg = @sprintf("Number of rows in X (%d), y (%d), and g (%d) must match", n1, n2, n3)
        throw(DimensionMismatch(msg))
    end

    if (length(offset) > 0) && (length(offset) != length(y))
        throw(DimensionMismatch("If offset is provided its length must equal the length of y."))
    end

    y = try
        Int64.(y)
    catch InexactError
        throw(InexactError("Response values must be integers"))
    end

    c = ConditionalPoissonModel(ConditionalModel(X, y, g; offset=offset))
    return dofit ? fit!(c; fitargs...) : c
end


function loglike(m::ConditionalPoissonModel, params::AbstractVector)::Float64

    cm = m.cm
    X = cm.pp.X
    Xty = cm.pp.Xty
    linpred = X * params
    if length(cm.offset) > 0
        linpred .+= cm.offset
    end
    exb = exp.(linpred)
    wts = cm.wts
    ll = 0.0
    for g in 1:cm.ngrp
        ll += dot(Xty[g, :], params)
        i1, i2 = cm.gix[:, g]
        ll -= cm.ys[g] * log(sum(exb[i1:i2]))
    end

    return ll
end

function score(m::ConditionalPoissonModel, params::AbstractVector, scr::AbstractVector)

    cm = m.cm
    X = cm.pp.X
    Xty = cm.pp.Xty
    linpred = X * params
    if length(cm.offset) > 0
        linpred .+= cm.offset
    end
    exb = exp.(linpred)
    wts = cm.wts

    scr .= 0
    for g = 1:cm.ngrp
        scr .+= Xty[g, :]
        i1, i2 = cm.gix[:, g]
        scr .-= cm.ys[g] * vec(exb[i1:i2]' * X[i1:i2, :]) / sum(exb[i1:i2])
    end
end

function hessian(m::ConditionalPoissonModel, params::AbstractVector, hess::AbstractMatrix)

    cm = m.cm
    X = cm.pp.X
    Xty = cm.pp.Xty
    linpred = X * params
    if length(cm.offset) > 0
        linpred .+= cm.offset
    end
    exb = exp.(linpred)
    wts = cm.wts

    hess .= 0
    for g = 1:cm.ngrp
        i1, i2 = cm.gix[:, g]
        xx = X[i1:i2, :]
        ee = exb[i1:i2]
        numer = vec(ee' * xx)
        denom = sum(ee)
        ddenom = numer
        hess .-= cm.ys[g] * (denom * xx' * diagm(ee) * xx - numer * numer') ./ denom^2
    end
end

function Base.show(io::IO, m::ConditionalPoissonModel)
    println(io, coeftable(m))
end
