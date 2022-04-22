struct ConditionalLogitModel <: AbstractConditionalModel
    cm::ConditionalModel
end

coef(m::ConditionalLogitModel) = coef(m.cm)
vcov(m::ConditionalLogitModel) = vcov(m.cm)

"""
	fit(ConditionalLogitModel, X, y, g; <keyword arguments>)

Fit a conditional logistic regression model to the response vector `y`
given the covariates in the columns of `X`.  Individuals belong to groups
as given in `g`.  The values in `g` must be sorted.

# Keyword Arguments
- `dofit::Bool`: If true, fit the model, otherwise return an unfit model.
"""
function fit(
    ::Type{M},
    X::Matrix,
    y::AbstractVector,
    g::AbstractVector;
    dofit::Bool = true,
    fitargs...,
) where {M<:ConditionalLogitModel}

    if !(size(X, 1) == length(y) == length(g))
        throw(DimensionMismatch("Number of rows in X, y and g must match"))
    end

    y = try
        Int64.(y)
    catch InexactError
        throw(InexactError("Response values must be integers"))
    end

    c = ConditionalLogitModel(ConditionalModel(X, y, g))
    return dofit ? fit!(c; fitargs...) : c
end

function loglike(m::ConditionalLogitModel, params)::Float64

    cm = m.cm
    X = cm.pp.X
    Xty = cm.pp.Xty
    exb = exp.(X * params)
    wts = cm.wts

    ll = 0.0
    for g = 1:cm.ngrp

        # In the recursions, dh may be called multiple times with the
        # same arguments, so we memoize the results.
        memo = Dict{Tuple{Int64,Int64},Float64}()

        w = length(wts) > 0 ? wts[g] : 1.0

        i1, i2 = cm.gix[:, g]
        ll += w * dot(Xty[g, :], params)
        denom = dh(i2 - i1 + 1, cm.ys[g], exb[i1:i2], memo)
        ll -= w * log(denom)
    end

    return ll
end

function score(m::ConditionalLogitModel, params, scr)

    cm = m.cm
    X = cm.pp.X
    Xty = cm.pp.Xty
    exb = exp.(X * params)
    wts = cm.wts

    p = size(X)[2]
    @assert length(scr) == p
    scr .= 0

    for g = 1:cm.ngrp

        # ds may be called multiple times in the recursions with the
        # same arguments, so memoize the results.
        memo = Dict{Tuple{Int64,Int64},Tuple{Float64,Vector{Float64}}}()

        w = length(wts) > 0 ? wts[g] : 1.0

        i1, i2 = cm.gix[:, g]
        d, h = ds(i2 - i1 + 1, cm.ys[g], p, X[i1:i2, :], exb[i1:i2], memo)
        scr .+= w * Xty[g, :] - h ./ d
    end
end

# Helper function for log-likelihood recursions
function dh(
    t::Int64,
    k::Int64,
    exb::Vector{Float64},
    memo::Dict{Tuple{Int64,Int64},Float64},
)::Float64
    if t < k
        return 0
    end
    if k == 0
        return 1
    end

    u = get(memo, (t, k), nothing)
    if !isnothing(u)
        return u
    end

    v = dh(t - 1, k, exb, memo) + dh(t - 1, k - 1, exb, memo) * exb[t]
    memo[(t, k)] = v

    return v
end

# Helper function for score recursions
function ds(
    t::Int,
    k::Int,
    p::Int,
    ex::Matrix{Float64},
    exb::Vector{Float64},
    memo::Dict{Tuple{Int64,Int64},Tuple{Float64,Vector{Float64}}},
)::Tuple{Float64,Vector{Float64}}

    if t < k
        return 0, zeros(p)
    end
    if k == 0
        return 1, zeros(p)
    end

    u = get(memo, tuple(t, k), nothing)
    if !isnothing(u)
        return u
    end

    h = exb[t]
    a, b = ds(t - 1, k, p, ex, exb, memo)
    c, e = ds(t - 1, k - 1, p, ex, exb, memo)
    d = c * h * ex[t, :]

    u, v = a + c * h, b + d + e * h
    memo[(t, k)] = (u, v)

    return u, v
end

clogit(F, D, args...; kwargs...) = fit(ConditionalLogitModel, F, D, args...; kwargs...)
