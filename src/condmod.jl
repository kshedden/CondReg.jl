using LinearAlgebra, Optim, FiniteDifferences, Distributions, Missings

abstract type AbstractConditionalModel <: LinPredModel end

struct ConditionalLogitModel{S<:Integer,T<:Real} <: AbstractConditionalModel

    "`y`: response vector"
    y::Vector{S}

    pp::LinPred

    "`ys`: sum of y within each group"
    ys::Vector{S}

    "`wts`: weights (per-group not per-observation)"
    wts::Vector{T}

    "`g`: group indicators, must be sorted"
    g::AbstractVector

    "`gix`: each column contains the first and last index of a group"
    gix::Matrix{Int}

    "`ngrp`: number of groups"
    ngrp::Int
end


mutable struct DensePred{T<:Real} <: LinPred

    "`X`: covariates"
    X::Matrix{T}

    "`Xty`: y' times X within each group"
    Xty::Matrix{T}

    "`beta0`: coefficients"
    beta0::Vector{T}

    "`cov`: sampling covariance matrix of coefficients"
    cov::Matrix{T}
end


function ConditionalLogitModel(
    X::AbstractMatrix,
    y::AbstractVector,
    g::AbstractVector;
    wts::AbstractVector = zeros(0),
)
    g = try
        disallowmissing(g)
    catch MethodError
        error("Missing values not allowed in group variable")
    end

    gix, mg = groupix(g)
    ngrp = size(gix, 2)

    # Sufficient statistics, y' 1 and y' X for each greoup
    p = size(X, 2)
    Xty = zeros(ngrp, p)
    ys = zeros(eltype(y), ngrp)
    for i = 1:ngrp
        i1, i2 = gix[1, i], gix[2, i]
        if var(y[i1:i2]) < 1e-8
            @warn "The response appears to be constant in the $(i)th group."
        end
        ys[i] = sum(y[i1:i2])
        Xty[i, :] = y[i1:i2]' * X[i1:i2, :]
    end

    pp = DensePred(X, Xty)

    return ConditionalLogitModel(y, pp, ys, wts, g, gix, ngrp)
end

function DensePred(X::Matrix{T}, Xty::Matrix{T}) where {T<:Real}
    p = size(X)[2]
    return DensePred(X, Xty, zeros(p), zeros(p, p))
end

function _fit!(
    m::ConditionalLogitModel,
    verbose::Bool,
    maxiter::Int,
    atol::Float64,
    rtol::Float64,
    start,
)
    X = m.pp.X
    p = size(X)[2]

    u, _, _ = svd(X)
    if isapprox(norm(sum(u, dims = 1)), sqrt(size(X, 1)), atol = 1e-8)
        @warn("Conditional logit models should not contain an intercept.")
    end

    # Wrap the log-likelihood and score functions
    # for minimization.
    f = x -> -loglike(m, x)
    g! = function (g, x)
        score(m, x, g)
        g .*= -1
    end

    if isnothing(start)
        start = rand(p) .- 0.5
    end

    r = optimize(
        f,
        g!,
        start,
        LBFGS(),
        Optim.Options(iterations = maxiter, show_trace = verbose),
    )

    if !Optim.converged(r)
        println("CondReg fitting did not converge")
    end

    b = Optim.minimizer(r)
    m.pp.beta0 .= b

    # Use numerical differentiation to get the Hessian.
    score1 = function (x)
        g = zeros(length(x))
        score(m, x, g)
        return g
    end
    hess = jacobian(central_fdm(12, 1), score1, b)[1]
    hess = Symmetric(-(hess + hess') ./ 2)
    m.pp.cov = inv(hess)

    return m
end

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

    if !(size(X, 1) == size(y, 1) == size(g, 1))
        throw(DimensionMismatch("Number of rows in X, y and g must match"))
    end

    y = try
        Int64.(y)
    catch InexactError
        throw(InexactError("Response values must be integers"))
    end

    c = ConditionalLogitModel(X, y, g)

    return dofit ? fit!(c; fitargs...) : c
end

function groupix(g::AbstractVector)::Tuple{Matrix{Int},Int}

    if !issorted(g)
        error("Group vector is not sorted")
    end

    ii = Int[]
    b, mx = 1, 0
    for i = 2:length(g)
        if g[i] != g[i-1]
            push!(ii, b, i - 1)
            mx = i - b > mx ? i - b : mx
            b = i
        end
    end
    push!(ii, b, length(g))
    mx = length(g) - b + 1 > mx ? length(g) - b + 1 : mx
    ii = reshape(ii, 2, div(length(ii), 2))

    return tuple(ii, mx)
end


function loglike(m::ConditionalLogitModel, params)::Float64

    X = m.pp.X
    Xty = m.pp.Xty
    exb = exp.(X * params)
    wts = m.wts

    ll = 0.0
    for g = 1:m.ngrp

        # In the recursions, dh may be called multiple times with the
        # same arguments, so we memoize the results.
        memo = Dict{Tuple{Int64,Int64},Float64}()

        w = length(wts) > 0 ? wts[g] : 1.0

        i1, i2 = m.gix[1, g], m.gix[2, g]
        ll += w * dot(Xty[g, :], params)
        denom = dh(i2 - i1 + 1, m.ys[g], exb[i1:i2], memo)
        ll -= w * log(denom)
    end

    return ll
end

function score(m::ConditionalLogitModel, params, scr)

    X = m.pp.X
    Xty = m.pp.Xty
    exb = exp.(X * params)
    wts = m.wts

    p = size(X)[2]
    @assert length(scr) == p
    scr .= 0

    for g = 1:m.ngrp

        # ds may be called multiple times in the recursions with the
        # same arguments, so memoize the results.
        memo = Dict{Tuple{Int64,Int64},Tuple{Float64,Vector{Float64}}}()

        w = length(wts) > 0 ? wts[g] : 1.0

        i1, i2 = m.gix[1, g], m.gix[2, g]
        d, h = ds(i2 - i1 + 1, m.ys[g], p, X[i1:i2, :], exb[i1:i2], memo)
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

function vcov(m::AbstractConditionalModel)
    return m.pp.cov
end

function coeftable(mm::AbstractConditionalModel; level::Real = 0.95)
    cc = coef(mm)
    se = sqrt.(diag(vcov(mm)))
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1 - level) / 2)
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    CoefTable(
        hcat(cc, se, zz, p, cc + ci, cc - ci),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        ["x$i" for i = 1:size(mm.pp.X, 2)],
        4,
        3,
    )
end

function StatsBase.fit!(
    m::AbstractConditionalModel;
    verbose::Bool = false,
    maxiter::Integer = 50,
    atol::Real = 1e-6,
    rtol::Real = 1e-6,
    start = nothing,
    kwargs...,
)
    _fit!(m, verbose, maxiter, atol, rtol, start)
end

clogit(F, D, args...; kwargs...) = fit(ConditionalLogitModel, F, D, args...; kwargs...)
