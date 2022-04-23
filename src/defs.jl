abstract type AbstractConditionalModel <: LinPredModel end

# Conditional models should not contain an intercept
StatsModels.drop_intercept(::Type{<:AbstractConditionalModel}) = true

"""
    ConditionalModel

Information specifying a regression model to be fit with using the
conditional likelihood.  This struct is intended to be embedded into
specific conditional models such as ConditionalLogitModel.
"""
struct ConditionalModel{S<:Integer,T<:Real} <: LinPredModel

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

    "`offset`: offset is added to the linear predictor"
    offset::Vector{T}
end

mutable struct DensePred{T<:Real} <: LinPred

    "`X`: covariates"
    X::Matrix{T}

    "`Xty`: y' times X within each group"
    Xty::Matrix{T}

    "`oty`: y dotted with the offset in each group"
    oty::Vector{T}

    "`beta0`: coefficients"
    beta0::Vector{T}

    "`cov`: sampling covariance matrix of coefficients"
    cov::Matrix{T}
end

function DensePred(X::Matrix{T}, Xty::Matrix{T}, oty::Vector{T}) where {T<:Real}
    p = size(X)[2]
    return DensePred(X, Xty, oty, zeros(p), zeros(p, p))
end

# Return the indices defining each consecutive run of constant values
# in the sorted vector g.
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

function ConditionalModel(
    X::AbstractMatrix,
    y::AbstractVector,
    g::AbstractVector;
    wts::AbstractVector = zeros(0),
    offset::AbstractVector = zeros(0)
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
    oty = length(offset) > 0 ? zeros(ngrp) : zeros(0)
    ys = zeros(eltype(y), ngrp)
    xvar = zeros(Int, p)
    for g in 1:ngrp
        i1, i2 = gix[:, g]
        if var(y[i1:i2]) < 1e-8
            @warn "The response appears to be constant in the $(i)th group."
        end
        ys[g] = sum(y[i1:i2])
        Xty[g, :] = y[i1:i2]' * X[i1:i2, :]
        if length(offset) > 0
            oty[g] = dot(y[i1:i2], offset[i1:i2])
        end
        for j in 1:p
            xvar[j] += var(X[i1:i2, j]) > 1e-8 ? 1 : 0
        end
    end

    jj = findall(xvar .== 0)
    if length(jj) > 0
        jx = join(string.(jj), ", ")
        c = length(jj) > 1 ? "s" : ""
        msg = @sprintf("Variable%s %s appear to be constant in all groups.", c, jx)
        @warn(msg)
    end

    pp = DensePred(X, Xty, oty)

    return ConditionalModel(y, pp, ys, wts, g, gix, ngrp, offset)
end

function _fit!(
    m::T,
    verbose::Bool,
    maxiter::Int,
    atol::Float64,
    rtol::Float64,
    start,
) where {T<:AbstractConditionalModel}
    cm = m.cm
    X = cm.pp.X
    p = size(X)[2]

    u, _, _ = svd(X)
    if isapprox(norm(sum(u, dims = 1)), sqrt(size(X, 1)), atol = 1e-8)
        @warn("Conditional models should not contain an intercept.")
    end

    # Wrap the log-likelihood and score functions
    # for minimization.
    f = x -> -loglike(m, x)
    g! = function (g, x)
        score(m, x, g)
        g .*= -1
    end

    if isnothing(start)
        start = 1e-4 * (rand(p) .- 0.5) ./ std(X, dims = 1)[:]
    end

    r = optimize(
        f,
        g!,
        start,
        LBFGS(),
        Optim.Options(iterations = maxiter, show_trace = verbose),
    )

    if !Optim.converged(r)
        @warn("CondReg fitting did not converge")
    end

    b = Optim.minimizer(r)
    cm.pp.beta0 .= b

    # Standard errors via Fisher information
    hess = zeros(length(b), length(b))
    hessian(m, b, hess)
    cm.pp.cov = inv(-hess)

    return m
end

# Use numerical differentiation to get the Hessian.
function hessian(m, b, hess)
    score1 = function (x)
        g = zeros(length(x))
        score(m, x, g)
        return g
    end
    hess .= jacobian(central_fdm(12, 1), score1, b)[1]
    hess .= (hess + hess') ./ 2
end

function vcov(m::ConditionalModel)
    return m.pp.cov
end

function coeftable(mm::AbstractConditionalModel; level::Real = 0.95)
    coeftable(mm.cm)
end

function coeftable(mm::ConditionalModel; level::Real = 0.95)
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
    m::T;
    verbose::Bool = false,
    maxiter::Integer = 50,
    atol::Real = 1e-6,
    rtol::Real = 1e-6,
    start = nothing,
    kwargs...,
) where {T<:AbstractConditionalModel}
    _fit!(m, verbose, maxiter, atol, rtol, start)
end
