using CondReg, Test, LinearAlgebra, StableRNGs

function gen_simple(n::Int)

    rng = StableRNG(123)

    X = randn(rng, n, 3)

    g = zeros(Int, n)
    for i = 1:n
        g[i] = div(i - 1, 20) + 1
    end

    lp = X[:, 1] - X[:, 3]
    mn = 1 ./ (1 .+ exp.(-lp))
    y = [rand(rng) < mn[i] ? 1 : 0 for i = 1:n]

    return y, X, g
end

@testset "numgrad" begin

    y, X, g = gen_simple(1000)

    c = fit(ConditionalLogitModel, X, y, g; dofit = false)

    x = [1, 0, -1]
    grad = zeros(3)
    score(c, x, grad)

    ee = 1e-6
    ngrad = zeros(3)
    ll0 = loglike(c, x)
    for j = 1:3
        e = zeros(3)
        e[j] += ee
        ngrad[j] = (loglike(c, x + e) - ll0) / ee
    end

    @test isapprox(grad, ngrad, atol = 1e-4, rtol = 1e-3)
end

@testset "simplefit" begin

    y, X, g = gen_simple(1000)

    # Test log-likelihood
    pa = [1.0, 0.0, -1.0]
    m = fit(ConditionalLogitModel, X, y, g, dofit = false)
    @test isapprox(loglike(m, pa), -448.9648679975831, atol = 1e-4, rtol = 1e-4)

    # Test score
    gr = zeros(3)
    score(m, pa, gr)
    @test isapprox(
        gr,
        [-13.642599916175753, -9.580236049456111, -13.720581704575254],
        atol = 1e-4,
        rtol = 1e-4,
    )

    r = fit(ConditionalLogitModel, X, y, g)
    @test isapprox(
        coef(r),
        [0.9205497138848794, -0.05772870577778407, -1.0788357395092034],
        atol = 1e-4,
        rtol = 1e-4,
    )

    se = sqrt.(diag(vcov(r)))
    @test isapprox(
        se,
        [0.09002707019324953, 0.07615874517091309, 0.09285127671105863],
        atol = 1e-4,
        rtol = 1e-4,
    )

    score(m, coef(r), gr)
    @test isapprox(gr, Float64[0, 0, 0], atol = 1e-8, rtol = 1e-8)
end
