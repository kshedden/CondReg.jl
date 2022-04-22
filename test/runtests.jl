using CondReg, Test, LinearAlgebra, StableRNGs, Distributions, DataFrames
using StatsModels, FiniteDifferences

function gen_simple_clogit(n::Int)

    rng = StableRNG(123)

    X = randn(rng, n, 3)
    g = [div(i - 1, 20) + 1 for i = 1:n]

    lp = X[:, 1] - X[:, 3]
    mn = 1 ./ (1 .+ exp.(-lp))
    y = [rand(rng) < mn[i] ? 1 : 0 for i = 1:n]

    return y, X, g
end

function gen_simple_cpoisson(n::Int)

    rng = StableRNG(123)

    X = randn(rng, n, 3)
    g = [div(i - 1, 20) + 1 for i = 1:n]

    lp = X[:, 1] - X[:, 3]
    mn = exp.(lp)
    y = rand.(rng, Poisson.(mn))

    return y, X, g
end

@testset "formula intercept" begin

    y, X, g = gen_simple_cpoisson(100)
    df = DataFrame(:y => y, :g => g)
    for j = 1:size(X, 2)
        df[:, Symbol("x$(j)")] = X[:, j]
    end

    # Check that there is no intercept in the model
    f = @formula(y ~ x1 + x2 + x3)
    m = cpoisson(f, df, g)
    @test length(coefnames(m)) == 3
end

@testset "numgrad clogit" begin

    y, X, g = gen_simple_clogit(1000)

    for include_offset in [false, true]
		args = include_offset ? [:offset=>randn(length(y))] : []
	    c = fit(ConditionalLogitModel, X, y, g; dofit = false, args...)

	    x = Float64[1, 0, -1]
    	agrad = zeros(3)
    	score(c, x, agrad)

	    # Check the gradient
   	 	ngrad = grad(central_fdm(5, 1), y -> loglike(c, y), x)[1]
    	@test isapprox(agrad, ngrad, atol = 1e-4, rtol = 1e-3)
	end
end

@testset "numgrad cpoisson" begin

    y, X, g = gen_simple_cpoisson(1000)

    for include_offset in [false, true]
		args = include_offset ? [:offset=>randn(length(y))] : []
	    c = fit(ConditionalPoissonModel, X, y, g; dofit = false, args...)

	    x = Float64[1, 0, -1]
    	agrad = zeros(3)
    	score(c, x, agrad)

	    # Check the gradient
    	ngrad = grad(central_fdm(5, 1), y -> loglike(c, y), x)[1]
    	@test isapprox(agrad, ngrad, atol = 1e-4, rtol = 1e-3)

	    # Check the analytic Hessian
   	    score1 = function (x)
        	gr = zeros(length(x))
        	score(c, x, gr)
        	return gr
    	end
    	nhess = jacobian(central_fdm(12, 1), score1, x)[1]
    	nhess = (nhess + nhess') ./ 2
    	ahess = zeros(size(nhess)...)
    	hessian(c, x, ahess)
    	@test isapprox(ahess, nhess)
	end
end

@testset "Simple clogit fit" begin

    y, X, g = gen_simple_clogit(1000)

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

@testset "Simple cpoisson fit" begin

    y, X, g = gen_simple_cpoisson(1000)

    # Test log-likelihood
    pa = [1.0, 0.0, -1.0]
    m = fit(ConditionalPoissonModel, X, y, g, dofit = false)
    @test isapprox(loglike(m, pa), -5589.031293999247, atol = 1e-4, rtol = 1e-4)

    # Test score
    gr = zeros(3)
    score(m, pa, gr)
    @test isapprox(
        gr,
        [-111.18286798566271, -33.929553430360436, 27.35377124972365],
        atol = 1e-4,
        rtol = 1e-4,
    )

    r = fit(ConditionalPoissonModel, X, y, g)
    @test isapprox(
        coef(r),
        [0.9390890149699354, -0.013509334683457636, -0.9746520727976506],
        atol = 1e-4,
        rtol = 1e-4,
    )

    se = sqrt.(diag(vcov(r)))
    @test isapprox(
        se,
        [0.0231116492306953, 0.02445426198461823, 0.024036656499321647],
        atol = 1e-4,
        rtol = 1e-4,
    )

    score(m, coef(r), gr)
    @test isapprox(gr, Float64[0, 0, 0], atol = 1e-8, rtol = 1e-8)
end
