using CondReg, Test, Random, LinearAlgebra

function gen_simple(n::Int)

    Random.seed!(123)

    X = randn(n, 3)

    g = zeros(Int, n)
    for i = 1:n
        g[i] = div(i - 1, 20) + 1
    end

    lp = X[:, 1] - X[:, 3]
    mn = 1 ./ (1 .+ exp.(-lp))
    y = [rand() < mn[i] ? 1 : 0 for i = 1:n]

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
	@test isapprox(loglike(m, pa), -470.74749, atol=1e-4, rtol=1e-4)

	# Test score
    gr = zeros(3)
    score(m, pa, gr)
    @test isapprox(gr, [-25.91228169, -13.18333921, -0.35274937], atol=1e-4, rtol=1e-4)

    r = fit(ConditionalLogitModel, X, y, g)
	@test isapprox(coef(r), [0.814469, -0.0731922, -0.95152], atol=1e-4, rtol=1e-4)
	se = sqrt.(diag(vcov(r)))
	@test isapprox(se, [0.0816031, 0.0733603, 0.090174], atol=1e-4, rtol=1e-4)
end
