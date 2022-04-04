using CondReg, LinearAlgebra

# Number of simulation replications
nrep = 1000

# Sample size
n = 100

# Correlation between first two covariates
r = 0.5

# The true coefficients
beta = Float64[1, 0, -0.5]

function simstudy()

    # Number of times that each covariate's effect is
    # significant at the 0.05 level.
    sig2 = zeros(length(beta))

    # This will hold the mean of coefficient estimates.
    cf = zeros(length(beta))

    for i = 1:nrep

        x = randn(n, 3)
        x[:, 2] = r .* x[:, 1] + sqrt(1 - r^2) .* x[:, 2]

        lp = x * beta
        ey = 1 ./ (1 .+ exp.(-lp))
        y = [rand() < ey[i] ? 1 : 0 for i in eachindex(ey)]

        # Two groups
        g = zeros(Int, n)
        g[div(n, 2)+1:end] .= 1

        m = fit(ConditionalLogitModel, x, y, g)
        se = sqrt.(diag(vcov(m)))
        z = coef(m) ./ se
        sig2 .+= abs.(z) .> 2

        cf += coef(m)
    end

    sig2 ./= nrep
    cf ./= nrep

    return sig2, cf
end

sig2, cf = simstudy()

println(sig2)
println(cf)
