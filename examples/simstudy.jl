using CondReg, LinearAlgebra

# Number of simulation replications
nrep = 1000

# Sample size
n = 30

# Correlation between first two covariates
r = 0.5

# Number of times that each covariate's effect is
# significant at the 0.05 level.
sig2 = [0, 0, 0]

for i in 1:nrep

	x = randn(n, 3)
	x[:, 2] = r .* x[:, 1] + sqrt(1 - r^2) .* x[:, 2]
	
	lp = x[:, 1] - x[:, 3]
	ey = 1 ./ (1 .+ exp.(-lp))
	y = [rand() < ey[i] ? 1 : 0 for i in eachindex(ey)]
	g = ones(Int, n)

	m = fit(ConditionalLogitModel, x, y, g)
	se = sqrt.(diag(vcov(m)))
	z = coef(m) ./ se
	sig2 .+= abs.(z) .> 2

end

# Print the proportion of the runs when each covariate is deemed
# to be statistically significant.
println(sig2 ./ nrep)
