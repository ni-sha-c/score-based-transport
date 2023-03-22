using Test
include("../1DGaussianToBimodal.jl")
function test_solve_newton_step()
	# v''(x) + cos(πx) v'(x) + π sin(πx)  v(x) = π - π^2 sin(πx)
	# Solution v(x) = sin(πx).
	a = 0
	b = 1
	n = 50
	
	x_gr = Array(LinRange(a,b,n))
	q = cos.(π*x_gr)
	dq = π*sin.(π*x_gr)
	p = π .- π*dq
	v = solve_newton_step(p, q, dq, a, b, n)

	v_true = dq./π
	return v, v_true

end
function test_bimodal_score()
	x = rand()
	ϵ = 1.e-6
	m1, m2, σ1, σ2, w1, w2 = rand(6)
	xp, xm = x + ϵ, x - ϵ
	bup_p, bup_m = bimodal_unnormalized_prob(xp,m1,m2,σ1,σ2,w1,w2),bimodal_unnormalized_prob(xm,m1,m2,σ1,σ2,w1,w2)
	s_fd = (log(bup_p) - log(bup_m))/(2*ϵ)
	s_ana = bimodal_score(x,m1,m2,σ1,σ2,w1,w2)
	@test s_fd ≈ s_ana atol=1.e-8
end
function test_bimodal_score_derivative()
	x = rand()
	ϵ = 1.e-6
	m1, m2, σ1, σ2, w1, w2 = rand(6)
	xp, xm = x + ϵ, x - ϵ
	bs_p, bs_m = bimodal_score(xp,m1,m2,σ1,σ2,w1,w2),bimodal_score(xm,m1,m2,σ1,σ2,w1,w2)
	ds_fd = (bs_p - bs_m)/(2*ϵ)
	ds_ana = bimodal_score_derivative(x,m1,m2,σ1,σ2,w1,w2)
	@test ds_fd ≈ ds_ana atol=1.e-8
end
function test_newton_update()
	x_gr = Array(LinRange(0,1,100))
	v_gr = sin.(π*x_gr)
	x = randn(10000)
	Tx = newton_update(x_gr, v_gr, x)
	return x, Tx
end
