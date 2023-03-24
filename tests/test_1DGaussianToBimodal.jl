using Test
using PyPlot
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
function test_kam_newton()
	m_s, σ_s = 0, 0.5
	m1,m2,σ1,σ2,w1,w2=-0.5,0.5,0.1,0.1,0.5,0.5	
	k = 2
	n_gr = 500
	n=10000
	x, Tx, x_gr, v_gr, p_gr, q_gr = kam_newton(m_s,σ_s,m1,m2,σ1,σ2,w1,w2,k,n_gr,n)
	fig, ax = subplots()
	ax.plot(x_gr, v_gr,"P",label="v")
	ax.plot(x, Tx,"o",label="v interpolated")
	ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	ax.legend(fontsize=24)
	fig, ax = subplots()
	ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	ax.hist(x,bins=100,density=true)
	ax.hist(Tx,bins=100,density=true)
	fig, ax = subplots()
	ax.plot(x_gr, p_gr,"P",label="source score")
	ax.plot(x_gr, q_gr,"o",label="tar score")
	ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	ax.legend(fontsize=24)

	return x, Tx, x_gr, v_gr, p_gr, q_gr
end
function test_target_score()
	m_s, σ_s = 0, 0.5
	m1,m2,σ1,σ2,w1,w2=-0.5,0.5,0.1,0.1,0.5,0.5	
	k = 2
	n_gr = 500
	n=10000
	x, Tx, x_gr, v_gr, p_gr, q_gr = kam_newton(m_s,σ_s,m1,m2,σ1,σ2,w1,w2,k,n_gr,n)
	fig, ax = subplots()
	ax.plot(x_gr, v_gr,"P",label="v")
	ax.plot(x, Tx,"o",label="v interpolated")
	ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	ax.legend(fontsize=24)
	fig, ax = subplots()
	ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	ax.hist(x,bins=100,density=true)
	ax.hist(Tx,bins=100,density=true)
	fig, ax = subplots()
	ax.plot(x_gr, p_gr,"P",label="source score")
	ax.plot(x_gr, q_gr,"o",label="tar score")
	ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	ax.legend(fontsize=24)

	return x, Tx, x_gr, v_gr, p_gr, q_gr
end
