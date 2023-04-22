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
	bup_p, bup_m = bimodal_prob(xp,m1,m2,σ1,σ2,w1,w2),bimodal_prob(xm,m1,m2,σ1,σ2,w1,w2)
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
function test_kam_newton(k)
	m_s, σ_s = 0, 0.5
	m1,m2,σ1,σ2,w1,w2=-0.5,0.5,0.1,0.1,0.5,0.5	
	#k = 1
	n_gr = 128
	n=30000
	x, Tx, x_gr, v_gr, p_gr, q_gr, normv, vp, vpp = kam_newton(m_s,σ_s,m1,m2,σ1,σ2,w1,w2,k,n_gr,n)
	x_tar = sample_bimodal(m1,m2,σ1,σ2,w1,w2,n)
	fig, ax = subplots()
	ax.plot(x_gr, v_gr,"P",label="final v")
	ax.xaxis.set_tick_params(labelsize=16)
	ax.yaxis.set_tick_params(labelsize=16)
	ax.legend(fontsize=16)
	ax.grid(true)
	fig, ax = subplots()
	ax.xaxis.set_tick_params(labelsize=16)
	ax.yaxis.set_tick_params(labelsize=16)
	ax.hist(x,bins=75,lw=3.0,histtype="step",density=true,label="source")
	ax.hist(Tx,bins=75,lw=3.0,histtype="step",density=true,label="transported source")
	ax.hist(x_tar,bins=75,lw=3.0,histtype="step",density=true,label="target")
	ax.set_title("After $k iteration(s)",fontsize=16)
	ax.grid(true)
	ax.legend(fontsize=16)
	ax.set_xlim([-5,5])
	tight_layout()
	savefig("../plots/hist-k$k.png")	
	fig, ax = subplots()
	ax.plot(x_gr, p_gr, "P", label="transported src score (interp)")
	ax.plot(x_gr, q_gr, "o", label="tar score")
	ax.plot(vp, vpp, "o", label="transported src score uninterp")
	ax.xaxis.set_tick_params(labelsize=16)
	ax.yaxis.set_tick_params(labelsize=16)
	ax.legend(fontsize=16)
	ax.grid(true)
	ax.set_title("After $k iteration(s)",fontsize=16)
	tight_layout()
	savefig("../plots/scores-k$k.png")	

	fig, ax = subplots()
	ax.plot(Array(1:k),normv,"P--",label="||v||")
	ax.xaxis.set_tick_params(labelsize=16)
	ax.yaxis.set_tick_params(labelsize=16)
	ax.legend(fontsize=16)
	ax.set_xlabel("KAM-Newton iteration number",fontsize=16)
	ax.grid(true)
	tight_layout()
	savefig("../plots/normv.png")	

	m_true = sum(x_tar)/n
	v_true = sum(x_tar.*x_tar)/n - m_true*m_true
	m_comp = sum(Tx)/n
	v_comp = sum(Tx.*Tx)/n - m_comp*m_comp

	@show m_true, m_comp
	@show v_true, v_comp
	@show v_gr[1:3]
	@show vp[1], vpp[1]
	
	#return x, Tx, x_gr, v_gr, p_gr, q_gr, normv
end
function test_target_score()
	m_s, σ_s = 0, 0.5
	m1,m2,σ1,σ2,w1,w2=-0.5,0.5,0.1,0.1,0.5,0.5	
	k = 2
	n_gr = 500
	a, b = min(m1-5*σ1,m2-5*σ2),max(m1+5*σ1,m2+5*σ2)
	a, b = -4,4
	x_gr = Array(LinRange(a,b,n_gr))	
	tar_score(x) = bimodal_score(x,m1,m2,σ1,σ2,w1,w2)
    dtar_score(x) = bimodal_score_derivative(x,m1,m2,σ1,σ2,w1,w2)
	q_gr = Array(tar_score.(x_gr))
	fig, ax = subplots()
	ax.plot(x_gr, q_gr,"P",label="target score")
	ax.xaxis.set_tick_params(labelsize=16)
	ax.yaxis.set_tick_params(labelsize=16)
	ax.legend(fontsize=16)
	return q_gr
end
