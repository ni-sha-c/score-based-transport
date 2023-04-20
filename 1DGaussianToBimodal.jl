using Interpolations
using LinearAlgebra
m1,m2,σ1,σ2,w1,w2=-0.5,0.5,0.1,0.1,0.5,0.5
"""
	Solve for v_n:
	L(q) v_n = (p_n - q)
	where
		L(q) = d^2/dx^2 + q d/dx + dq/dx
	Inputs:
		p_n: intermediate score at the n points
		q: target score at the n points
		dq/dx: dq/dx at the n points
		n, a, b: used to grid [a,b] with n uniform points 
	Output:
		v_n: solution v_n at the n points.
"""
function solve_newton_step(p, q, dq, a, b, n)
	dx = (b-a)/(n-1)
	dx_inv = 1/dx
	dx2_inv = dx_inv*dx_inv
	
	# Av = b
	A = zeros(n,n)
	b = p .- q
	
	# Second-order term
	sii = -2*dx2_inv
	# First-order term
	fi = 0.5*dx_inv

	for i = 2:(n-1)
		qi = q[i]
		qf = q[i]*fi
		# Second-order + first-order + zeroth order
		A[i,i] += sii + dq[i]
		A[i,i+1] += dx2_inv + qf
		A[i,i-1] += dx2_inv - qf
	end

	# Left boundary
	#A[1,1] += -fi*q[1] + dq[1]
	#A[1,3] += q[1]*fi

	# Right boundary
	#A[n,n] += fi*q[n] + dq[n]
	#A[n,n-2] += -fi*q[n]
	v = zeros(n)
	v[2:n-1] = A[2:n-1,2:n-1]\b[2:n-1]
    return v
end
"""
	Get score = d/dx log p (x), where p is a bimodal probability 
	distribution.
	p(x) =  (w1 e^(-(x-m1)^2/2σ1*σ1) +  w2 e^(-(x-m2)^2/2σ2*σ2))/Z
	Inputs:
		x: point to evaluate score at
		m1, m2, σ1, σ2, w1, w2: parameters of bimodal distribution
	Output:
		s(x) = (d/dx log p)(x)
"""
function bimodal_score(x,m1,m2,σ1,σ2,w1,w2)
	σ1sq_inv, σ2sq_inv = 1.0/(σ1*σ1), 1.0/(σ2*σ2)
	c = 1/sqrt(2*π)
	w1p, w2p = w1*c/σ1, w2*c/σ2
	p_g1 = exp(-(x-m1)^2*σ1sq_inv/2)
	p_g2 = exp(-(x-m2)^2*σ2sq_inv/2)
	px = w1p*p_g1 + w2p*p_g2
	dpx = -w1p*p_g1*(x-m1)*σ1sq_inv-w2p*p_g2*(x-m2)*σ2sq_inv
	return dpx/px
end
"""
	Get probability p(x) for a bimodal probability 
	distribution.
	p(x) =  (w1/(σ1 √2π) e^(-(x-m1)^2/2σ1*σ1) +  w2/(σ2 √2π) e^(-(x-m2)^2/2σ2*σ2))
	Inputs:
		x: point of evaluation
		m1, m2, σ1, σ2, w1, w2: parameters of bimodal distribution
	Output:
		p(x)
"""
function bimodal_prob(x,m1,m2,σ1,σ2,w1,w2)
	σ1sq_inv, σ2sq_inv = 1.0/(σ1*σ1), 1.0/(σ2*σ2)
	c = 1/sqrt(2*π)
	w1p, w2p = w1*c/σ1, w2*c/σ2

	p_g1 = exp(-(x-m1)^2*σ1sq_inv/2)
	p_g2 = exp(-(x-m2)^2*σ2sq_inv/2)
	px = w1p*p_g1 + w2p*p_g2
	return px
end
"""
   Sample from a bimodal Gaussian
   Inputs:
   		m1,m2,σ1,σ2,w1,w2: parameters of the bimodal distribution
		n: number of samples needed
	Output:
		x: n samples from bimodal distribution
"""
function sample_bimodal(m1,m2,σ1,σ2,w1,w2,n)
	x = zeros(n)
	for i = 1:n
		u = rand()
		if u < w1
			x[i] = m1 + σ1*randn()
		else 
			x[i] = m2 + σ2*randn()
		end
	end
	return x
end
"""
	Get score derivative = d^2/dx^2 log p (x), where p is a bimodal probability 
	distribution.
	Inputs:
		x: point of evaluation
		m1, m2, σ1, σ2, w1, w2: parameters of bimodal distribution
	Output:
		ds(x) = (d^2/dx^2 log p)(x)
"""
function bimodal_score_derivative(x,m1,m2,σ1,σ2,w1,w2)
	σ1sq_inv, σ2sq_inv = 1.0/(σ1*σ1), 1.0/(σ2*σ2)
	c = 1/sqrt(2*π)
	w1p, w2p = w1*c/σ1, w2*c/σ2

	p1 = exp(-(x-m1)^2*σ1sq_inv/2)
	p2 = exp(-(x-m2)^2*σ2sq_inv/2)
	a1 = -(x-m1)*σ1sq_inv
	a2 = -(x-m2)*σ2sq_inv

	dp1 = p1*a1
	dp2 = p2*a2
	da1 = -σ1sq_inv
	da2 = -σ2sq_inv

	p = w1p*p1 + w2p*p2
	# s = 1/p(w1*dp1 + w2*dp2)
	t1 = 1/p*(w1p*dp1 + w2p*dp2)
	t2 = 1/p*(w1p*(dp1*a1 + p1*da1) + 
			  w2p*(dp2*a2 + p2*da2))
	return -t1*t1 + t2
end
"""
	Evaluate the transformed score function G(p,Id+v)
	Inputs:
		p: value of the score at a set of n points
		vp: value of v' at the n points
		vpp: value of v'' at the n points
	Output:
		Gp: value of G(p,Id+v) at the transformed n points.
"""
function H(p,vp,vpp)
	opvp_inv = 1.0./(1.0.+vp)
	return p.*opvp_inv .- vpp.*opvp_inv.*opvp_inv
end
"""
	Update the score and the transport map using the Newton iterate solution v
	Inputs:
	    x_gr: n_gr grid points
		v_gr: values of v at the n_gr grid points x_gr
		x: n samples
	Outputs:
		x1_gr: new grid points
		q_gr: target scores at x1_gr
		dq_gr: target score derivative at x1_gr
		a, b: boundaries of x1_gr
		p1_gr: transported score at x1_gr
		Tx: transported samples
"""
function newton_update(x_gr, v_gr, p_gr, x, tar_score, dtar_score, n_gr, n)
	v_int = linear_interpolation(x_gr, v_gr, extrapolation_bc=Line())
	Tx = x .+ v_int.(x)
	dx_inv = 1/(x_gr[2]-x_gr[1])
	dx2_inv = dx_inv*dx_inv
	vp_gr = (v_gr[3:n_gr].-v_gr[1:n_gr-2]).*dx_inv.*0.5
	vpp_gr = (-2*v_gr[2:n_gr-1].+v_gr[3:n_gr].+v_gr[1:n_gr-2]).*dx2_inv
	Gp_gr = H(p_gr[2:n_gr-1],vp_gr,vpp_gr)

	a, b = max(minimum(x_gr),minimum(Tx)), min(maximum(x_gr),maximum(Tx))
	@show a, b, tar_score(a), tar_score(b)
	x1_gr = Array(LinRange(a,b,n_gr))
	p1_int = linear_interpolation(x_gr[2:n_gr-1],Gp_gr,extrapolation_bc=Line())
	p1_gr = Array(p1_int.(x1_gr))
	q_gr = Array(tar_score.(x_gr))
	dq_gr = Array(dtar_score.(x1_gr)) 
	return x1_gr, p1_gr, q_gr, dq_gr, a, b, Tx 
end
"""
	Main driver function that performs KAM-Newton iteration to construct transport maps
	Inputs:
		m_s, σ_s: mean and std of the source distribution
		m1, m2, σ1, σ2, w1, w2: parameters of the target distribution
		k: maximum number of iterations of Newton method
		n_gr: number of grid points for ODE solve in Newton iteration
		n: number of target samples needed

	Outputs:
		x: n samples from a source distribution
		Tx: evaluations of the final transport map at x
		x_gr: grid points from the last iteration
		v: values of the final v at x_gr
		p_gr: values of the final transported score at x_gr
		q_gr: values of the target score at x_gr
"""
function kam_newton(m_s,σ_s,m1,m2,σ1,σ2,w1,w2,k,n_gr,n)
	#Set up function definitions
	source_score(x) = bimodal_score(x,m_s,m2,σ_s,σ2,1.0,0.0)
	tar_score(x) = bimodal_score(x,m1,m2,σ1,σ2,w1,w2)
	dtar_score(x) = bimodal_score_derivative(x,m1,m2,σ1,σ2,w1,w2)

	# Set up initial grid
	x = m_s .+ σ_s*randn(n)
	Tx = zeros(n)
	Tx .= x
	a, b = min(m1-3*σ1,m2-3*σ2),max(m1+3*σ1,m2+3*σ2)
	x_gr = Array(LinRange(a,b,n_gr))

	# Set up first iteration
	p_gr = Array(source_score.(x_gr))
	q_gr = Array(tar_score.(x_gr))
	dq_gr = Array(dtar_score.(x_gr))
	v_gr = zeros(n_gr)	
	
	# Set up some metrics to return
	normv = zeros(k)
	x_src = copy(x)
	@show sum(x_src)/n, sum(x_src.*x_src)/n
	# Run Newton iterations
	for i = 1:k
		v_gr .= solve_newton_step(p_gr, q_gr, dq_gr, a, b, n_gr)
		normv[i] = norm(v_gr)
		x1_gr, p1_gr, q_gr1, dq_gr1, a, b, Tx1 = newton_update(x_gr, v_gr, p_gr, x, tar_score, dtar_score, n_gr, n)
		@show maximum(p1_gr), minimum(p1_gr), maximum(q_gr1), minimum(q_gr1)	
		#Update
		p_gr .= p1_gr
		q_gr .= q_gr1
		x_gr .= x1_gr
		dq_gr .= dq_gr1
		x .= Tx
		Tx .= Tx1
	end
	return x_src, Tx, x_gr, v_gr, p_gr, q_gr, normv 
end

