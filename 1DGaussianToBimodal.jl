using Interpolations
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
function bimodal_score(x,m1=m1,m2=m2,σ1=σ1,σ2=σ2,w1=w1,w2=w2)
	σ1sq_inv, σ2sq_inv = 1.0/(σ1*σ1), 1.0/(σ2*σ2)
	p_g1 = exp(-(x-m1)^2*σ1sq_inv/2)
	p_g2 = exp(-(x-m2)^2*σ2sq_inv/2)
	px = w1*p_g1 + w2*p_g2
	dpx = -w1*p_g1*(x-m1)*σ1sq_inv-w2*p_g2*(x-m2)*σ2sq_inv
	return dpx/px
end
"""
	Get unnormalized probability p(x) for a bimodal probability 
	distribution.
	p(x) =  (w1 e^(-(x-m1)^2/2σ1*σ1) +  w2 e^(-(x-m2)^2/2σ2*σ2))
	Inputs:
		x: point of evaluation
		m1, m2, σ1, σ2, w1, w2: parameters of bimodal distribution
	Output:
		p(x)
"""
function bimodal_unnormalized_prob(x,m1,m2,σ1,σ2,w1,w2)
	σ1sq_inv, σ2sq_inv = 1.0/(σ1*σ1), 1.0/(σ2*σ2)
	p_g1 = exp(-(x-m1)^2*σ1sq_inv/2)
	p_g2 = exp(-(x-m2)^2*σ2sq_inv/2)
	px = w1*p_g1 + w2*p_g2
	return px
end
"""
	Get score derivative = d^2/dx^2 log p (x), where p is a bimodal probability 
	distribution.
	p(x) =  (w1 e^(-(x-m1)^2/2σ1*σ1) +  w2 e^(-(x-m2)^2/2σ2*σ2))/Z
	Inputs:
		x: point of evaluation
		m1, m2, σ1, σ2, w1, w2: parameters of bimodal distribution
	Output:
		ds(x) = (d^2/dx^2 log p)(x)
"""
function bimodal_score_derivative(x,m1=m1,m2=m2,σ1=σ1,σ2=σ2,w1=w1,w2=w2)
	σ1sq_inv, σ2sq_inv = 1.0/(σ1*σ1), 1.0/(σ2*σ2)
	p1 = exp(-(x-m1)^2*σ1sq_inv/2)
	p2 = exp(-(x-m2)^2*σ2sq_inv/2)
	a1 = -(x-m1)*σ1sq_inv
	a2 = -(x-m2)*σ2sq_inv

	dp1 = p1*a1
	dp2 = p2*a2
	da1 = -σ1sq_inv
	da2 = -σ2sq_inv

	p = w1*p1 + w2*p2
	dp = w1*dp1 + w2*dp2
	# s = 1/p(w1*dp1 + w2*dp2)
	t1 = -1/p/p*dp*(w1*dp1 + w2*dp2)
	t2 = 1/p*(w1*(dp1*a1 + p1*da1) + 
			  w2*(dp2*a2 + p2*da2))
	return t1 + t2
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
	opvp_inv = 1.0./(1.+vp)
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
function newton_update(x_gr, v_gr, p_gr, x, n_gr, n)
	v_int = linear_interpolation(x_gr, v_gr, extrapolation_bc=Line())
	Tx = x .+ v_int.(x)
	
	dx_inv = 1/(x_gr[2]-x_gr[1])
	dx2_inv = dx_inv*dx_inv
	vp_gr = (v_gr[3:n_gr].-v_gr[1:n_gr-2]).*dx_inv.*0.5
	vpp_gr = (-2*v_gr[2:n_gr-1].+v_gr[3:n_gr].+v_gr[1:n_gr-2]).*dx2_inv
	Gp_gr = H(p_gr[2:n_gr-1],vp_gr,vpp_gr)

	a, b = minimum(Tx), maximum(Tx)
	x1_gr = Array(LinRange(a,b,n_gr))
	p1_int = linear_interpolation(x_gr[2:n_gr-1],Gp_gr)
	p1_gr = Array(p1_int.(x1_gr))
	q_gr = Array(bimodal_score.(x1_gr))
	dq_gr = Array(bimodal_score_derivative.(x1_gr)) 

	return x1_gr, p1_gr, q_gr, dq_gr, a, b, Tx 
end
"""
	Set up Newton method
	Inputs:
		m_s, σ_s: source mean and std
		m1, m2, σ1, σ2, w1, w2: bimodal (target) distribution parameters
	Outputs:
		src_score: function that evaluates the source score
		target_score: function that evaluates the target score
		dtarget_score: function that evaluates the target score derivative
"""
function setup_newton_funs(m_s,σ_s,m1,m2,σ1,σ2,w1,w2)
	src_score(x) = 

end
"""
	Set up first iteration of Newton method
	Inputs:
		src_score: a function that evaluates the source score
		target_score: a function that evaluates the target score
		dtarget_score: a function that evaluates the target score derivative
		a, b: boundaries of the grid points
		n: number of grid points
	Outputs:
		x: grid points
		p: source score evaluated at x
		q: target score evaluated at x
		dq: target score derivative evaluated at x
"""
function setup_first_iteration(src_score,tar_score,dtar_score,a,b,n)
	x_gr = Array(LinRange(a,b,n))
	p_gr = Array(src_score.(x_gr))
	q_gr = Array(tar_score.(x_gr))
	dq_gr = Array(dtar_score.(x_gr))
	return x_gr, p_gr, q_gr, dq_gr
end
"""
	Main driver function that performs KAM-Newton iteration to construct transport maps
	Inputs:
		x: n samples from a source distribution
		q: target score function
		dq: target score derivative function
		k: maximum number of iterations of Newton method
		n_gr: number of grid points for ODE solve in Newton iteration

	Outputs:
		Tx: evaluations of the final transport map at x
		x_gr: grid points from the last iteration
		v: values of the final v at x_gr
		p_gr: values of the final transported score at x_gr
		q_gr: values of the target score at x_gr
"""
function kam_newton(x,tar_score,dtar_score,k,n_gr)
	a, b = minimum(x), maximum(x)

function solve_newton_step(p, q, dq, a, b, n)
	p, q, dq  = setup_newton(tar_score,a,b,n) 	


end

