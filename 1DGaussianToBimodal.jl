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
function bimodal_score_derivative(x,m1,m2,σ1,σ2,w1,w2)
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
