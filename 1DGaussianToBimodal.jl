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
