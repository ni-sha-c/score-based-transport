using Interpolations
using LinearAlgebra
function q1(x)
    return -14/3/(7*x+1) 
end
function dq1(x)
    return 98/3/(7*x+1)/(7*x+1)  
end
#function q2(x)
#    return 3/2/(4-3*x) 
#end
#function dq2(x)
#    return 9/2/(4-3*x)^2
#end
function q2(x)
    return -14/3/(7*x+1) + 0.5*rand()
end
function dq2(x)
    return 98/3/(7*x+1)/(7*x+1) + rand() 
end


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
function newton_update(x_gr, v_gr, p_gr, x, n_gr, n)
	v_int = linear_interpolation(x_gr, v_gr, extrapolation_bc=Line())
	Tx = x .+ v_int.(x)
	dx_inv = 1/(x_gr[2]-x_gr[1])
	dx2_inv = dx_inv*dx_inv
	vp_gr = (v_gr[3:n_gr].-v_gr[1:n_gr-2]).*dx_inv.*0.5
	vpp_gr = (-2*v_gr[2:n_gr-1].+v_gr[3:n_gr].+v_gr[1:n_gr-2]).*dx2_inv
	#vp_gr .= vp_gr[round(Int64, n_gr/2)]
	#vpp_gr = zeros(n_gr-2)
	Gp_gr = H(p_gr[2:n_gr-1],vp_gr,vpp_gr)
        @show maximum(Gp_gr), minimum(Gp_gr)
	Tx_gr = x_gr .+ v_gr
	Tx_gr = Tx_gr[2:n_gr-1]
	order_gr = sortperm(Tx_gr)
	Tx_gr, Gp_gr = Tx_gr[order_gr], Gp_gr[order_gr]
	p1_int = linear_interpolation(Tx_gr,Gp_gr,extrapolation_bc=Line())
	p1_gr = Array(p1_int.(x_gr))
	return p1_gr, Tx, Tx_gr, Gp_gr  
end
"""
	Main driver function that performs KAM-Newton iteration to construct transport maps
	Inputs:
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
function kam_newton(k,n_gr,n,tar_score,dtar_score)
	#Set up function definitions
	source_score(x) = 0.0

	# Set up initial grid
	x = rand(n)
	Tx = zeros(n)
	Tx .= x
	a, b = 0,1
	x_gr = Array(LinRange(a,b,n_gr))

	# Set up first iteration
	p_gr = Array(source_score.(x_gr))
	q_gr = Array(tar_score.(x_gr))
	dq_gr = Array(dtar_score.(x_gr))
	v_gr = zeros(n_gr)	
	vp = zeros(n_gr-2)	
	vpp = zeros(n_gr-2)	
	# Set up some metrics to return
	normv = zeros(k)
	@show sum(x)/n, sum(x.*x)/n

    # Run Newton iterations
	for i = 1:k
		v_gr .= solve_newton_step(p_gr, q_gr, dq_gr, a, b, n_gr)
		normv[i] = norm(v_gr)
		p1_gr, Tx1, vp1, vpp1 = newton_update(x_gr, v_gr, p_gr, Tx, n_gr, n)
		vp .= vp1
		vpp .= vpp1
		
		@show maximum(p1_gr), minimum(p1_gr), maximum(Tx1), minimum(Tx1)	
		#Update
		p_gr .= p1_gr
		Tx .= Tx1
	end
   
	return x, Tx, x_gr, v_gr, p_gr, q_gr, normv, vp, vpp 
end

