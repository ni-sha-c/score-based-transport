from numpy import *
import scipy as sp
m1,m2,s1,s2,w1,w2=-0.5,0.5,0.1,0.1,0.5,0.5

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
def solve_newton_step(p, q, dq, a, b, n):
    dx = (b-a)/(n-1)
    dx_inv = 1/dx
    dx2_inv = dx_inv*dx_inv
    A = zeros((n,n))
    b = p - q
    sii = -2*dx2_inv
    fi = 0.5*dx_inv
    for i in range(1,n-1):
        qi = q[i]
        qf = q[i]*fi
        A[i,i] += sii + dq[i]
        A[i,i+1] += dx2_inv + qf
        A[i,i-1] += dx2_inv - qf
    v = zeros(n)
    v[1:n-1] = linalg.solve(A[1:n-1,1:n-1], b[1:n-1])
    return v

def unimodal_score(x,m,s):
    return -(x-m)/s/s


"""
	Get score = d/dx log p (x), where p is a bimodal probability 
	distribution.
	p(x) =  (w1 e^(-(x-m1)^2/2s1*s1) +  w2 e^(-(x-m2)^2/2s2*s2))/Z
	Inputs:
		x: point to evaluate score at
		m1, m2, s1, s2, w1, w2: parameters of bimodal distribution
	Output:
		s(x) = (d/dx log p)(x)
"""
def bimodal_score(x,m1,m2,s1,s2,w1,w2):
    s1sq_inv, s2sq_inv = 1.0/(s1*s1), 1.0/(s2*s2)
    c = 1/sqrt(2*pi)
    w1p, w2p = w1*c/s1, w2*c/s2
    p_g1 = exp(-(x-m1)^2*s1sq_inv/2)
    p_g2 = exp(-(x-m2)^2*s2sq_inv/2)
    px = w1p*p_g1 + w2p*p_g2
    dpx = -w1p*p_g1*(x-m1)*s1sq_inv-w2p*p_g2*(x-m2)*s2sq_inv
    return dpx/px
"""
	Get probability p(x) for a bimodal probability 
	distribution.
	p(x) =  (w1/(s1 √2pi) e^(-(x-m1)^2/2s1*s1) +  w2/(s2 √2pi) e^(-(x-m2)^2/2s2*s2))
	Inputs:
		x: point of evaluation
		m1, m2, s1, s2, w1, w2: parameters of bimodal distribution
	Output:
		p(x)
"""
def bimodal_prob(x,m1,m2,s1,s2,w1,w2):
    s1sq_inv, s2sq_inv = 1.0/(s1*s1), 1.0/(s2*s2)
    c = 1/sqrt(2*pi)
    w1p, w2p = w1*c/s1, w2*c/s2
    p_g1 = exp(-(x-m1)^2*s1sq_inv/2)
    p_g2 = exp(-(x-m2)^2*s2sq_inv/2)
    px = w1p*p_g1 + w2p*p_g2
    return px
"""
   Sample from a bimodal Gaussian
   Inputs:
   		m1,m2,s1,s2,w1,w2: parameters of the bimodal distribution
		n: number of samples needed
	Output:
		x: n samples from bimodal distribution
"""
def sample_bimodal(m1,m2,s1,s2,w1,w2,n):
    x = zeros(n)
    for i in range(n):
        u = random.rand()
        if u < w1:
            x[i] = m1 + s1*randn()
        else: 
            x[i] = m2 + s2*randn()
    return x
"""
	Get score derivative = d^2/dx^2 log p (x), where p is a bimodal probability 
	distribution.
	Inputs:
		x: point of evaluation
		m1, m2, s1, s2, w1, w2: parameters of bimodal distribution
	Output:
		ds(x) = (d^2/dx^2 log p)(x)
"""
def bimodal_score_derivative(x,m1,m2,s1,s2,w1,w2):
    s1sq_inv, s2sq_inv = 1.0/(s1*s1), 1.0/(s2*s2)
    c = 1/sqrt(2*pi)
    w1p, w2p = w1*c/s1, w2*c/s2
    p1 = exp(-(x-m1)^2*s1sq_inv/2)
    p2 = exp(-(x-m2)^2*s2sq_inv/2)
    a1 = -(x-m1)*s1sq_inv
    a2 = -(x-m2)*s2sq_inv
    dp1 = p1*a1
    dp2 = p2*a2
    da1 = -s1sq_inv
    da2 = -s2sq_inv
    p = w1p*p1 + w2p*p2
    t1 = 1/p*(w1p*dp1 + w2p*dp2)
    t2 = 1/p*(w1p*(dp1*a1 + p1*da1) + w2p*(dp2*a2 + p2*da2))
    return -t1*t1 + t2
"""
	Evaluate the transformed score function G(p,Id+v)
	Inputs:
		p: value of the score at a set of n points
		vp: value of v' at the n points
		vpp: value of v'' at the n points
	Output:
		Gp: value of G(p,Id+v) at the transformed n points.
"""
def H(p,vp,vpp):
    opvp_inv = 1.0/(1.0+vp)
    return p*opvp_inv - vpp*opvp_inv*opvp_inv

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
def newton_update(x_gr, v_gr, p_gr, x, n_gr, n):
	v_int = interp(x, x_gr, v_gr)
	Tx = x + v_int
	dx_inv = 1/(x_gr[1]-x_gr[0])
	dx2_inv = dx_inv*dx_inv
	vp_gr = (v_gr[2:]-v_gr[:n_gr-2])*dx_inv*0.5
	vpp_gr = zeros(n_gr-2)
	Gp_gr = H(p_gr[1:n_gr-1],vp_gr,vpp_gr)

	Tx_gr = x_gr + v_gr
	Tx_gr = Tx_gr[1:n_gr-1]
	order_gr = argsort(Tx_gr)
	Tx_gr, Gp_gr = Tx_gr[order_gr], Gp_gr[order_gr]
	p1_gr = interp(x_gr,Tx_gr,Gp_gr)
	return p1_gr, Tx  
"""
	Main driver function that performs KAM-Newton iteration to construct transport maps
	Inputs:
		m_s, s_s: mean and std of the source distribution
		m1, m2, s1, s2, w1, w2: parameters of the target distribution
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
def kam_newton(m_s,s_s,m1,m2,s1,s2,w1,w2,k,n_gr,n):

	# Set up initial grid
    x = m_s + s_s*random.randn(n)
    Tx = copy(x)
    a, b = min(m1-4*s1,m2-4*s2),max(m1+4*s1,m2+4*s2)
    x_gr = linspace(a,0,n_gr)

	# Set up first iteration
    p_gr = unimodal_score(x_gr, m_s, s_s)
    q_gr = bimodal_score(x_gr,m1,m2,s1,s2,w1,w2)
    dq_gr =  bimodal_score_derivative(x_gr,m1,m2,s1,s2,w1,w2)
    v_gr = zeros(n_gr)	
    
    # Set up some metrics to return
    normv = zeros(k)
    print(sum(x)/n, sum(x*x)/n)
	
    # Run Newton iterations
    for i in range(k):
        v_gr = solve_newton_step(p_gr, q_gr, dq_gr, a, b, n_gr)
        normv[i] = linalg.norm(v_gr)
        p_gr, Tx = newton_update(x_gr, v_gr, p_gr, Tx, n_gr, n)
        print(max(p_gr), min(p_gr), max(Tx), min(Tx))
    return x, Tx, x_gr, v_gr, p_gr, q_gr, normv
