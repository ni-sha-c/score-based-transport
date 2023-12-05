from numpy import *
import scipy.sparse as scsp
import scipy.interpolate as spin
m1,m2,s1,s2,w1,w2=-0.5,0.5,0.1,0.1,0.5,0.5
def q1(x):
    return -14/3/(7*x+1)
def dq1(x):
    return 98/3/(7*x+1)/(7*x+1)
def q2(x):
    return 3/2/(4-3*x) 
def dq2(x):
    return 9/2/(4-3*x)**2
def uni_sc(x):
    return 0.0*x
def gaussian_score(x,m=-4,s=1):
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
def bimodal_score(x,m1=-2,m2=2,s1=1,s2=1,w1=1/3,w2=2/3):
    s1sq_inv, s2sq_inv = 1.0/(s1*s1), 1.0/(s2*s2)
    c = 1/sqrt(2*pi)
    w1p, w2p = w1*c/s1, w2*c/s2
    p_g1 = exp(-(x-m1)**2*s1sq_inv/2)
    p_g2 = exp(-(x-m2)**2*s2sq_inv/2)
    px = w1p*p_g1 + w2p*p_g2
    dpx = -w1p*p_g1*(x-m1)*s1sq_inv-w2p*p_g2*(x-m2)*s2sq_inv
    return dpx/px
"""
	Get score derivative = d^2/dx^2 log p (x), where p is a bimodal probability 
	distribution.
	Inputs:
		x: point of evaluation
		m1, m2, s1, s2, w1, w2: parameters of bimodal distribution
	Output:
		ds(x) = (d^2/dx^2 log p)(x)
"""
def bimodal_score_derivative(x,m1=-2,m2=2,s1=1,s2=1,w1=1/3,w2=2/3):
    s1sq_inv, s2sq_inv = 1.0/(s1*s1), 1.0/(s2*s2)
    c = 1/sqrt(2*pi)
    w1p, w2p = w1*c/s1, w2*c/s2
    p1 = exp(-(x-m1)**2*s1sq_inv/2)
    p2 = exp(-(x-m2)**2*s2sq_inv/2)
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
	Get probability p(x) for a bimodal probability 
	distribution.
	p(x) =  (w1/(s1 √2pi) e^(-(x-m1)^2/2s1*s1) +  w2/(s2 √2pi) e^(-(x-m2)^2/2s2*s2))
	Inputs:
		x: point of evaluation
		m1, m2, s1, s2, w1, w2: parameters of bimodal distribution
	Output:
		p(x)
"""
def bimodal_prob(x,m1=-2,m2=2,s1=1,s2=1,w1=1/3,w2=2/3):
    s1sq_inv, s2sq_inv = 1.0/(s1*s1), 1.0/(s2*s2)
    c = 1/sqrt(2*pi)
    w1p, w2p = w1*c/s1, w2*c/s2
    p_g1 = exp(-(x-m1)**2*s1sq_inv/2)
    p_g2 = exp(-(x-m2)**2*s2sq_inv/2)
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
def sample_bimodal(n,m1=-2,m2=2,s1=1,s2=1,w1=1/3,w2=2/3):
    x = zeros(n)
    for i in range(n):
        u = random.rand()
        if u < w1:
            x[i] = m1 + s1*random.randn()
        else: 
            x[i] = m2 + s2*random.randn()
    return x

"""
This function gives grid locations based on following assumptions:
    1. There are 2 boundary layers and 1 interior layer centered at argmax dq.
    2. Width of interior layer is 3: set manually for weighted sums of N(-/+2,1) 
    3. Total number of points is n
    4. Density of points in layer (interior/boundary) is 10x that in nonlayer
"""
def adapt_grid(a, b, dq, dx, n):
    dens_fac = 4
    int_wid, bl_wid, int_cent = 3.0, 2.0, argmax(dq)*dx + a
    nl_wid = b - a - 2*bl_wid - int_wid
    nl_dens = n/(dens_fac*(b-a) - (dens_fac-1)*nl_wid)
    int_dens, bl_dens = dens_fac*nl_dens, dens_fac*nl_dens
    

    x = zeros(n)
    # left boundary layer
    lbl_beg, lbl_end, n_bl  = a, a+bl_wid, int(bl_dens*bl_wid)
    x[:n_bl] = array(linspace(lbl_beg, lbl_end, n_bl))
    # nonlayer
    nl1_end = int_cent - int_wid/2
    n_nl1 = int((nl1_end-lbl_end)*nl_dens) 
    x[n_bl:n_nl1+n_bl] = array(linspace(lbl_end, nl1_end, n_nl1+1))[1:]
    # interior layer
    int_end, n_int = int_cent + int_wid/2, int(int_dens*int_wid)
    x[n_bl+n_nl1:n_bl+n_nl1+n_int] = array(linspace(nl1_end, int_end, n_int+1))[1:]
    # nonlayer
    nl2_end = b-bl_wid
    n_nl2 = int(nl_dens*(nl2_end-int_end))
    nl_ind = n_bl+n_nl1+n_int
    x[nl_ind:nl_ind+n_nl2] = array(linspace(int_end,nl2_end,n_nl2+1))[1:]
    # right boundary layer
    rbl_ind = nl_ind + n_nl2
    x[rbl_ind:n] = array(linspace(nl2_end, b, n-rbl_ind+1))[1:]
    return x

def fd_coeff_2(x, a, b):
    bx, xa, ba = b-x,x-a,b-a
    den = xa*bx*ba
    return array([2*bx/den, -2*ba/den, 2*xa/den])
def fd_coeff_1(x, a, b):
    ba = b-a
    return array([-1/ba, 0, 1/ba])


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
def help_solver(x, q, dq, n):
    dqs,qs = dq[1:-1],q[1:-1]
    A = zeros((n,n))
    for i in range(1,n-1):
        A[i,i-1:i+2] = fd_coeff_2(x[i],x[i-1],x[i+1]) + \
                        fd_coeff_1(x[i],x[i-1],x[i+1])*q[i]
        A[i,i] += dq[i]
    A = scsp.csr_matrix(A[1:-1,1:-1]) 
    return A

def solve_newton_step(p, q, A, n):
    qs, ps = q[1:-1], p[1:-1]
    b = ps - qs
    v = zeros(n)
    v[1:-1] = scsp.linalg.spsolve(A, b)
    return v
def solve_newton_step_regularized(p, q, A, h, n):
    qs, ps = q[1:-1], p[1:-1]
    b = ps - qs
    v = zeros(n)
    AT = A.T
    Ahat = dot(AT,A)+h*h*scsp.eye(n-2)
    bhat = dot(AT.todense(),b).T
    v[1:-1] = scsp.linalg.spsolve(Ahat, bhat)
    return v

def get_derivs(v,x,n):
    dv, d2v = zeros(n),zeros(n)
    for i in range(1,n-1):
        dv[i] = dot(fd_coeff_1(x[i],x[i-1],x[i+1]),v[i-1:i+2])
        d2v[i] = dot(fd_coeff_2(x[i],x[i-1],x[i+1]),v[i-1:i+2])
    return dv[1:-1],d2v[1:-1]

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
    vp_gr, vpp_gr = get_derivs(v_gr, x_gr, n_gr)
    Gp_gr = H(p_gr[1:-1],vp_gr,vpp_gr)
    Tx_gr = x_gr + v_gr
    Tx_gr = Tx_gr[1:-1]
    order_gr = argsort(Tx_gr)
    Tx_gr, Gp_gr = Tx_gr[order_gr], Gp_gr[order_gr]
    p1_gr_fn = spin.interp1d(Tx_gr,Gp_gr,kind="linear",fill_value="extrapolate")
    v_int = spin.interp1d(x_gr,v_gr,kind="linear",fill_value="extrapolate")
    return p1_gr_fn(x_gr), x+v_int(x)
"""
	Main driver function that performs KAM-Newton iteration to construct transport maps
	Inputs:
        x: source samples
        k: maximum number of iterations of Newton method
		n_gr: number of grid points for ODE solve in Newton iteration
		n: number of target samples needed (=dim(x))

	Outputs:
		Tx: evaluations of the final transport map at x
		x_gr: grid points from the last iteration
		v: values of the final v at x_gr
		p_gr: values of the final transported score at x_gr
		q_gr: values of the target score at x_gr
        normv: norms of v during KAM-Newton iterations
"""
def kam_newton(x,a,b,k,n_gr,n,tar_sc,dtar_sc,src_sc):
    Tx = copy(x)
    x_gr = linspace(a,b,n_gr)
    dq_gr = dtar_sc(x_gr)
    dx = x_gr[1] - x_gr[0]
    x_gr = adapt_grid(a,b,dq_gr,dx,n_gr) 
    # Set up first iteration
    p_gr = src_sc(x_gr)
    q_gr = tar_sc(x_gr)
    dq_gr = dtar_sc(x_gr)
    normv = zeros(k)
    A = help_solver(x_gr, q_gr, dq_gr, n_gr) 
    
    #Run Newton iterations
    for i in range(k):
        v_gr = solve_newton_step_regularized(p_gr, q_gr, A, 0.08, n_gr)
        normv[i] = linalg.norm(v_gr)
        p_gr, Tx = newton_update(x_gr, v_gr, p_gr, Tx, n_gr, n)
        #print(max(q_gr), min(q_gr), max(p_gr), min(p_gr), max(Tx), min(Tx))
        print("||v_%d|| is %f" % (i,linalg.norm(v_gr)))
    return Tx, x_gr, v_gr, p_gr, q_gr, normv
    
