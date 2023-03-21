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
