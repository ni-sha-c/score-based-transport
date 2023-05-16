using PyPlot
include("../1DZeroDirichletTransports.jl")
function KRMap1(x)
     return ((x+1)^3 - 1)/7
end
function KRMap2(x)
    return 4/3 - (2-x)^2/3
end
function plot_target()
    x = rand(100000)
    y = KRMap1.(x)
    fig, ax = subplots()
    ax.hist(y, bins=500)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
end
function test_kam_newton()
	k = 1
	n_gr = 32
	n=30000
	x1, Tx1, x_gr, v_gr1, p_gr1, q_gr1, normv1, vp1, vpp1 = kam_newton1(k,n_gr,n)
	x2, Tx2, x_gr, v_gr2, p_gr2, q_gr2, normv2, vp2, vpp2 = kam_newton2(k,n_gr,n)
    x_source = rand(n)
    x_tar1 = zeros(n)
    x_tar1 .= KRMap1.(x_source)
	x_tar2 = zeros(n)
    x_tar2 .= KRMap2.(x_source)

    fig, ax = subplots()
	ax.plot(x_gr, v_gr1,"P",label="v1")
	ax.plot(x_gr, v_gr2,"o",label="v2")
	ax.xaxis.set_tick_params(labelsize=16)
	ax.yaxis.set_tick_params(labelsize=16)
	ax.legend(fontsize=16)
	ax.grid(true)
	tight_layout()
	savefig("../plots/v.png")	

    fig, ax = subplots()
	ax.xaxis.set_tick_params(labelsize=16)
	ax.yaxis.set_tick_params(labelsize=16)
	#ax.hist(x,bins=75,lw=3.0,histtype="step",density=true,label="source")
	ax.hist(Tx1,bins=75,lw=3.0,histtype="step",density=true,label="KAM target 1")
	ax.hist(Tx2,bins=75,lw=3.0,histtype="step",density=true,label="KAM target 2")
	ax.hist(x_tar1,bins=75,lw=3.0,histtype="step",density=true,label="target 1")
	ax.hist(x_tar2,bins=75,lw=3.0,histtype="step",density=true,label="target 2")
	ax.set_title("After $k iteration(s)",fontsize=16)
	ax.grid(true)
	ax.legend(fontsize=16)
	tight_layout()
	savefig("../plots/hist-k$k.png")	
	fig, ax = subplots()
	ax.plot(x_gr, p_gr1, "P", label="KAM score 1")
	ax.plot(x_gr, q_gr1, "o", label="tar score 1")
    ax.plot(x_gr, p_gr2, "P", label="KAM score 2")
	ax.plot(x_gr, q_gr2, "o", label="tar score 2")
	ax.xaxis.set_tick_params(labelsize=16)
	ax.yaxis.set_tick_params(labelsize=16)
	ax.legend(fontsize=16)
	ax.grid(true)
	ax.set_title("After $k iteration(s)",fontsize=16)
	tight_layout()
	savefig("../plots/scores-k$k.png")	
    
    fig, ax = subplots()
    ax.plot(x_gr, x_gr .+ v_gr1, "o", label="KAM transport 1")
    ax.plot(x_gr, KRMap1.(x_gr), "P", label="KR transport 1")
	ax.xaxis.set_tick_params(labelsize=16)

    ax.plot(x_gr, x_gr .+ v_gr2, "o", label="KAM transport 2")
    ax.plot(x_gr, KRMap2.(x_gr), "P", label="KR transport 2")
	ax.yaxis.set_tick_params(labelsize=16)
	ax.legend(fontsize=16)
	ax.grid(true)
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("T(x)", fontsize=16)
    tight_layout()
    #=
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
    =#
    
end

