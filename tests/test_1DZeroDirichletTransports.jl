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
	k = 5
	n_gr = 128
	n=30000
	x1, Tx1, x_gr, v_gr1, p_gr1, q_gr1, normv1, vp1, vpp1 = kam_newton(k,n_gr,n,q1,dq1)
	x2, Tx2, x_gr, v_gr2, p_gr2, q_gr2, normv2, vp2, vpp2 = kam_newton(k,n_gr,n,q2,dq2)
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
    ax.plot(x1, Tx1, "o", label="KAM transport 1")
    ax.plot(x1, KRMap1.(x1), "P",ms=0.5, label="KR transport 1")
	ax.xaxis.set_tick_params(labelsize=16)

    ax.plot(x2, Tx2, "o", label="KAM transport 2")
    ax.plot(x2, KRMap2.(x2), "P", ms=0.5,label="KR transport 2")
	ax.yaxis.set_tick_params(labelsize=16)
	ax.legend(fontsize=16)
	
    ax.grid(true)
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("T(x)", fontsize=16)
    tight_layout()
   
	fig, ax = subplots()
	ax.plot(Array(1:k),normv1,"P--",label="||v1||")
	ax.plot(Array(1:k),normv2,"o--",label="||v2||")
	ax.xaxis.set_tick_params(labelsize=16)
	ax.yaxis.set_tick_params(labelsize=16)
	ax.legend(fontsize=16)
	ax.set_xlabel("KAM-Newton iteration number",fontsize=16)
	ax.grid(true)
	tight_layout()
	savefig("../plots/normv.png")	
    
	    
end

