from matplotlib.pyplot import *
import sys
import os
sys.path.append(os.path.abspath("/home/nisha/Research/faculty/code/score-based-transport/bimodal"))
from oneD import *

def KRMap1(x):
    return ((x+1)**3 - 1)/7
def KRMap2(x):
    return 4/3 - (2-x)**2/3
def plot_target():
    x = random.rand(100000)
    y = KRMap1(x)
    fig, ax = subplots()
    ax.hist(y, bins=500)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)

def test_kam_newton():
    k = 5
    n_gr = 128
    n=30000
    x1 = rand(n)
    x2 = rand(n)
    
    Tx1, x_gr, v_gr1, p_gr1, q_gr1, normv1 = kam_newton(k,n_gr,n,q1,dq1)
	Tx2, x_gr, v_gr2, p_gr2, q_gr2, normv2 = kam_newton(k,n_gr,n,q2,dq2)
    x_source = rand(n)
    x_tar1 = zeros(n)
    x_tar1 .= KRMap1.(x_source)
	x_tar2 = zeros(n)
    x_tar2 .= KRMap2.(x_source)

    fig, ax = subplots()
	ax.set_xlabel("x", fontsize=24)
    ax.plot(x_gr, v_gr1,"P",label="v1")
	ax.plot(x_gr, v_gr2,"o",label="v2")
	ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	ax.legend(fontsize=24,framealpha=0.1)
	ax.grid(true)
	tight_layout()
    ax.yaxis.offsetText.set_fontsize(24)
    savefig("../plots/v$k.png")	

    fig, ax = subplots()
	ax.set_xlabel("x", fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	#ax.hist(x,bins=75,lw=3.0,histtype="step",density=true,label="source")
	#ax.hist(Tx1,bins=75,lw=3.0,histtype="step",density=true,label=L"KAM\: \nu 1")
	ax.hist(Tx2,bins=75,lw=3.0,histtype="step",density=true,label=L"KAM\: \nu 2")
	ax.hist(x_tar1,bins=75,lw=3.0,histtype="step",density=true,label=L"\nu 1")
	#ax.hist(x_tar2,bins=75,lw=3.0,histtype="step",density=true,label=L"\nu 2")
	ax.set_title("After $k iteration(s)",fontsize=24)
	ax.grid(true)
	ax.legend(fontsize=20,framealpha=0.1)
	tight_layout()
	#savefig("../plots/hist-k$k.png")	
	
    fig, ax = subplots()
    ax.set_xlabel("x", fontsize=24)
	#ax.plot(x_gr, p_gr1, "v", ms=6.0, label="KAM score 1")
	ax.plot(x_gr, q_gr1, "o", ms=3.0, label="tar score 1")
    ax.plot(x_gr, p_gr2, "v", ms=6.0, label="KAM score 2")
	#ax.plot(x_gr, q_gr2, "o", ms=3.0, label="tar score 2")
	ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	ax.legend(fontsize=20,markerscale=3,framealpha=0.1)
	ax.grid(true)
	ax.set_title("After $k iteration(s)",fontsize=24)
	tight_layout()
	#savefig("../plots/scores-k$k.png")	
    
    fig, ax = subplots()
    #ax.plot(x1, Tx1, "o", label="KAM T1")
    ax.plot(x1, KRMap1.(x1), "v",ms=3, label="KR T1")
	ax.xaxis.set_tick_params(labelsize=24)

    ax.plot(x2, Tx2, "o", label="KAM T2")
    #ax.plot(x2, KRMap2.(x2), "v", ms=3,label="KR T2")
	ax.yaxis.set_tick_params(labelsize=24)
	ax.legend(fontsize=20,markerscale=3,framealpha=0.1)
    ax.grid(true)
    ax.set_xlabel("x", fontsize=24)
    ax.set_ylabel("T(x)", fontsize=24)
    ax.set_title("After $k iteration(s)",fontsize=24)
    tight_layout()
    #savefig("../plots/transport-k$k.png")	
    
	fig, ax = subplots()
	ax.plot(Array(1:k),normv1,"P--",label="||v1||")
	ax.plot(Array(1:k),normv2,"o--",label="||v2||")
	ax.xaxis.set_tick_params(labelsize=24)
	ax.yaxis.set_tick_params(labelsize=24)
	ax.legend(fontsize=24)
	ax.set_xlabel("KAM-Newton iteration number",fontsize=24)
	ax.grid(true)
	tight_layout()
	savefig("../plots/normv$k.png")	
    
	    
end
"""
