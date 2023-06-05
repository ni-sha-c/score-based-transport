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
    k,n_gr,n = 2,128,30000
    x1,x2 = random.rand(n),random.rand(n)
    Tx1, x_gr1, v_gr1, p_gr1, q_gr1, normv1 = kam_newton(x1,k,n_gr,n,q1,dq1,uni_sc)
    Tx2, x_gr2, v_gr2, p_gr2, q_gr2, normv2 = kam_newton(x2,k,n_gr,n,q2,dq2,uni_sc)
    x_source = random.rand(n)
    x_tar1 = KRMap1(x_source)
    x_tar2 = KRMap2(x_source)

    fig, ax = subplots()
    ax.set_xlabel("x", fontsize=24)
    ax.plot(x_gr1, v_gr1, "P", label="v1")
    ax.plot(x_gr2, v_gr2, "o", label="v2")
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24,framealpha=0.1)
    ax.set_title("After {} iteration(s)".format(k),fontsize=24)
    ax.grid(True)
    tight_layout()
    ax.yaxis.offsetText.set_fontsize(24)
    savefig("../plots/v$k-py.png")

    fig, ax = subplots()
    ax.set_xlabel("x", fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    #ax.hist(x,bins=75,lw=3.0,histtype="step",density=True,label="source")
    ax.hist(Tx1,bins=75,lw=3.0,histtype="step",density=True,label=R"KAM $\nu 1$")
    ax.hist(Tx2,bins=75,lw=3.0,histtype="step",density=True,label=R"KAM $\nu 2$")
    ax.hist(x_tar1,bins=75,lw=3.0,histtype="step",density=True,label=R"$\nu 1$")
    ax.hist(x_tar2,bins=75,lw=3.0,histtype="step",density=True,label=R"$\nu 2$")
    ax.set_title("After $k iteration(s)",fontsize=24)
    ax.grid(True)
    ax.legend(fontsize=20,framealpha=0.1)
    tight_layout()
    savefig("../plots/hist-k$k-py.png")	
	
    fig, ax = subplots()
    ax.set_xlabel("x", fontsize=24)
    ax.plot(x_gr1, p_gr1, "v", ms=6.0, label="KAM score 1")
    ax.plot(x_gr2, q_gr1, "o", ms=3.0, label="tar score 1")
    ax.plot(x_gr2, p_gr2, "v", ms=6.0, label="KAM score 2")
    ax.plot(x_gr2, q_gr2, "o", ms=3.0, label="tar score 2")
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=20,markerscale=3,framealpha=0.1)
    ax.grid(True)
    ax.set_title("After $k iteration(s)",fontsize=24)
    tight_layout()
    savefig("../plots/scores-k{}-py.png".format(k))	
    
    fig, ax = subplots()
    ax.plot(x1, Tx1, "o", label="KAM T1")
    ax.plot(x1, KRMap1(x1), "v",ms=3, label="KR T1")
    ax.xaxis.set_tick_params(labelsize=24)

    ax.plot(x2, Tx2, "o", label="KAM T2")
    ax.plot(x2, KRMap2(x2), "v", ms=3,label="KR T2")
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=20,markerscale=3,framealpha=0.1)
    ax.grid(True)
    ax.set_xlabel("x", fontsize=24)
    ax.set_ylabel("T(x)", fontsize=24)
    ax.set_title("After {} iteration(s)".format(k),fontsize=24)
    tight_layout()
    savefig("../plots/transport-k{}-py.png".format(k))
    
    fig, ax = subplots()
    ax.plot(range(1,k+1),normv1,"P--",label="||v1||")
    ax.plot(range(1,k+1),normv2,"o--",label="||v2||")
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(fontsize=24)
    ax.set_xlabel("KAM-Newton iteration number",fontsize=24)
    ax.grid(True)
    tight_layout()
    savefig("../plots/normv-k{}-py.png".format(k))	
    
	    
