"""Run inside the output directory after running the five chapter demos."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def read(name):
    return pd.read_csv(name + ".csv")

def page(title, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(11, 8), squeeze=False,
                             layout="constrained")
    fig.suptitle(title, fontsize=16)
    return fig, axes.flat

def finish(fig, pdf):
    for ax in fig.axes:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)

with PdfPages("exercise-plots.pdf") as pdf:
    fig, axes = page("17 · Oscillator over one period", 2, 2)
    for idx, n in enumerate([16, 64]):
        a, b = axes[2*idx], axes[2*idx+1]
        t = np.linspace(0, 2*np.pi, 300)
        a.plot(t, np.cos(t), "k--", label="exact")
        b.plot(np.cos(t), -np.sin(t), "k--", label="exact")
        for method in ["explicit", "implicit", "midpoint", "cn"]:
            d = read(f"{method}_spring_{n}")
            a.plot(d.t, d.p, label=method)
            b.plot(d.p, d.v, label=method)
        a.set(title=f"{n} steps", xlabel="time", ylabel="position")
        b.set(xlabel="position", ylabel="velocity", aspect="equal")
    finish(fig, pdf)
    fig, axes = page("17 · Long-time amplitude and convergence", 1, 2)
    for method, d in read("ch17").groupby("method"):
        trace = read(f"{method}_long_128")
        axes[0].semilogy(trace.t/(2*np.pi), np.hypot(trace.p,trace.v), label=method)
        axes[1].loglog(d.h, d.error, ".-", label=method)
    axes[0].set(xlabel="periods", ylabel="amplitude", title="128 steps per period")
    axes[1].set(xlabel="step size", ylabel="final state error")
    finish(fig,pdf)
    fig, axes = page("17 · Capacitor voltage under sinusoidal forcing", 2, 2)
    for axis,(rc,n) in zip(axes,[(1,100),(1,1000),(.0001,100),(.0001,1000)]):
        for method in ["explicit","implicit","cn"]:
            d=read(f"{method}_rc_{rc:.6f}_{n}")
            axis.plot(d.t,d.voltage,label=method)
        axis.plot(d.t,d.exact,"k--",label="exact")
        if rc<1 and n==100: axis.set_yscale("symlog",linthresh=1)
        axis.set(xlabel="time (s)",ylabel="voltage (V)",title=f"RC={rc:g} s, N={n}")
    finish(fig,pdf)
    fig, axes = page("18 · Legendre polynomials and their AD derivatives", 1, 2)
    for n,d in read("ch18").groupby("n"):
        axes[0].plot(d.x,d.P,label=f"P{n}")
        axes[1].plot(d.x,d.derivative,label=f"P{n}′")
    axes[0].set(xlabel="x",ylabel="value");axes[1].set(xlabel="x",ylabel="derivative")
    finish(fig,pdf)
    fig, axes = page("19 · Runge–Kutta convergence and the roundoff floor", 1, 2)
    for name,d in read("ch19").groupby("method"):
        axes[int(name.startswith("radau"))].loglog(d.h,d.error,".-",label=name)
    for a in axes: a.set(xlabel="step size",ylabel="final state error")
    finish(fig,pdf)
    fig, axes = page("20 · Adaptive Arenstorf orbit", 1, 2)
    d=read("orbit")
    axes[0].plot(d.x,d.y);axes[0].set(xlabel="x (Earth–Moon distances)",ylabel="y",aspect="equal")
    axes[1].semilogy(d.t[1:],d.h[1:]);axes[1].set(xlabel="time (model units)",ylabel="accepted step")
    finish(fig,pdf)
    fig,axes=page("20 · Oregonator concentrations, scaled error control",2,1)
    for a,method in zip(axes,["explicit","implicit"]):
        d=read("chem_"+method)
        for c in d.columns[1:]: a.semilogy(d.t,d[c],label=c)
        a.set(xlabel="time",ylabel="concentration",title=method)
    finish(fig,pdf)
    fig,axes=page("21 · Mechanical structures",2,2)
    for a,name in zip(axes,["chain2","chain4","crane","top"]):
        d=read(name);dim=3 if name=="top" else 2
        for j in range(len([c for c in d if c.startswith("x")])//dim):
            a.plot(d[f"x{dim*j}"],d[f"x{dim*j+1}"],label=f"mass {j+1}")
        a.set(title=name,xlabel="x",ylabel="y",aspect="equal")
    finish(fig,pdf)
    fig,axes=page("21 · Pendulum energy over 1,000 periods",2,1)
    d=read("energy");period=d.t.iloc[-1]/1000
    for method in ["explicit","implicit","cn","verlet"]:
        axes[0].plot(d.t/period,d[method],label=method)
    for method in ["cn","verlet"]:
        axes[1].plot(d.t/period,d[method]-d[method][0],label=method)
    axes[0].set(xlabel="periods",ylabel="energy")
    axes[1].set(xlabel="periods",ylabel="energy change")
    finish(fig,pdf)
    fig,axes=page("21 · Generalized-alpha pendulum constraint and multiplier",2,1)
    for name in ["dae_1","dae_08"]:
        d=read(name)
        for a,col in zip(axes,["c0","lambda0"]): a.plot(d.t,d[col],label=name)
    axes[0].set(xlabel="time",ylabel="squared length residual")
    axes[1].set(xlabel="time",ylabel="Lagrange multiplier")
    finish(fig,pdf)
