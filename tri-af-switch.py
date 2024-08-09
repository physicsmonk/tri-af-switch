import numpy as np
from scipy import integrate
from scipy.integrate import solve_ivp
#from basic_units import degrees, radians
import matplotlib.pyplot as plt
plt.style.use("publication")
plt.rc('text.latex', preamble=r"\usepackage{amssymb}")
import matplotlib.ticker as ticker
import pandas as pd

def totxt(**kwargs):
    txt = ""
    for key, val in kwargs.items():
        txt += f"{key}={val:g} "
    return txt

omega_E = 27.4 * 2.0 * np.pi
omega_a = 0.001 * 2.0 * np.pi  # 0.003; 0.001 0.0001
ea_order = 6
omega_A = 0.01 * 2.0 * np.pi
alpha = 0.005   # 0.0068 0.005; 0.00443 0.00481

tpi = 5.0
def sc(t, omega_s, tp):
    #if t < tpi:
    #    return 0.0
    #elif t < tpi + tp:
    #    return omega_s
    #else:
    #    return 0.0
    return omega_s * np.logical_and(t > tpi, t < tpi + tp)  # Works for t being both a scalar and a numpy array

def llg(t, y, omega_s, tp):  # y is spherical angles of m1, m2, and m3
    s = np.sin(y)
    c = np.cos(y)
    tmp1a = ( omega_E * (s[0] * (c[2] + c[4]) - c[0] * s[1] * (s[2] * s[3] + s[4] * s[5])
                         - c[0] * c[1] * (s[2] * c[3] + s[4] * c[5])) 
              # + omega_a * s[0] * c[0] * c[1] * c[1] 
              + omega_a * s[0] ** (ea_order - 1) * c[0] * c[1] ** ea_order
              + omega_A * s[0] * c[0] )
    tmp1b = ( omega_E * (c[1] * (s[2] * s[3] + s[4] * s[5]) - s[1] * (s[2] * c[3] + s[4] * c[5])) 
              # + omega_a * s[0] * s[1] * c[1] 
              + omega_a * s[0] ** (ea_order - 1) * s[1] * c[1] ** (ea_order - 1)
              + sc(t, omega_s, tp) * s[0] )
    tmp2a = ( omega_E * (s[2] * (c[4] + c[0]) - c[2] * s[3] * (s[4] * s[5] + s[0] * s[1])
                         - c[2] * c[3] * (s[4] * c[5] + s[0] * c[1])) 
              # + omega_a * s[2] * c[2] * c[3] * c[3] 
              + omega_a * s[2] ** (ea_order - 1) * c[2] * c[3] ** ea_order
              + omega_A * s[2] * c[2] )
    tmp2b = ( omega_E * (c[3] * (s[4] * s[5] + s[0] * s[1]) - s[3] * (s[4] * c[5] + s[0] * c[1])) 
              # + omega_a * s[2] * s[3] * c[3] 
              + omega_a * s[2] ** (ea_order - 1) * s[3] * c[3] ** (ea_order - 1)
              + sc(t, omega_s, tp) * s[2] )
    tmp3a = ( omega_E * (s[4] * (c[0] + c[2]) - c[4] * s[5] * (s[0] * s[1] + s[2] * s[3])
                         - c[4] * c[5] * (s[0] * c[1] + s[2] * c[3])) 
              # + omega_a * s[4] * c[4] * c[5] * c[5] 
              + omega_a * s[4] ** (ea_order - 1) * c[4] * c[5] ** ea_order
              + omega_A * s[4] * c[4] )
    tmp3b = ( omega_E * (c[5] * (s[0] * s[1] + s[2] * s[3]) - s[5] * (s[0] * c[1] + s[2] * c[3])) 
              # + omega_a * s[4] * s[5] * c[5] 
              + omega_a * s[4] ** (ea_order - 1) * s[5] * c[5] ** (ea_order - 1)
              + sc(t, omega_s, tp) * s[4] )
    return [(alpha * tmp1a - tmp1b) / (1.0 + alpha * alpha), 
            -(tmp1a + alpha * tmp1b) / ((1.0 + alpha * alpha) * s[0]),
            (alpha * tmp2a - tmp2b) / (1.0 + alpha * alpha), 
            -(tmp2a + alpha * tmp2b) / ((1.0 + alpha * alpha) * s[2]),
            (alpha * tmp3a - tmp3b) / (1.0 + alpha * alpha), 
            -(tmp3a + alpha * tmp3b) / ((1.0 + alpha * alpha) * s[4])]
# Magnetic loss due to Gilbert damping is 
# int[alpha * dm_dt^2 * dt] = int[alpha * (dtheta_dt^2 + sin(theta)^2 * dphi_dt^2) * dt]
def loss(t, y, omega_s, tp):  
    dy_dt = llg(t, y, omega_s, tp)
    return alpha * integrate.simpson(np.sum(np.square(dy_dt[::2]) + np.square(np.sin(y[::2]) * dy_dt[1::2]), axis=0), t)

def llg_simpl(t, y, omega_s, tp): # y = [mz, phi1]
    return [3.0 / 32.0 * omega_a * np.sin(6.0 * y[1]) - 3.0 * omega_E * alpha * y[0] + 3.0 * sc(t, omega_s, tp),
            -omega_E * y[0]]

def llg_coll(t, y, omega_s, tp): # y = [mz, phi1]
    return [omega_a * np.sin(2.0 * y[1]) - 2.0 * omega_E * alpha * y[0] + 2.0 * sc(t, omega_s, tp),
            -omega_E * y[0]]
def loss_coll(t, y, omega_s, tp):
    dy_dt = llg_coll(t, y, omega_s, tp)
    return alpha * integrate.simpson(0.5 * np.square(dy_dt[0]) + 2.0 * np.square(dy_dt[1]), t)

savefig = False
figseries = 3
omega_s = 0.0005  # 0.0034
tp = 10.0
tf = 200.0
y0 = [np.pi * 0.5, 0.0, np.pi * 0.5, np.pi * 2.0 / 3.0, np.pi * 0.5, -np.pi * 2.0 / 3.0]
y0_simpl = [0.0, 0.0]
sol = solve_ivp(llg, [0.0, tf], y0, args=(omega_s, tp), rtol=1e-6, atol=1e-9, max_step=tp * 0.5)
print(sol.y[1,-1] * 180.0 / np.pi)
fig, ax = plt.subplots(2, 1, figsize=(8.6 / 2.54, 9 / 2.54))
#ax.yaxis.set_units(radians)
ax[0].plot(sol.t, sol.y[1,:] * 180.0 / np.pi, label=r"$\phi_1$")
ax[0].plot(sol.t, sol.y[3,:] * 180.0 / np.pi, label=r"$\phi_2$")
ax[0].plot(sol.t, sol.y[5,:] * 180.0 / np.pi, label=r"$\phi_3$")
ax[0].axvspan(tpi, tpi + tp, alpha=0.3)
ax[0].yaxis.set_major_locator(ticker.MultipleLocator(60))
#ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(10))
ax[0].grid(axis="y", linestyle="--")
#ax[0].axhline(-60.0, linestyle=":", color="gray")
#ax[0].annotate(fr"$\omega_S={omega_s * 1000:g}\,$GHz", (6, 10))
ax[0].set_xlim((0, 40))
ax[0].set_xlabel(r"$t$ [ps]")
ax[0].set_ylabel("Azimuthal angle [Deg]", color="C0")
ax[0].legend(loc="upper right", frameon=True, shadow=True)
ax2 = ax[0].twinx()
ax2.plot(sol.t, (np.cos(sol.y[0,:]) + np.cos(sol.y[2,:]) + np.cos(sol.y[4,:])) * 1e3, color="C3")
#ax2.plot(sol.t, np.cos(sol.y[0,:]) * 1e3)
#ax2.plot(sol.t, np.cos(sol.y[2,:]) * 1e3)
#ax2.plot(sol.t, np.cos(sol.y[4,:]) * 1e3)
ax2.set_ylabel(r"$m_z$ [$10^{-3}$]", color="C3")

num = 50
omega_sl = 0.0
omega_sh = 0.002  # 0.005
tpl = 2.0
tph = 20.0  # 20.0
domega_s = (omega_sh - omega_sl) / (num - 1)
dtp = (tph - tpl) / (num - 1)
#wss = np.linspace(0.0, 0.0035, num)
term_angs = np.empty((num, num))
losses = np.empty_like(term_angs)
for i in range(num):
    for j in range(num):
        omega_s = omega_sl + domega_s * i
        tp = tpl + dtp * j
        sol = solve_ivp(llg, [0.0, tf], y0, args=(omega_s, tp), rtol=1e-4, atol=1e-7, max_step=tp * 0.5)
        term_angs[i,j] = sol.y[1,-1]
        losses[i,j] = loss(sol.t, sol.y, omega_s, tp)
#ax.yaxis.set_units(radians)
#ax.plot(wss * 1e3, term_angs * 180.0 / np.pi)
#ax.axvline(3.4, color="black", linestyle=":")
#ax.annotate(fr"$t_p={tp:g}\,$ps", (0.4, 0.85), xycoords="axes fraction")
#ax.yaxis.set_major_locator(ticker.MultipleLocator(60))
#ax.grid(axis="y", linestyle="--")
#WS, TP = np.meshgrid(np.linspace(wsl, wsh, num), np.linspace(tpl, tph, num))
#term_angs = -WS * TP / alpha
im = ax[1].imshow(-term_angs * 180.0 / np.pi, origin="lower", aspect="auto", cmap="nipy_spectral", alpha=0.8,
                  extent=(tpl - dtp * 0.5, tph + dtp * 0.5, (omega_sl - domega_s * 0.5) * 1e3, (omega_sh + domega_s * 0.5) * 1e3))
#ax.plot(10.0, 3.4, marker="D", markerfacecolor="white", markeredgecolor="black")
ax[1].set_xlabel(r"$t_p$ [ps]")
ax[1].set_ylabel(r"$\omega_S$ [GHz]")
#def cd2w(w):
#    return (w * 1e-9 * 1e12) * 1.602176634e-19 * 3e-7 / 4.19e-8 ** 3
#def w2cd(j):
#    return j * 4.19e-8 ** 3 / (1.602176634e-19 * 3e-7) * (1e9 * 1e-12)
#secax = ax.secondary_xaxis("top", functions=(cd2w, w2cd))
#secax.set_xlabel(r"Current density [$10^6\,\mathrm{A/cm}^2$]")
cb = fig.colorbar(im, ax=ax[1], pad=-0.15)
cb.ax.yaxis.set_major_locator(ticker.MultipleLocator(180))
cb.ax.yaxis.set_minor_locator(ticker.MultipleLocator(60))
cb.ax.set_ylabel("Switched angle [Deg]")
#coll = pd.read_excel("coll-data.xlsx", sheet_name=0, header=[0,1])
#for i, df in coll.T.groupby(level=0):
#    ax[1].plot(df.droplevel(0).T["tp [ps]"], df.droplevel(0).T["ws [GHz]"], color="black")
if savefig:
    fig.savefig("dyn-ang-" + str(figseries) + ".pdf",
                metadata={"Keywords": totxt(omega_E_2pi=omega_E / (2.0 * np.pi), omega_a_2pi=omega_a / (2.0 * np.pi),
                                            AO=ea_order, omega_A_2pi=omega_A / (2.0 * np.pi), GD=alpha,
                                            omega_s=omega_s)})
    
fig, ax = plt.subplots()
im = ax.imshow(losses, origin="lower", aspect="auto", cmap="turbo",
               extent=(tpl - dtp * 0.5, tph + dtp * 0.5, (omega_sl - domega_s * 0.5) * 1e3, (omega_sh + domega_s * 0.5) * 1e3))
ax.set_xlabel(r"$t_p$ [ps]")
ax.set_ylabel(r"$\omega_S$ [GHz]")
cb = fig.colorbar(im, ax=ax)
cb.ax.set_ylabel("Magnetic loss")
if savefig:
    fig.savefig("loss-" + str(figseries) + ".pdf",
                metadata={"Keywords": totxt(omega_E_2pi=omega_E / (2.0 * np.pi), omega_a_2pi=omega_a / (2.0 * np.pi),
                                            AO=ea_order, omega_A_2pi=omega_A / (2.0 * np.pi), GD=alpha)})

sw_angs = np.array([np.pi / 3.0, np.pi])
leg = [r"Triangular $\circlearrowright 60^\circ$", r"Triangular $\circlearrowright 180^\circ$"]
tps = np.linspace(1.0 / 100.0, 1.0 / 2.0, 40)
tps = 1.0 / tps
omega_sc = np.zeros((sw_angs.shape[0], tps.shape[0]))
for i, ang in enumerate(sw_angs):
    for j, tp in enumerate(tps):
        domega_s = 0.0005
        which_side = 0
        while abs(domega_s) > 1e-6:
            omega_sc[i,j] += domega_s
            sol = solve_ivp(llg, [0.0, tf], y0, args=(omega_sc[i,j], tp), rtol=1e-4, atol=1e-7, max_step=tp * 0.5)
            which_side2 = (abs(sol.y[1,-1]) > ang - 0.01)
            if which_side2 != which_side:
                domega_s *= -0.5
            which_side = which_side2
sw_angs_coll = np.array([np.pi])
leg_coll = [r"Collinear $\circlearrowright 180^\circ$"]
omega_sc_coll = np.zeros((sw_angs_coll.shape[0], tps.shape[0]))
for i, ang in enumerate(sw_angs_coll):
    for j, tp in enumerate(tps):
        domega_s = 0.0005
        which_side = 0
        while abs(domega_s) > 1e-6:
            omega_sc_coll[i,j] += domega_s
            sol = solve_ivp(llg_coll, [0.0, tf], y0_simpl, args=(omega_sc_coll[i,j], tp), rtol=1e-4, atol=1e-7, max_step=tp * 0.5)
            which_side2 = (abs(sol.y[1,-1]) > ang - 0.01)
            if which_side2 != which_side:
                domega_s *= -0.5
            which_side = which_side2
fig, ax = plt.subplots()
for i in range(omega_sc.shape[0]):
    ax.plot(tps, omega_sc[i] * 1e3, label=leg[i])
for i in range(omega_sc_coll.shape[0]):
    ax.plot(tps, omega_sc_coll[i] * 1e3, linestyle="--", label=leg_coll[i])
ax.set_xlabel(r"$t_p$ [ps]")
ax.set_ylabel(r"$\omega_S$ [GHz]")
ax.legend()
if savefig:
    fig.savefig("comp-" + str(figseries) + ".pdf",
                metadata={"Keywords": totxt(omega_E_2pi=omega_E / (2.0 * np.pi), omega_a_2pi=omega_a / (2.0 * np.pi),
                                            AO=ea_order, omega_A_2pi=omega_A / (2.0 * np.pi), GD=alpha)})

plt.show()
