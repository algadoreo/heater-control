import os, sys, json
import pygetdata as gd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = 16, 9
plt.rcParams["figure.dpi"] = 70

# Constants
QMIN, QMAX = 0, 100

###########################################################################
def HeaterIVP(t, T, params_circ, params_PID = None, Q = 0):
    a, b, c, d = params_circ[0], params_circ[1], params_circ[2], params_circ[3]
    x1, x2 = T[0], T[1]

    dx1 = x2
    dx2 = a*b*c*(Q - d*(x1-params_circ[4])) - (a*b + b*c + c*d)*x2
    
    return [dx1, dx2]

def updateSimParams(t, t_on, sim_params):
    sp_copy = np.copy(sim_params)
    if t < t_on:
        sp_copy[4] = 0.
    return sp_copy

###########################################################################
# Load data
if len(sys.argv) < 2:
    print("Usage: python sim_from_dirfile.py <params_file>")
    sys.exit()

with open(sys.argv[1]) as handle:
    pc = json.loads(handle.read())

TSTEP = pc['samp_rate']
TSTR = pc['time_start']
TEND = pc['time_end']
ivec = np.arange(TEND+1-TSTR)
tvec = ivec * TSTEP
t_span = [ivec[0], ivec[-1]]

ROOTDIR = "/data/mole/"
FILESET = pc['fileset']
TDATASET = pc['t_dataset']
HDATASET = pc['h_dataset']

df = gd.dirfile(os.path.join(ROOTDIR, FILESET))
Tdf = df.getdata(TDATASET)[TSTR:TEND+1]
Qdf = np.maximum(QMIN, np.minimum(QMAX, df.getdata(HDATASET)))[TSTR:TEND+1]
i_Qon = np.where(Qdf>0.5)[0][0]
t_Qon = i_Qon * TSTEP

###########################################################################
# From circuit parameters, simulate timestream
circ_params = np.array([pc['Ch'], pc['Rh'], pc['Ct'], pc['Rt']], dtype=float)
Tb = float(pc['Tb'])
sim_params = np.append(1./circ_params, Tb)

T0 = [0, 0]     # ODE initial conditions ([T0, Tdot0])
soln = solve_ivp(lambda t, T: HeaterIVP(t, T, updateSimParams(t, t_Qon, sim_params), Q=np.interp(t, tvec, Qdf)),
    t_span, T0, t_eval=tvec, max_step=TSTEP)

###########################################################################
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2, 1]})
fig.subplots_adjust(hspace=0)
ax1.plot(ivec+TSTR, Qdf, 'C3', label=HDATASET)
ax1.set_ylabel("Power [%]")
ax1.tick_params(right=True)
ax1.legend()
ax2.plot(ivec+TSTR, Tdf, label=TDATASET)
ax2.plot(ivec+TSTR, soln.y[0]+pc['T0'], '--', label="Best fit")
ax2.set_ylabel(u"Temperature [\u2103]")
ax2.tick_params(right=True, top=True)
ax2.legend()
ax3.axhline(0, color='C7', linestyle='--')
ax3.plot(ivec+TSTR, soln.y[0]+pc['T0']-Tdf, 'C8', label=r"Best fit $-$ " + TDATASET)
ax3.set_xlabel("Index")
ax3.set_ylabel(r"$\Delta T$ " + u"[\u2103]")
plt.legend()
plt.show()

