# -*- coding: utf-8 -*-

import os, sys, json
import pygetdata as gd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from collections import OrderedDict
from subroutines import initvals, heater

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = 16, 9
plt.rcParams["figure.dpi"] = 70

###########################################################################
# Initialize objects
p = initvals()
p.loadData()
h = heater(p)

# Fit (thermal) circuit parameters
print("Best fit circuit parameters:")
lsq = least_squares(h.fit_params, p.iparams_circ, bounds=(1.e-5, 1.e5), args=(p.tvec, p.Tdat-p.Tdat[p.ton_idx]), max_nfev=1e5)

bestfit = dict(zip(["Ch", "Rh", "Ct", "Rt"], 1./lsq.x[:4]))
bestfit["Tb"] = lsq.x[4]
for k in ["Ch", "Rh", "Ct", "Rt", "Tb"]:
    print("{}: {}".format(k, bestfit[k]))

# Sanity check: Plot the temperature based on best fit parameters
soln_on = solve_ivp(lambda t, T: h.HeaterIVP(t, T, lsq.x, Q = p.QMAX), [p.ton, p.toff], p.T0, t_eval=p.tvec[p.sli_on])
soln_off = solve_ivp(lambda t, T: h.HeaterIVP(t, T, lsq.x, Q = 0), [p.toff, p.tvec[-1]], [soln_on.y[0][-1], soln_on.y[1][-1]], t_eval=p.tvec[p.sli_off])
soln_all = np.concatenate((soln_on.y[0], soln_off.y[0][1:]))

plt.figure()
plt.subplot(3, 1, (1,2))
plt.axhline(p.Tdat[p.ton_idx]+273.15, color='C7', linestyle='--')
plt.axvline(p.ton, color='C7', linestyle='--')
plt.axvline(p.toff, color='C7', linestyle='--')
plt.plot(p.tvec, p.Tdat+273.15, label='Data')
plt.plot(p.tvec[p.ton_idx:], soln_all+p.Tdat[p.ton_idx]+273.15, label='LSq fit')
plt.ylabel('Temperature [K]')
plt.legend()
plt.tick_params(right=True, top=True)
plt.subplot(3, 1, 3)
plt.axhline(0, color='C7', linestyle='--')
plt.axvline(p.ton, color='C7', linestyle='--')
plt.axvline(p.toff, color='C7', linestyle='--')
plt.plot(p.tvec[p.ton_idx::50], (soln_all-p.Tdat[p.ton_idx:]+p.Tdat[p.ton_idx])[::50], 'C8', label=r'LSq fit $-$ Data')
plt.xlabel("Time [s]")
plt.ylabel(r'$\Delta T$ [K]')
plt.tick_params(right=True, top=True)
plt.legend()

# Fit PID parameters
soln = solve_ivp(lambda t, T: h.HeaterIVP(t, T, lsq.x, params_PID=p.params_PID), p.t_span, p.T0, t_eval=p.tpid)

print("\nBest fit PID gains:")
lsqPID = least_squares(h.fit_paramsPID, p.params_PID, bounds=(0., 1.e4), args=(p.tpid, lsq.x), max_nfev=1e5)

bestfit.update(zip(["Kp", "Ki", "Kd"], lsqPID.x))
for k in ["Kp", "Ki", "Kd"]:
    print("{}: {}".format(k, bestfit[k]))

# Sanity check: Plot the temperature for the given set point and PID fit
solnPIDfit = solve_ivp(lambda t, T: h.HeaterIVP(t, T, lsq.x, params_PID=lsqPID.x), p.t_span, p.T0, t_eval=p.tpid)

# Write best fit values to disk
FILEOUT = "bestfit_" + p.TDATASET + ".out"
outputdict = OrderedDict(zip(["fileset", "t_dataset", "h_dataset", "time_start", "time_end", "time_lag", "samp_rate"], [p.FILESET, p.TDATASET, p.HDATASET, p.TSTR, p.TEND, p.TLAG, p.TSTEP]))
outputdict.update(OrderedDict((k, bestfit[k]) for k in ["Ch", "Rh", "Ct", "Rt", "Tb"]))
outputdict.update(zip(["T0", "setpoint", "sim_len", "T_tol"], [float(p.Tdat[p.ton_idx]), p.SETPOINT, p.params['sim_len'], p.params['T_tol']]))
outputdict.update(zip(["weight_ovrsh", "weight_trise", "weight_tset", "weight_errss"], [p.params['weight_ovrsh'], p.params['weight_trise'], p.params['weight_tset'], p.params['weight_errss']]))
outputdict.update(OrderedDict((k, bestfit[k]) for k in ["Kp", "Ki", "Kd"]))
with open(FILEOUT, "w") as outfile:
    json.dump(outputdict, outfile, indent=4)

plt.figure()
plt.subplot(3, 1, (1, 2))
plt.axhline(p.SETPOINT, color='C7', linestyle='--')
plt.plot(p.tpid, soln.y[0], 'C1--', label='Initial guess')
plt.plot(p.tpid, solnPIDfit.y[0], 'C0', label='Optimized PID')
plt.ylabel('Temperature [K]')
plt.legend()
plt.tick_params(right=True, top=True)
plt.subplot(3, 1, 3)
plt.semilogy(p.tpid, np.abs(p.SETPOINT - soln.y[0])/p.SETPOINT, 'C8--')
plt.semilogy(p.tpid, np.abs(p.SETPOINT - solnPIDfit.y[0])/p.SETPOINT, 'C8')
plt.xlabel("Time [s]")
plt.ylabel(r'$\Delta T/T_0$')
plt.tick_params(right=True, top=True)

# Plot the heater power Q(t)
newtt = np.delete(h.tt, list(OrderedDict.fromkeys(h.tloops)))
newQt = np.delete(h.Qt, list(OrderedDict.fromkeys(h.tloops)))
np.savetxt("qt.out", np.array([newtt, newQt]).T)

plt.figure()
plt.plot(h.tt, h.Qt, 'x--')
plt.plot(newtt, newQt)
plt.xlabel("Time [s]")
plt.ylabel("Power [%]")
plt.tick_params(right=True, top=True)
plt.show()
