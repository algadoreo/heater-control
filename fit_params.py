import os, sys, json
import pygetdata as gd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from collections import OrderedDict

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = 16, 9
plt.rcParams["figure.dpi"] = 70

from subroutines import *

###########################################################################

# Fit (thermal) circuit parameters
print("Best fit circuit parameters:")
lsq = least_squares(fit_params, params0, bounds=(1.e-5, 1.e5), args=(tvec, Tdat-Tdat[ton_idx]), max_nfev=1e5)

bestfit = dict(zip(["Ch", "Rh", "Ct", "Rt"], 1./lsq.x[:4]))
bestfit["Tb"] = lsq.x[4]
for k in ["Ch", "Rh", "Ct", "Rt", "Tb"]:
    print("{}: {}".format(k, bestfit[k]))

# Sanity check: Plot the temperature based on best fit parameters
soln_on = solve_ivp(lambda t, T: HeaterIVP(t, T, lsq.x, Q = QMAX), [ton, toff], T0, t_eval=tvec[sli_on])
soln_off = solve_ivp(lambda t, T: HeaterIVP(t, T, lsq.x, Q = 0), [toff, tvec[-1]], [soln_on.y[0][-1], soln_on.y[1][-1]], t_eval=tvec[sli_off])
soln_all = np.concatenate((soln_on.y[0], soln_off.y[0][1:]))

plt.figure()
plt.subplot(3, 1, (1,2))
plt.axhline(Tdat[ton_idx]+273.15, color='C7', linestyle='--')
plt.axvline(ton, color='C7', linestyle='--')
plt.axvline(toff, color='C7', linestyle='--')
plt.plot(tvec, Tdat+273.15, label='Data')
plt.plot(tvec[ton_idx:], soln_all+Tdat[ton_idx]+273.15, label='LSq fit')
plt.ylabel('Temperature [K]')
plt.legend()
plt.tick_params(right=True, top=True)
plt.subplot(3, 1, 3)
plt.axhline(0, color='C7', linestyle='--')
plt.axvline(ton, color='C7', linestyle='--')
plt.axvline(toff, color='C7', linestyle='--')
plt.plot(tvec[ton_idx::50], (soln_all-Tdat[ton_idx:]+Tdat[ton_idx])[::50], 'C8', label=r'LSq fit $-$ Data')
plt.xlabel("Time [s]")
plt.ylabel(r'$\Delta T$ [K]')
plt.tick_params(right=True, top=True)
plt.legend()

# Fit PID parameters
soln = solve_ivp(lambda t, T: HeaterIVP(t, T, lsq.x, params_PID=params_PID0), t_span, T0, t_eval=tpid)

print("\nBest fit PID gains:")
lsqPID = least_squares(fit_paramsPID, params_PID0, bounds=(0., 1.e4), args=(tpid, lsq.x), max_nfev=1e5)

bestfit.update(zip(["Kp", "Ki", "Kd"], lsqPID.x))
for k in ["Kp", "Ki", "Kd"]:
    print("{}: {}".format(k, bestfit[k]))

# Sanity check: Plot the temperature for the given set point and PID fit
solnPIDfit = solve_ivp(lambda t, T: HeaterIVP(t, T, lsq.x, params_PID=lsqPID.x), t_span, T0, t_eval=tpid)

# Write best fit values to disk
FILEOUT = "bestfit_" + initvals['t_dataset'] + ".out"
outputdict = OrderedDict(zip(["fileset", "t_dataset", "h_dataset", "time_start", "time_end", "samp_rate"], [FILESET, TDATASET, HDATASET, TSTR, TEND, TSTEP]))
outputdict.update(OrderedDict((k, bestfit[k]) for k in ["Ch", "Rh", "Ct", "Rt", "Tb"]))
outputdict.update(zip(["T0", "setpoint"], [float(Tdat[ton_idx]), SETPOINT]))
outputdict.update(OrderedDict((k, bestfit[k]) for k in ["Kp", "Ki", "Kd"]))
with open(FILEOUT, "w") as outfile:
    json.dump(outputdict, outfile, indent=4)

plt.figure()
plt.subplot(3, 1, (1, 2))
plt.axhline(SETPOINT, color='C7', linestyle='--')
plt.plot(tpid, soln.y[0], 'C1--', label='Initial guess')
plt.plot(tpid, solnPIDfit.y[0], 'C0', label='Optimized PID')
plt.ylabel('Temperature [K]')
plt.legend()
plt.tick_params(right=True, top=True)
plt.subplot(3, 1, 3)
plt.semilogy(tpid, np.abs(SETPOINT - soln.y[0])/SETPOINT, 'C8--')
plt.semilogy(tpid, np.abs(SETPOINT - solnPIDfit.y[0])/SETPOINT, 'C8')
plt.xlabel("Time [s]")
plt.ylabel(r'$\Delta T/T_0$')
plt.tick_params(right=True, top=True)

# Plot the heater power Q(t)
tt, Tt, Qt, It, tloops = returnAuxArrays()
newtt = np.delete(tt, list(OrderedDict.fromkeys(tloops)))
newQt = np.delete(Qt, list(OrderedDict.fromkeys(tloops)))
np.savetxt("qt.out", np.array([newtt, newQt]).T)

plt.figure()
plt.plot(tt, Qt, 'x--')
plt.plot(newtt, newQt)
plt.xlabel("Time [s]")
plt.ylabel("Power [%]")
plt.tick_params(right=True, top=True)
plt.show()
