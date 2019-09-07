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

# Load parameters from file, or exit if no file found
try:
    with open('params.json') as handle:
        initvals = json.loads(handle.read())
except:
    print("File 'params.json' not found.")
    sys.exit()

# Constants
TSTR = int(initvals['time_start'])
TEND = int(initvals['time_end'])
TLEN = TEND - TSTR
TSTEP = float(initvals['samp_rate'])    # timestep [seconds]

QMIN = 0
QMAX = 100

# Physical constants
circ_params0 = np.array([initvals['Ch0'], initvals['Rh0'], initvals['Ct0'], initvals['Rt0']], dtype=float)
Tb0 = float(initvals['Tb0'])
params0 = np.append(1./circ_params0, Tb0)

# ODE initial conditions ([T0, Tdot0])
T0 = [0, 0]

# PID parameters
SETPOINT = float(initvals['setpoint'])
params_PID0 = np.array([initvals['Kp0'], initvals['Ki0'], initvals['Kd0']], dtype=float)
TOPT = float(initvals['time_opt'])      # time from which to start PID optimization
ttopt = int(TOPT/TSTEP)

# Initialize auxiliary arrays
tt = np.array([])
Tt = np.array([])
Qt = np.array([])
tloops = []

###########################################################################
# Load data
ROOTDIR = initvals['rootdir']
FILESET = initvals['fileset']
TDATASET = 't_' + initvals['t_dataset']
HDATASET = 'h_' + initvals['t_dataset']
if ('h_dataset' in initvals.keys()) and (initvals['h_dataset'] != ""):
    HDATASET = 'h_' + initvals['h_dataset']

df = gd.dirfile(os.path.join(ROOTDIR, FILESET))
Tdat = df.getdata(TDATASET)[TSTR:TEND]
Qdat = np.maximum(QMIN, np.minimum(QMAX, df.getdata(HDATASET)))[TSTR:TEND]

tvec = TSTEP * np.arange(TLEN)

idx_power_on = np.where(Qdat>0.5)[0]
ton_idx = idx_power_on[0]
toff_idx = idx_power_on[-1] + 1
sli_on = slice(ton_idx, toff_idx+1) # +1 because initial conds. needed at endpoints
sli_off = slice(toff_idx, len(tvec))

ton = ton_idx * TSTEP
toff = toff_idx * TSTEP

###########################################################################
def Qfunc(t, T_present, T_deriv, params_PID):
    Kp, Ki, Kd = params_PID[0], params_PID[1], params_PID[2]
    
    err = SETPOINT - T_present
    try:
        integral = np.dot(SETPOINT - Tt, np.ediff1d(tt, to_begin=0)) # Riemann sum
    except:
        integral = 0.
    deriv = -T_deriv
    Q = Kp*err + Ki*integral + Kd*deriv
    return max(QMIN, min(QMAX, Q))

def writeAuxArrays(t, T, Q):
    global tt, Tt, Qt
    if len(tt) and t <= tt[-1]:
        tloops.append(len(tt)-3)
        tloops.append(len(tt)-2)
        Tt = Tt[tt<t]
        Qt = Qt[tt<t]
        tt = tt[tt<t]
    Tt = np.append(Tt, T)
    Qt = np.append(Qt, Q)
    tt = np.append(tt, t)

def HeaterIVP(t, T, params_circ, params_PID = None, Q = 0):
    a, b, c, d = params_circ[0], params_circ[1], params_circ[2], params_circ[3]
    x1, x2 = T[0], T[1]
    if params_PID is not None:
        Q = Qfunc(t, x1, x2, params_PID)
        writeAuxArrays(t, x1, Q)
    
    dx1 = x2
    dx2 = a*b*c*(Q - d*(x1-params_circ[4])) - (a*b + b*c + c*d)*x2
    
    return [dx1, dx2]

def fit_params(params, tvec, Tt):
    soln_on = solve_ivp(lambda t, T: HeaterIVP(t, T, params, Q = QMAX), [ton, toff], T0, t_eval=tvec[sli_on])
    
    T1 = [soln_on.y[0][-1], soln_on.y[1][-1]]
    soln_off = solve_ivp(lambda t, T: HeaterIVP(t, T, params, Q = 0), [toff, tvec[-1]], T1, t_eval=tvec[sli_off])
    
    soln_all = np.concatenate((soln_on.y[0], soln_off.y[0][1:]))
    return (Tt[ton_idx:] - soln_all)

def fit_paramsPID(params, tvec):
    solnPID = solve_ivp(lambda t, T: HeaterIVP(t, T, lsq.x, params_PID=params), t_span, T0, t_eval=tpid)
    return (SETPOINT - solnPID.y[0][ttopt:])

###########################################################################
#
# MAIN
#
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
tpid = np.arange(0, 1000, TSTEP)
t_span = [tpid[0], tpid[-1]]
soln = solve_ivp(lambda t, T: HeaterIVP(t, T, lsq.x, params_PID=params_PID0), t_span, T0, t_eval=tpid)

print("\nBest fit PID gains:")
lsqPID = least_squares(fit_paramsPID, params_PID0, bounds=(0., 1.e4), args=(tpid,), max_nfev=1e5)

bestfit.update(zip(["Kp", "Ki", "Kd"], lsqPID.x))
for k in ["Kp", "Ki", "Kd"]:
    print("{}: {}".format(k, bestfit[k]))

# Sanity check: Plot the temperature for the given set point and PID fit
solnPIDfit = solve_ivp(lambda t, T: HeaterIVP(t, T, lsq.x, params_PID=lsqPID.x), t_span, T0, t_eval=tpid)

# Write best fit values to disk
FILEOUT = "bestfit_" + initvals['t_dataset'] + ".out"
outputdict = OrderedDict(zip(["fileset", "t_dataset", "h_dataset", "time_start", "time_end", "samp_rate", "t_opt"], [FILESET, TDATASET, HDATASET, TSTR, TEND, TSTEP, TOPT]))
outputdict.update(OrderedDict((k, bestfit[k]) for k in ["Ch", "Rh", "Ct", "Rt", "Tb"]))
outputdict.update(zip(["T0", "setpoint"], [float(Tdat[ton_idx]), SETPOINT]))
outputdict.update(OrderedDict((k, bestfit[k]) for k in ["Kp", "Ki", "Kd"]))
with open(FILEOUT, "w") as outfile:
    json.dump(outputdict, outfile, indent=4)

plt.figure()
plt.subplot(3, 1, (1, 2))
plt.axhline(SETPOINT, color='C7', linestyle='--')
plt.axvline(TOPT, color='C7', linestyle='--')
plt.plot(tpid, soln.y[0], 'C1--', label='Initial guess')
plt.plot(tpid, solnPIDfit.y[0], 'C0', label='Optimized PID')
plt.ylabel('Temperature [K]')
plt.legend()
plt.tick_params(right=True, top=True)
plt.subplot(3, 1, 3)
plt.axvline(TOPT, color='C7', linestyle='--')
plt.semilogy(tpid, np.abs(SETPOINT - soln.y[0])/SETPOINT, 'C8--')
plt.semilogy(tpid, np.abs(SETPOINT - solnPIDfit.y[0])/SETPOINT, 'C8')
plt.xlabel("Time [s]")
plt.ylabel(r'$\Delta T/T_0$')
plt.tick_params(right=True, top=True)

# Plot the heater power Q(t)
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
