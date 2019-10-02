import os, sys, json
import pygetdata as gd
import numpy as np
from scipy import integrate
from scipy.integrate import solve_ivp

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

QMIN, QMAX = 0, 100

# Physical constants
circ_params0 = np.array([initvals['Ch0'], initvals['Rh0'], initvals['Ct0'], initvals['Rt0']], dtype=float)
Tb0 = float(initvals['Tb0'])
params0 = np.append(1./circ_params0, Tb0)

# ODE initial conditions ([T0, Tdot0])
T0 = [0, 0]

# PID parameters
SETPOINT = float(initvals['setpoint'])
params_PID0 = np.array([initvals['Kp0'], initvals['Ki0'], initvals['Kd0']], dtype=float)

# Initialize auxiliary arrays
tt = np.array([])
Tt = np.array([])
Qt = np.array([])
It = np.array([])
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

tpid = np.arange(0, initvals['sim_len'], TSTEP)
t_span = [tpid[0], tpid[-1]]

###########################################################################

def Qfunc(t, T_present, T_deriv, params_PID):
    Kp, Ki, Kd = params_PID[0], params_PID[1], params_PID[2]
    try:
        h = t - tt[-1]
    except:
        h = 0.
    a = 5.e-6 * (100*h)
    b = 1.0 - a

    # Proportional term
    err = SETPOINT - T_present

    # Integral term
    global It
    integral = a * err
    try:
        integral = integral + b*It[-1]
    except:
        pass
    integral = max(0, min(0.5, integral))
    It = np.append(It, integral)

    # Derivative term
    if h != 0:
        deriv = (Tt[-1] - T_present) / (100*h)
    else:
        deriv = -T_deriv

    Q = Kp*err + Ki*integral + Kd*deriv
    return max(QMIN, min(QMAX, Q))

def rewindTime(t):
    global tt, Tt, Qt, It
    tloops.append(len(tt)-3)
    tloops.append(len(tt)-2)
    Tt = Tt[tt<t]
    Qt = Qt[tt<t]
    It = It[tt<t]
    tt = tt[tt<t]

def writeAuxArrays(t, T, Q):
    global tt, Tt, Qt
    Tt = np.append(Tt, T)
    Qt = np.append(Qt, Q)
    tt = np.append(tt, t)

def returnAuxArrays():
    return tt, Tt, Qt, It, tloops

def HeaterIVP(t, T, params_circ, params_PID = None, Q = 0):
    a, b, c, d = params_circ[0], params_circ[1], params_circ[2], params_circ[3]
    x1, x2 = T[0], T[1]
    if params_PID is not None:
        if len(tt) and t <= tt[-1]:
            rewindTime(t)
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

def fit_paramsPID(params, tvec, iparams_circ):
    solnPID = solve_ivp(lambda t, T: HeaterIVP(t, T, iparams_circ, params_PID=params), t_span, T0, t_eval=tvec)

    # Objective function to minimize
    weights = np.array([initvals['weight_ovrsh'], initvals['weight_trise'], initvals['weight_tset'], initvals['weight_errss']])
    extrema_list = np.where(np.diff(np.sign(solnPID.y[1])))[0]  # returns indices just before a sign change in derivative
    if solnPID.y[1][0] == 0:
        extrema_list = extrema_list[1:]     # if initial conditions have deriv = 0, ignore initial point

    ovrsh, t_rise, t_set, err_ss = 1.e8, 1.e8, 1.e8, 1.e8

    # Overshoot
    if len(extrema_list) > 0:
        first_peak = extrema_list[0]
        ovrsh = max(solnPID.y[0][first_peak], solnPID.y[0][first_peak+1]) - SETPOINT

    # Rise time
    t_rise = 0.

    # Settling time
    T_TOL = initvals['T_tol']
    t_setflr = np.where(np.abs(np.flip(solnPID.y[0]) - SETPOINT) > T_TOL)[0][0]
    t_set = initvals['sim_len'] - t_setflr

    # Steady state error
    if t_set < initvals['sim_len']:
        err_ss = np.mean(SETPOINT - solnPID.y[0][t_set:])

    return np.array([ovrsh, t_rise, t_set, err_ss]) * weights
