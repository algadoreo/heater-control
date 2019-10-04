# -*- coding: utf-8 -*-

import os, sys, json
import pygetdata as gd
import numpy as np
from scipy import integrate
from scipy.integrate import solve_ivp

class initvals:
    def __init__(self):
        # Load parameters from file, or exit if no file found
        try:
            with open('params.json') as handle:
                self.params = json.loads(handle.read())
        except:
            print("File 'params.json' not found.")
            sys.exit()

        # Constants
        self.TSTR = int(self.params['time_start'])
        self.TEND = int(self.params['time_end'])
        self.TLEN = self.TEND - self.TSTR
        self.TLAG = int(self.params['time_lag'])
        self.TSTEP = float(self.params['samp_rate'])    # timestep [seconds]

        self.QMIN, self.QMAX = 0, 100

        # Physical constants
        circ_params0 = np.array([self.params['Ch0'], self.params['Rh0'], self.params['Ct0'], self.params['Rt0']], dtype=float)
        Tb0 = float(self.params['Tb0'])
        self.params0 = np.append(1./circ_params0, Tb0)

        # ODE initial conditions ([T0, Tdot0])
        self.T0 = [0, 0]

        # PID parameters
        self.SETPOINT = float(self.params['setpoint'])
        self.params_PID0 = np.array([self.params['Kp0'], self.params['Ki0'], self.params['Kd0']], dtype=float)

        # Load data
        self.ROOTDIR = self.params['rootdir']
        self.FILESET = self.params['fileset']
        self.TDATASET = 't_' + self.params['t_dataset']
        self.HDATASET = 'h_' + self.params['t_dataset']
        if ('h_dataset' in self.params.keys()) and (self.params['h_dataset'] != ""):
            self.HDATASET = 'h_' + self.params['h_dataset']

        df = gd.dirfile(os.path.join(self.ROOTDIR, self.FILESET))
        self.Tdat = df.getdata(self.TDATASET)[self.TSTR:self.TEND]
        self.Qdat = np.maximum(self.QMIN, np.minimum(self.QMAX, df.getdata(self.HDATASET)))[self.TSTR-self.TLAG:self.TEND-self.TLAG]

        self.tvec = self.TSTEP * np.arange(self.TLEN)

        idx_power_on = np.where(self.Qdat>0.5)[0]
        self.ton_idx = idx_power_on[0]
        self.toff_idx = idx_power_on[-1] + 1
        self.sli_on = slice(self.ton_idx, self.toff_idx+1) # +1 because initial conds. needed at endpoints
        self.sli_off = slice(self.toff_idx, len(self.tvec))

        self.ton = self.ton_idx * self.TSTEP
        self.toff = self.toff_idx * self.TSTEP

        self.tpid = np.arange(0, self.params['sim_len'], self.TSTEP)
        self.t_span = [self.tpid[0], self.tpid[-1]]

###########################################################################
class heater:
    def __init__(self, params):
        # Set parameters
        self.p = params

        # Initialize auxiliary arrays
        self.tt = np.array([])
        self.Tt = np.array([])
        self.Qt = np.array([])
        self.It = np.array([])
        self.tloops = []

    def Qfunc(self, t, T_present, T_deriv, params_PID):
        Kp, Ki, Kd = params_PID[0], params_PID[1], params_PID[2]
        try:
            h = t - self.tt[-1]
        except:
            h = 0.
        a = 5.e-6 * (100*h)
        b = 1.0 - a

        # Proportional term
        err = self.p.SETPOINT - T_present

        # Integral term
        integral = a * err
        try:
            integral = integral + b*self.It[-1]
        except:
            pass
        integral = max(0, min(0.5, integral))
        self.It = np.append(self.It, integral)

        # Derivative term
        if h != 0:
            deriv = (self.Tt[-1] - T_present) / (100*h)
        else:
            deriv = -T_deriv

        Q = Kp*err + Ki*integral + Kd*deriv
        return max(self.p.QMIN, min(self.p.QMAX, Q))

    def rewindTime(self, t):
        self.tloops.append(len(self.tt)-3)
        self.tloops.append(len(self.tt)-2)
        self.Tt = self.Tt[self.tt<t]
        self.Qt = self.Qt[self.tt<t]
        self.It = self.It[self.tt<t]
        self.tt = self.tt[self.tt<t]

    def writeAuxArrays(self, t, T, Q):
        self.Tt = np.append(self.Tt, T)
        self.Qt = np.append(self.Qt, Q)
        self.tt = np.append(self.tt, t)

    def HeaterIVP(self, t, T, params_circ, params_PID = None, Q = 0):
        a, b, c, d = params_circ[0], params_circ[1], params_circ[2], params_circ[3]
        x1, x2 = T[0], T[1]
        if params_PID is not None:
            if len(self.tt) and t <= self.tt[-1]:
                self.rewindTime(t)
            Q = self.Qfunc(t, x1, x2, params_PID)
            self.writeAuxArrays(t, x1, Q)

        dx1 = x2
        dx2 = a*b*c*(Q - d*(x1-params_circ[4])) - (a*b + b*c + c*d)*x2

        return [dx1, dx2]

    def fit_params(self, params, tvec, Tt):
        soln_on = solve_ivp(lambda t, T: self.HeaterIVP(t, T, params, Q = self.p.QMAX), [self.p.ton, self.p.toff], self.p.T0, t_eval=tvec[self.p.sli_on])

        T1 = [soln_on.y[0][-1], soln_on.y[1][-1]]
        soln_off = solve_ivp(lambda t, T: self.HeaterIVP(t, T, params, Q = 0), [self.p.toff, tvec[-1]], T1, t_eval=tvec[self.p.sli_off])

        soln_all = np.concatenate((soln_on.y[0], soln_off.y[0][1:]))
        return (Tt[self.p.ton_idx:] - soln_all)

    def fit_paramsPID(self, params, tvec, iparams_circ):
        # Reset auxiliary arrays
        self.__init__(self.p)

        # Solve heater IVP using PID values
        solnPID = solve_ivp(lambda t, T: self.HeaterIVP(t, T, iparams_circ, params_PID=params), self.p.t_span, self.p.T0, t_eval=tvec)

        # Objective function to minimize
        weights = np.array([self.p.params['weight_ovrsh'], self.p.params['weight_trise'], self.p.params['weight_tset'], self.p.params['weight_errss']])
        extrema_list = np.where(np.diff(np.sign(solnPID.y[1])))[0]  # returns indices just before a sign change in derivative
        if solnPID.y[1][0] == 0:
            extrema_list = extrema_list[1:]     # if initial conditions have deriv = 0, ignore initial point

        ovrsh, t_rise, t_set, err_ss = 1.e8, 1.e8, 1.e8, 1.e8

        # Overshoot
        if len(extrema_list) > 0:
            first_peak = extrema_list[0]
            ovrsh = max(solnPID.y[0][first_peak], solnPID.y[0][first_peak+1]) - self.p.SETPOINT

        # Rise time
        t_rise = 0.

        # Settling time
        T_TOL = self.p.params['T_tol']
        t_setflr = np.where(np.abs(np.flip(solnPID.y[0]) - self.p.SETPOINT) > T_TOL)[0][0]
        t_set = self.p.params['sim_len'] - t_setflr

        # Steady state error
        if t_set < self.p.params['sim_len']:
            err_ss = np.mean(self.p.SETPOINT - solnPID.y[0][t_set:])

        return np.array([ovrsh, t_rise, t_set, err_ss]) * weights
