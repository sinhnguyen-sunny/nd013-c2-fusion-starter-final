# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))


sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params


class Filter:
    '''Kalman filter class'''

    def __init__(self):
        self.dim_state = params.dim_state
        self.dt = params.dt
        self.q = params.q

    def F(self):
        dt = self.dt
        F = np.eye(self.dim_state)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return np.matrix(F)

    def Q(self):
        dt = self.dt
        q = self.q
        q1 = (dt ** 3) * q / 3
        q2 = (dt ** 2) * q / 2
        q3 = dt * q
        return np.matrix([[q1, 0, 0, q2, 0, 0], [0, q1, 0, 0, q2, 0], [0, 0, q1, 0, 0, q2],
                          [q2, 0, 0, q3, 0, 0], [0, q2, 0, 0, q3, 0], [0, 0, q2, 0, 0, q3]])

    def predict(self, track):
        F = self.F()
        Q = self.Q()
        x = F @ track.x
        P = F @ track.P @ F.T + Q
        track.set_x(x)
        track.set_P(P)

    def update(self, track, meas):
        H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)
        K = track.P @ H.T @ np.linalg.inv(S)
        x = track.x + K @ gamma
        I = np.identity(self.dim_state)
        P = (I - K @ H) @ track.P

        track.set_x(x)
        track.set_P(P)
        track.update_attributes(meas)

    def gamma(self, track, meas):
        gm = meas.z - meas.sensor.get_hx(track.x)
        return gm

    def S(self, track, meas, H):
        P = track.P
        R = meas.R
        S = H @ P @ H.T + R

        return S
