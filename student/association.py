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
from scipy.stats.distributions import chi2
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params

import numpy as np
from scipy.stats.distributions import chi2
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))


sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params


class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''

    def __init__(self):
        self.association_matrix = np.empty((0, 0))
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):
        N = len(track_list)
        M = len(meas_list)
        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))

        self.association_matrix = np.inf * np.ones((N, M))

        for i in range(N):
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]
                MHD = self.MHD(track, meas, KF)
                if self.gating(MHD, meas.sensor):
                    self.association_matrix[i, j] = MHD

    def get_closest_track_and_meas(self):
        M = self.association_matrix
        if np.min(M) == np.inf:
            return np.nan, np.nan

        min_ij = np.unravel_index(np.argmin(M, axis=None), M.shape)
        track_idx = min_ij[0]
        meas_idx = min_ij[1]
        M = np.delete(M, track_idx, axis=0)
        M = np.delete(M, meas_idx, axis=1)
        self.association_matrix = M

        update_track = self.unassigned_tracks[track_idx]
        update_meas = self.unassigned_meas[meas_idx]

        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)

        return update_track, update_meas

    def gating(self, MHD, sensor):
        threshold = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        return MHD < threshold

    def MHD(self, track, meas, KF):
        H = meas.sensor.get_H(track.x)
        gamma = KF.gamma(track, meas)
        S = KF.S(track, meas, H)
        MHD = float(gamma.T @ np.linalg.inv(S) @ gamma)
        return MHD

    def associate_and_update(self, manager, meas_list, KF):
        self.associate(manager.track_list, meas_list, KF)

        while self.association_matrix.shape[0] > 0 and self.association_matrix.shape[1] > 0:
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]

            if not meas_list[0].sensor.in_fov(track.x):
                continue

            KF.update(track, meas_list[ind_meas])
            manager.handle_updated_track(track)
            manager.track_list[ind_track] = track

        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)

        for track in manager.track_list:
            print('track', track.id, 'score =', track.score)
