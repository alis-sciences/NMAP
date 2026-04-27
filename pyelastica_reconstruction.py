#!/usr/bin/env python3
"""
pyelastica_reconstruction.py
============================
Drive a PyElastica Cosserat rod using the curvature time-series extracted
from the MuJoCo CCMN simulation, with environment-appropriate drag.
The PrescribedCurvatureCallback applies κ(s,t) as a rest-curvature target
at each simulation substep.  PyElastica's elastic restoring forces drive the
rod toward the prescribed shape while anisotropic RFT fluid / substrate drag
acts physically on the deforming elastic body.
Two simulation modes (selected by --env):
    water     : RFT drag with C_N/C_T = 2.0  (swim)
    substrate : RFT drag with C_N/C_T = 8.0  (crawl)
Saves output .npz with:
    positions   : (T_out, n_elem+1, 3)   node positions along rod
    directors   : (T_out, 3, 3, n_elem)  material frame directors
    velocities  : (T_out, n_elem+1, 3)   node velocities
    kappa_actual: (T_out, n_elem-1, 3)   actual rod curvature vector
    t           : (T_out,)               time axis
    env         : str                    'water' or 'substrate'
Requirements:
    pip install pyelastica
"""
import argparse
import numpy as np

try:
    import elastica
    from elastica import *
    from elastica.timestepper.symplectic_steppers import PositionVerlet
    from elastica.timestepper import integrate
    from elastica.callback_functions import CallBackBaseClass
    from elastica.external_forces import NoForces
except ImportError:
    raise ImportError('PyElastica not found. Install with:  pip install pyelastica')

# ── Physical constants matching progressive_mixed_env.py ─────────────────────
_C_N_WATER     = 2.0    # normal drag coefficient ratio (water)
_C_N_SUBSTRATE = 8.0    # normal drag coefficient ratio (substrate)
_BASE_DRAG     = 5e-4   # tangential drag gain calibrated to MuJoCo

# ── C. elegans biological reference (metres) — kept for documentation ─────────
_WORM_LENGTH_M = 1.0e-3
_WORM_RADIUS_M = 4.0e-5

# Work in mm units internally to stay within PyElastica numerical tolerance
_SCALE = 1000.0          # metres → mm


# ─────────────────────────────────────────────────────────────────────────────
# PrescribedCurvature callback
# ─────────────────────────────────────────────────────────────────────────────

class PrescribedCurvatureCallback(CallBackBaseClass):
    """
    Applies time-varying rest curvature κ(s,t) to the rod at every substep.
    The callback interpolates the MuJoCo control-rate curvature array
    (sampled at control_dt) to the current simulation time, then sets
    rod.rest_kappa to drive elastic restoring forces toward the prescribed
    body shape.
    """
    def __init__(self, step_skip, callback_params, kappa_array,
                 control_dt, kappa_weight=1.0):
        super().__init__()
        self.step_skip   = step_skip
        self.params      = callback_params
        self.kappa_array = kappa_array   # (T, n_elem) float32
        self.control_dt  = control_dt
        self.kappa_weight = kappa_weight
        self.every       = step_skip
        self.params['positions']    = []
        self.params['directors']    = []
        self.params['velocities']   = []
        self.params['kappa_actual'] = []
        self.params['t']            = []

    def make_callback(self, system, time, current_step: int):
        # Interpolate curvature at current simulation time
        ctrl_idx = np.clip(
            int(time / self.control_dt), 0, self.kappa_array.shape[0] - 1
        )
        alpha = (time - ctrl_idx * self.control_dt) / self.control_dt
        if ctrl_idx + 1 < self.kappa_array.shape[0]:
            kappa_now = ((1 - alpha) * self.kappa_array[ctrl_idx]
                         + alpha     * self.kappa_array[ctrl_idx + 1])
        else:
            kappa_now = self.kappa_array[ctrl_idx]

        # Resample n_elem → n_elem-1 by averaging neighbours
        kappa_resampled = 0.5 * (kappa_now[:-1] + kappa_now[1:])

        # Component 0 = dorsoventral bending; 1,2 = lateral, twist = 0
        system.rest_kappa[0, :] = self.kappa_weight * kappa_resampled
        system.rest_kappa[1, :] = 0.0
        system.rest_kappa[2, :] = 0.0

        if current_step % self.step_skip == 0:
            self.params['positions'].append(system.position_collection.copy().T)
            self.params['directors'].append(system.director_collection.copy())
            self.params['velocities'].append(system.velocity_collection.copy().T)
            self.params['kappa_actual'].append(system.kappa.copy().T)
            self.params['t'].append(time)


# ─────────────────────────────────────────────────────────────────────────────
# Anisotropic RFT drag
# ─────────────────────────────────────────────────────────────────────────────

class AnisotropicRFTDrag(NoForces):
    """RFT drag with separate normal/tangential coefficients."""
    def __init__(self, c_tangential, c_normal):
        super().__init__()
        self.c_t = c_tangential
        self.c_n = c_normal

    def apply_forces(self, system, time=0.0):
        tangent    = system.director_collection[2, :, :]  # (3, n_elem)
        vel        = system.velocity_collection[:, :-1]   # (3, n_elem)
        v_t_scalar = np.einsum('ij,ij->j', vel, tangent)
        v_t        = v_t_scalar * tangent
        v_n        = vel - v_t
        drag       = -(self.c_t * v_t + self.c_n * v_n)
        system.external_forces[:, :-1] += drag * system.lengths


# ─────────────────────────────────────────────────────────────────────────────
# Simulator class
# ─────────────────────────────────────────────────────────────────────────────

class WormSimulator(BaseSystemCollection, Connections, Forcing,
                    CallBacks, Constraints):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Main reconstruction function
# ─────────────────────────────────────────────────────────────────────────────

def run_pyelastica_reconstruction(curvature_npz, kinematics_npz,
                                   output_path='pyelastica_output.npz',
                                   env='water', dt_sim=1e-4,
                                   step_skip=30, kappa_weight=1.0):
    print(f'📂 Loading curvature: {curvature_npz}')
    cv  = np.load(curvature_npz)
    kin = np.load(kinematics_npz, allow_pickle=True)

    kappa      = cv['kappa']           # (T, n_elem)
    control_dt = float(cv['control_dt'])
    n_elem     = int(cv['n_elem'])

    # ── Scale to mm units to satisfy PyElastica numerical tolerance ───────────
    rod_length = float(cv['rod_length']) * _SCALE        # mm
    radius     = float(cv['worm_radius']) * _SCALE       # mm
    E          = float(cv['elastic_modulus']) * 1e-6     # Pa → N/mm² (MPa)
    G          = float(cv['shear_modulus'])  * 1e-6      # Pa → N/mm²
    rho        = float(cv['density'])        * 1e-9      # kg/m³ → kg/mm³

    T_ctrl      = kappa.shape[0]
    t_final     = T_ctrl * control_dt
    n_steps_sim = int(t_final / dt_sim)

    c_n_ratio = _C_N_WATER if env == 'water' else _C_N_SUBSTRATE
    c_t = _BASE_DRAG
    c_n = _BASE_DRAG * c_n_ratio

    print(f'   Environment : {env}  (C_N/C_T = {c_n_ratio})')
    print(f'   Rod length  : {rod_length:.3f} mm  |  radius: {radius*1e3:.2f} µm')
    print(f'   E={E:.3f} N/mm²  G={G:.3f} N/mm²  ρ={rho:.2e} kg/mm³')
    print(f'   Sim steps   : {n_steps_sim:,}  (dt={dt_sim:.2e} s)')

    # ── Build Cosserat rod ────────────────────────────────────────────────────
    simulator = WormSimulator()

    rod = CosseratRod.straight_rod(
        n_elements=n_elem,
        start=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        base_length=rod_length,
        base_radius=radius,
        density=rho,
        youngs_modulus=E,
        shear_modulus=G,
    )
    simulator.append(rod)

    simulator.add_forcing_to(rod).using(
        AnisotropicRFTDrag, c_tangential=c_t, c_normal=c_n,
    )

    callback_params = {}
    simulator.collect_diagnostics(rod).using(
        PrescribedCurvatureCallback,
        step_skip=step_skip,
        callback_params=callback_params,
        kappa_array=kappa,
        control_dt=control_dt,
        kappa_weight=kappa_weight,
    )

    simulator.finalize()
    timestepper = PositionVerlet()

    print('🚀 Running PyElastica simulation…')
    integrate(timestepper, simulator, t_final, n_steps_sim)
    print('✅ Simulation complete.')

    positions    = np.array(callback_params['positions'],    dtype=np.float32)
    velocities   = np.array(callback_params['velocities'],   dtype=np.float32)
    kappa_actual = np.array(callback_params['kappa_actual'], dtype=np.float32)
    t_out        = np.array(callback_params['t'],            dtype=np.float32)

    np.savez(
        output_path,
        positions=positions, velocities=velocities,
        kappa_actual=kappa_actual, t=t_out,
        env=np.array(env),
        rod_length=np.float32(rod_length),
        n_elem=np.int32(n_elem),
        c_n_ratio=np.float32(c_n_ratio),
        dt_sim=np.float32(dt_sim),
        step_skip=np.int32(step_skip),
    )

    head_disp = float(np.linalg.norm(positions[-1, 0, :2] - positions[0, 0, :2]))
    print(f'💾 Saved: {output_path}')
    print(f'   Output frames    : {len(t_out)}')
    print(f'   Head displacement: {head_disp:.4f} mm')
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyElastica Cosserat rod reconstruction from MuJoCo kinematics')
    parser.add_argument('--curvature',    default='curvature.npz')
    parser.add_argument('--kinematics',   default='kinematics_export.npz')
    parser.add_argument('--output',       default='pyelastica_output.npz')
    parser.add_argument('--env',          choices=['water', 'substrate'], default='water')
    parser.add_argument('--dt_sim',       type=float, default=1e-4)
    parser.add_argument('--step_skip',    type=int,   default=30)
    parser.add_argument('--kappa_weight', type=float, default=1.0)
    args = parser.parse_args()

    run_pyelastica_reconstruction(
        curvature_npz=args.curvature,
        kinematics_npz=args.kinematics,
        output_path=args.output,
        env=args.env,
        dt_sim=args.dt_sim,
        step_skip=args.step_skip,
        kappa_weight=args.kappa_weight,
    )
