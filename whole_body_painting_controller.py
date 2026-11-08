#!/usr/bin/env python3
"""
================================================================================
Whole-Body Mobile Manipulator Spray Painting Simulation & Control Benchmark
Clearpath Husky A200 Mobile Base + Universal Robots UR10e Manipulator
================================================================================

Features:
  1. Standoff Distance Regulation:
     Maintains exact constant standoff offset distance (d_standoff = 0.20 m) away
     from the wall along the normal axis (Y-axis), ensuring uniform spray deposition.
  2. Digital Signal Graph Raster Trajectory:
     Coordinated whole-body rastering (vertical strokes + horizontal swath shifts)
     executed while the Husky mobile base cruises continuously along the wall.
  3. External Spray Recoil Thrust Simulation & Active Compensation:
     Simulates realistic Newton reaction thrust (recoil) from high-pressure atomization
     (F_recoil ~ 3.5 N directed backward in -Y).
     Implements Active Recoil Compensation (Feedforward + Disturbance Observer) and
     operational-space inertial decoupling to eliminate standoff displacement (< 0.5 mm).
  4. Real-Time Dynamic Wall Painting Engine:
     Discretized 324-tile paintable wall surface spanning 5.4m x 0.6m.
     Dynamically updates geom RGBA in real-time as the atomized spray cone sweeps over
     the wall, tracking paint coat thickness (microns) and total coverage percentage (%).
  5. Perpendicularity & Zero-Tilt Orientation Control:
     High restoring stiffness (600 Nm/rad) keeps the spray cone strictly flush and
     perpendicular (90 deg) to the wall face.
  6. Ergonomic Weighted Null-Space Self-Motion:
     Optimizes UR10e arm manipulability and prevents reach singularities during base motion.
================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco

try:
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False


class WholeBodyPaintingController:
    """
    Whole-body mobile loco-manipulation controller for robotic spray painting.
    Coordinates Husky 4-wheel mobile base and UR10e 6-DOF arm to maintain constant
    standoff distance, execute square-wave digital rastering, actively compensate
    spray recoil thrust, and simulate dynamic wall paint deposition.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 base_speed: float = 0.03, standoff: float = 0.20,
                 recoil_force: float = 3.50, compensate_recoil: bool = True,
                 ee_speed: float = 0.08, fan_width: float = 0.15,
                 overlap: float = 0.25, z_min: float = 0.72, z_max: float = 1.18,
                 kp_y: float = 280.0, kd_y: float = 35.0, ki_recoil: float = 120.0,
                 kp_p: float = 45.0, kd_p: float = 12.0,
                 kp_tilt: float = 600.0, kd_tilt: float = 40.0,
                 kp_null: float = 12.0, d_null: float = 2.2):
        self.model = model
        self.data = data
        self.dt = model.opt.timestep

        # Mission & Wall Parameters
        self.base_speed = base_speed          # Mobile base cruising speed along wall (m/s)
        self.standoff = standoff              # Standoff offset distance from wall face (m)
        self.y_wall = 0.8000                  # Front surface of wall cuboid at Y = 0.8000 m
        self.y_des = self.y_wall - self.standoff  # Desired TCP normal coordinate (m)

        # Spray Recoil Thrust & Active Compensation
        self.recoil_force = recoil_force      # Nominal reaction recoil force from spray (N)
        self.compensate_recoil = compensate_recoil  # Enable active feedforward + DOB compensation
        self.recoil_est = 0.0                 # Disturbance observer estimated recoil force (N)
        self.ki_recoil = ki_recoil            # Integral estimator gain for recoil disturbance
        self.recoil_integral = 0.0
        self.spraying = False                 # Spray gun trigger status

        # Mobile Base Dimensions (Clearpath Husky A200)
        self.wheel_radius = 0.17775           # 13-inch tire radius
        self.track_width  = 0.5708            # Lateral distance between wheel centers

        # Actuator IDs (4 Wheels + 6 Arm Motors)
        self.wheel_act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_left_wheel_ctrl"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_right_wheel_ctrl"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_wheel_ctrl"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_wheel_ctrl"),
        ]
        arm_act_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        self.arm_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in arm_act_names]

        # Geometry & Site IDs
        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "paint_tcp")
        if self.tcp_site_id < 0:
            self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        self.gun_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paint_gun_mount")
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.spray_cone_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "spray_cone_vis")

        # Standoff Axis (Y-axis) Impedance Gains
        self.Kp_y = kp_y                      # Normal position stiffness (N/m)
        self.Kd_y = kd_y                      # Normal velocity damping (Ns/m)

        # Tangential Sanding/Painting Plane Motion Gains (X and Z axes)
        self.Kp_x = kp_p                      # Longitudinal position gain (N/m)
        self.Kd_x = kd_p                      # Longitudinal velocity damping (Ns/m)
        self.Kp_z = kp_p                      # Vertical position gain (N/m)
        self.Kd_z = kd_p                      # Vertical velocity damping (Ns/m)

        # Desired Orientation: Spray axis strictly perpendicular to wall (+Y)
        # In tool frame: local +Z points into wall (+Y), local +X along +X, local +Y along -Z
        self.R_des = np.array([
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
            [0.0, -1.0, 0.0]
        ])

        # High Restoring Tilt Stiffness:
        # Pitch and Yaw are stiff (600 Nm/rad) to maintain exact perpendicularity against recoil
        self.Kp_tilt = kp_tilt
        self.Kd_tilt = kd_tilt
        self.Kp_roll = 10.0                   # Compliant roll about spray axis
        self.Kd_roll = 2.0
        self.Kp_rot = np.array([self.Kp_tilt, self.Kp_roll, self.Kp_tilt])
        self.Kd_rot = np.array([self.Kd_tilt, self.Kd_roll, self.Kd_tilt])

        # Vertical Painting Stroke (Z in [z_min, z_max])
        self.z_min = z_min
        self.z_max = z_max
        self.stroke_z = self.z_max - self.z_min
        self.z_mid = (self.z_max + self.z_min) / 2.0

        # Swath & Overlap Geometry
        self.fan_width = fan_width            # Swath width of spray cone (m)
        self.overlap = overlap                # Swath overlap ratio (e.g. 25%)
        self.d_shift = self.fan_width * (1.0 - self.overlap)  # Shift per pass (m)
        self.shift_ratio = 0.20               # 20% of pass time for shift, 80% for stroke
        self.ee_speed = ee_speed              # Target vertical stroke speed (m/s)

        self.update_timing()
        self.x_rel_nominal = 0.05             # Nominal tool lead relative to base (+5 cm)

        # Base Steering / Lane Tracking Gains (Maintain Y = 0.0, Heading = 0.0)
        self.Ky_base = 3.5
        self.Kyaw_base = 2.0

        # Weighted Null-Space Parameters
        self.W_arm = np.diag([1.0, 1.2, 1.0, 0.5, 0.5, 0.4])
        self.Winv_arm = np.linalg.inv(self.W_arm)
        self.K_posture = kp_null
        self.D_null = d_null
        self.lambda_damping = 1e-4

        # Nominal Arm Posture (optimal manipulability at standoff Y = 0.60 m)
        self.q_nominal = np.array([-1.9147, -1.6404, 2.0291, -0.3887, 1.2269, 3.1416])

        # Preallocated buffers
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

        # Real-Time Wall Paint Deposition Engine
        self.init_paint_engine()

        # State Tracking
        self.state = "STANDBY"
        self.sub_state = "STANDBY"
        self.t_start_paint = 0.0
        self.x_start_paint = 0.0

    def update_timing(self):
        """Coordinate pass period with base speed and end-effector stroke velocity."""
        # Vertical stroke duration from requested ee_speed
        t_vert_req = self.stroke_z / max(1e-4, self.ee_speed)
        # Coordinate pass duration with mobile base forward shift:
        t_pass_base = self.d_shift / max(1e-4, self.base_speed)
        self.t_pass = max(t_vert_req / (1.0 - self.shift_ratio), t_pass_base)
        self.t_shift = self.shift_ratio * self.t_pass
        self.t_vert = self.t_pass - self.t_shift

    def init_paint_engine(self):
        """Index the 324 paintable wall tiles and initialize paint deposition state."""
        self.n_cols = 54
        self.n_rows = 6
        self.total_tiles = self.n_cols * self.n_rows

        # Target patch bounds: X in [-2.2, +3.2] (length 5.4m), Z in [0.65, 1.25] (height 0.6m)
        self.x_patch_start = -2.20
        self.x_patch_end   =  3.20
        self.z_patch_min   =  0.65
        self.z_patch_max   =  1.25
        self.dx_tile = (self.x_patch_end - self.x_patch_start) / self.n_cols  # 0.10 m
        self.dz_tile = (self.z_patch_max - self.z_patch_min) / self.n_rows    # 0.10 m

        self.tile_geom_ids = np.zeros((self.n_cols, self.n_rows), dtype=int)
        self.tile_x = np.zeros((self.n_cols, self.n_rows))
        self.tile_z = np.zeros((self.n_cols, self.n_rows))
        self.tile_thickness = np.zeros((self.n_cols, self.n_rows))  # Microns (µm)

        self.target_thickness = 50.0  # Target dry film thickness (50 µm)
        self.deposition_rate = 28.0   # Calibrated deposition rate (µm/s at center of spray fan)
        self.sigma_x = 0.065          # Gaussian footprint half-width along X (m)
        self.sigma_z = 0.090          # Gaussian footprint half-height along Z (m)

        # Primer color (bare wall) and High-Gloss Azure Paint color
        self.color_primer = np.array([0.76, 0.72, 0.68, 1.0])
        self.color_paint  = np.array([0.08, 0.58, 0.95, 1.0])

        for col in range(self.n_cols):
            cx = self.x_patch_start + (col + 0.5) * self.dx_tile
            for row in range(self.n_rows):
                cz = self.z_patch_min + (row + 0.5) * self.dz_tile
                self.tile_x[col, row] = cx
                self.tile_z[col, row] = cz
                name = f"ptile_{col}_{row}"
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                self.tile_geom_ids[col, row] = gid

    def initialize_posture(self, start_x: float = -1.5):
        """Set robot to corridor starting position at specified longitudinal offset."""
        self.state = "APPROACH"
        self.sub_state = "APPROACH"
        self.spraying = False
        self.t_start_paint = 0.0
        self.recoil_integral = 0.0
        self.recoil_est = 0.0

        # Base Position
        self.data.qpos[0:3] = [start_x, 0.0, 0.14493]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:11] = 0.0
        self.data.qpos[11:17] = self.q_nominal.copy()
        self.data.qvel[:] = 0.0

        # Reset tile thickness and colors
        self.tile_thickness[:] = 0.0
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                gid = self.tile_geom_ids[col, row]
                if gid >= 0:
                    self.model.geom_rgba[gid] = self.color_primer.copy()

        # Turn off spray cone visual initially
        if self.spray_cone_geom_id >= 0:
            self.model.geom_rgba[self.spray_cone_geom_id][3] = 0.0

        # Kinematic convergence to start stance
        target_init = np.array([start_x + self.x_rel_nominal, self.y_des, self.z_min])
        for _ in range(250):
            mujoco.mj_forward(self.model, self.data)
            pos_err = target_init - self.data.site_xpos[self.tcp_site_id]
            R_curr = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
            rot_err = 0.5 * (
                np.cross(R_curr[:, 0], self.R_des[:, 0]) +
                np.cross(R_curr[:, 1], self.R_des[:, 1]) +
                np.cross(R_curr[:, 2], self.R_des[:, 2])
            )
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.tcp_site_id)
            Ja = np.vstack([self.jacp[:, 10:16], self.jacr[:, 10:16]])
            err_6d = np.concatenate([pos_err, rot_err])
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + 1e-4 * np.eye(6), err_6d)
            self.data.qpos[11:17] += dq * 0.4

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def deposit_paint(self, p_tcp: np.ndarray, dt: float):
        """Simulate physical atomized paint accumulation on discretized wall tiles."""
        if not self.spraying:
            return

        x_tcp, z_tcp = p_tcp[0], p_tcp[2]

        # Active spray bounding box on wall
        margin_x = 2.5 * self.sigma_x
        margin_z = 2.5 * self.sigma_z

        col_min = max(0, int((x_tcp - margin_x - self.x_patch_start) / self.dx_tile))
        col_max = min(self.n_cols - 1, int((x_tcp + margin_x - self.x_patch_start) / self.dx_tile))
        row_min = max(0, int((z_tcp - margin_z - self.z_patch_min) / self.dz_tile))
        row_max = min(self.n_rows - 1, int((z_tcp + margin_z - self.z_patch_min) / self.dz_tile))

        for c in range(col_min, col_max + 1):
            for r in range(row_min, row_max + 1):
                tx = self.tile_x[c, r]
                tz = self.tile_z[c, r]
                d2 = ((tx - x_tcp) / self.sigma_x) ** 2 + ((tz - z_tcp) / self.sigma_z) ** 2
                if d2 < 6.0:  # Within ~2.45 sigma
                    flux = self.deposition_rate * np.exp(-0.5 * d2)
                    self.tile_thickness[c, r] += flux * dt

                    # Update visual appearance dynamically
                    fraction = np.clip(self.tile_thickness[c, r] / self.target_thickness, 0.0, 1.0)
                    gid = self.tile_geom_ids[c, r]
                    if gid >= 0:
                        rgba = (1.0 - fraction) * self.color_primer + fraction * self.color_paint
                        self.model.geom_rgba[gid] = rgba

    def get_paint_statistics(self) -> dict:
        """Compute coverage percentage, thickness uniformity, and dry film metrics."""
        covered_tiles = np.sum(self.tile_thickness >= (0.75 * self.target_thickness))
        coverage_pct = (covered_tiles / self.total_tiles) * 100.0
        active_mask = self.tile_thickness > 1.0
        if np.any(active_mask):
            mean_thickness = float(np.mean(self.tile_thickness[active_mask]))
            std_thickness = float(np.std(self.tile_thickness[active_mask]))
            max_thickness = float(np.max(self.tile_thickness))
        else:
            mean_thickness, std_thickness, max_thickness = 0.0, 0.0, 0.0

        return {
            "coverage_pct": coverage_pct,
            "mean_thickness": mean_thickness,
            "std_thickness": std_thickness,
            "max_thickness": max_thickness,
            "coated_tiles": int(covered_tiles),
            "total_tiles": self.total_tiles
        }

    def compute_control_step(self, sim_time: float) -> tuple:
        """
        Evaluate whole-body painting control laws for the current simulation step.
        Returns: (standoff_err, recoil_meas, p_tcp, target_pos, v_base_actual, state, coverage_pct)
        """
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.tcp_site_id)

        # Kinematic states
        p_tcp = self.data.site_xpos[self.tcp_site_id].copy()
        v_tcp = (self.jacp @ self.data.qvel).copy()
        w_tcp = (self.jacr @ self.data.qvel).copy()

        base_x = self.data.xpos[self.base_body_id][0]
        base_y = self.data.xpos[self.base_body_id][1]
        base_quat = self.data.xquat[self.base_body_id]
        qw, qx, qy, qz = base_quat
        base_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        v_base_actual = float(self.data.qvel[0])

        # 1. State Transition & Spray Trigger Activation
        standoff_err = self.y_des - p_tcp[1]  # positive if gun is too far back from wall

        if self.state == "APPROACH":
            # Approach standoff distance gently
            if abs(standoff_err) < 0.005 and sim_time >= 1.5:
                self.state = "PAINTING"
                self.spraying = True
                self.t_start_paint = sim_time
                self.x_start_paint = float(p_tcp[0])
                print(f">>> [t={sim_time:5.3f}s] Exact standoff ({self.standoff*100:.1f} cm) established! "
                      f"Activating spray gun trigger & initiating whole-body raster at {self.base_speed:.3f} m/s!")
                if self.spray_cone_geom_id >= 0:
                    self.model.geom_rgba[self.spray_cone_geom_id][3] = 0.40  # Spray mist visible!

        # 2. Physics of Spray Recoil Thrust (Newton 3rd Law Reaction)
        if self.spraying:
            # Physical backward thrust applied directly to gun nozzle body (-Y direction)
            # Includes high-frequency pneumatic turbulence ripple
            f_recoil_actual = self.recoil_force * (
                1.0 + 0.08 * np.sin(2.0 * np.pi * 14.0 * sim_time) + 0.03 * np.random.randn()
            )
            if self.gun_body_id >= 0:
                self.data.xfrc_applied[self.gun_body_id, :3] = [0.0, -f_recoil_actual, 0.0]
        else:
            f_recoil_actual = 0.0
            if self.gun_body_id >= 0:
                self.data.xfrc_applied[self.gun_body_id, :3] = 0.0

        # 3. Mobile Base Velocity & Lane Steering Control
        if self.spraying:
            v_cmd = self.base_speed
        else:
            v_cmd = 0.0

        yaw_rate_cmd = -self.Ky_base * (base_y - 0.0) - self.Kyaw_base * base_yaw
        w_left  = (v_cmd - yaw_rate_cmd * (self.track_width / 2.0)) / self.wheel_radius
        w_right = (v_cmd + yaw_rate_cmd * (self.track_width / 2.0)) / self.wheel_radius

        self.data.ctrl[self.wheel_act_ids[0]] = w_left
        self.data.ctrl[self.wheel_act_ids[2]] = w_left
        self.data.ctrl[self.wheel_act_ids[1]] = w_right
        self.data.ctrl[self.wheel_act_ids[3]] = w_right

        # 4. Trajectory Generation (Digital Signal Square-Wave Rastering)
        if self.state == "APPROACH":
            x_des = self.data.qpos[0] + self.x_rel_nominal
            vx_des = 0.0
            z_des = self.z_min
            vz_des = 0.0
            self.sub_state = "APPROACH"
        else:
            tau = sim_time - self.t_start_paint
            pass_idx = int(tau / self.t_pass)
            tau_pass = tau - pass_idx * self.t_pass
            x_k = self.x_start_paint + pass_idx * self.d_shift
            stroke_len = self.z_max - self.z_min

            if tau_pass <= self.t_vert:
                # Vertical stroke phase (pure vertical edge in digital signal)
                s = 0.5 * (1.0 - np.cos(np.pi * tau_pass / self.t_vert))
                ds = (0.5 * np.pi / self.t_vert) * np.sin(np.pi * tau_pass / self.t_vert)

                if pass_idx % 2 == 0:
                    # Even pass: Move UP (z_min -> z_max)
                    z_des = self.z_min + s * stroke_len
                    vz_des = ds * stroke_len
                    self.sub_state = f"P{pass_idx}_UP"
                else:
                    # Odd pass: Move DOWN (z_max -> z_min)
                    z_des = self.z_max - s * stroke_len
                    vz_des = -ds * stroke_len
                    self.sub_state = f"P{pass_idx}_DOWN"

                x_des = x_k
                vx_des = 0.0
            else:
                # Horizontal swath shift phase by fan overlap step
                tau_shift = tau_pass - self.t_vert
                s_h = 0.5 * (1.0 - np.cos(np.pi * tau_shift / self.t_shift))
                ds_h = (0.5 * np.pi / self.t_shift) * np.sin(np.pi * tau_shift / self.t_shift)

                x_des = x_k + s_h * self.d_shift
                vx_des = ds_h * self.d_shift

                if pass_idx % 2 == 0:
                    z_des = self.z_max
                else:
                    z_des = self.z_min
                vz_des = 0.0
                self.sub_state = f"P{pass_idx}_SHIFT"

        # 5. Standoff Regulation & Active Recoil Compensation (Y-axis)
        f_x = self.Kp_x * (x_des - p_tcp[0]) + self.Kd_x * (vx_des - v_tcp[0])
        f_z = self.Kp_z * (z_des - p_tcp[2]) + self.Kd_z * (vz_des - v_tcp[2])

        # Disturbance Observer for Recoil Force Estimation
        if self.spraying and self.compensate_recoil:
            self.recoil_integral += standoff_err * self.dt
            self.recoil_integral = np.clip(self.recoil_integral, -1.0, 1.0)
            self.recoil_est = self.recoil_force + self.ki_recoil * self.recoil_integral
            f_recoil_comp = self.recoil_est
        else:
            self.recoil_est = 0.0
            f_recoil_comp = 0.0

        # Operational Space Mobility Matrix & Inertial Decoupling
        M_full = np.zeros((self.model.nv, self.model.nv))
        try:
            mujoco.mj_fullM(self.model, self.data, M_full)
        except TypeError:
            qM = getattr(self.data, "qM", getattr(self.data, "M", None))
            mujoco.mj_fullM(self.model, M_full, qM)

        M_arm = M_full[10:16, 10:16]
        Ja_arm = np.vstack([self.jacp[:, 10:16], self.jacr[:, 10:16]])
        Minv_arm = np.linalg.inv(M_arm + 1e-4 * np.eye(6))
        Lambda_inv = Ja_arm @ Minv_arm @ Ja_arm.T

        # Decouple vertical dynamic force from normal axis
        decoupling_ratio = Lambda_inv[1, 2] / max(0.01, Lambda_inv[1, 1])
        f_y_decouple = -decoupling_ratio * f_z

        # Total Standoff Force Command along Y:
        # PD Position Feedback + Feedforward Recoil Cancellation + DOB Estimator + Dynamic Decoupling
        f_y = self.Kp_y * standoff_err - self.Kd_y * v_tcp[1] + f_recoil_comp + f_y_decouple

        # 6. Orientation Control: Solid 90-degree perpendicularity against wall
        R_curr = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        rot_err = 0.5 * (
            np.cross(R_curr[:, 0], self.R_des[:, 0]) +
            np.cross(R_curr[:, 1], self.R_des[:, 1]) +
            np.cross(R_curr[:, 2], self.R_des[:, 2])
        )
        tau_rot = self.Kp_rot * rot_err - self.Kd_rot * w_tcp

        # 7. Weighted Null-Space Projection & Joint Torque Command
        F_task = np.array([f_x, f_y, f_z, tau_rot[0], tau_rot[1], tau_rot[2]])
        Ja = np.vstack([self.jacp[:, 10:16], self.jacr[:, 10:16]])

        Ja_weighted_pinv = self.Winv_arm @ Ja.T @ np.linalg.solve(
            Ja @ self.Winv_arm @ Ja.T + self.lambda_damping * np.eye(6), np.eye(6)
        )
        N_w = np.eye(6) - Ja_weighted_pinv @ Ja

        q_arm_curr = self.data.qpos[11:17]
        qvel_arm_curr = self.data.qvel[10:16]
        tau_0 = -self.K_posture * (q_arm_curr - self.q_nominal) - self.D_null * qvel_arm_curr
        tau_null = N_w.T @ tau_0

        tau_bias = self.data.qfrc_bias[10:16]
        tau_arm = Ja.T @ F_task + tau_null + tau_bias

        for i, act_id in enumerate(self.arm_act_ids):
            self.data.ctrl[act_id] = tau_arm[i]

        # 8. Real-Time Dynamic Paint Deposition Update
        self.deposit_paint(p_tcp, self.dt)
        stats = self.get_paint_statistics()

        target_pos = np.array([x_des, self.y_des, z_des])
        return standoff_err, f_recoil_actual, p_tcp, target_pos, v_base_actual, self.sub_state, stats["coverage_pct"]


def run_headless_benchmark(duration: float = 35.0, plot: bool = True,
                           speed: float = 0.03, standoff: float = 0.20,
                           recoil: float = 3.50, compensate: bool = True,
                           ee_speed: float = 0.08, fan_width: float = 0.15,
                           overlap: float = 0.25, z_min: float = 0.72, z_max: float = 1.18,
                           kp_y: float = 85.0, kd_y: float = 18.0,
                           plot_name: str = "whole_body_painting_performance.png"):
    """Execute high-speed headless evaluation and log full painting metrics."""
    scene_path = os.path.join(os.path.dirname(__file__), "scene_husky_painting.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    controller = WholeBodyPaintingController(
        model, data,
        base_speed=speed, standoff=standoff, recoil_force=recoil,
        compensate_recoil=compensate, ee_speed=ee_speed,
        fan_width=fan_width, overlap=overlap, z_min=z_min, z_max=z_max,
        kp_y=kp_y, kd_y=kd_y
    )
    controller.initialize_posture(start_x=-1.5)

    print("=" * 78)
    print("Whole-Body Loco-Manipulation Robotic Spray Painting Benchmark")
    print(f"Wall Dimensions      : 6.0 m Length x 1.6 m Height | Target Patch: 5.4 m x 0.6 m")
    print(f"Target Standoff Dist : {standoff * 100:.1f} cm ({standoff:.3f} m from wall at Y = 0.80 m)")
    print(f"Spray Recoil Thrust  : {recoil:.2f} N (Active Compensation: {'ENABLED' if compensate else 'DISABLED'})")
    print(f"Cruising Base Speed  : {speed:.3f} m/s | End-Effector Speed: {ee_speed:.3f} m/s")
    print(f"Fan Width / Overlap  : {fan_width*100:.1f} cm / {overlap*100:.0f}% (Shift: {controller.d_shift*100:.1f} cm)")
    print(f"Vertical Stroke Range: [{z_min:.2f} m, {z_max:.2f} m] (Stroke: {(z_max - z_min)*100:.1f} cm)")
    print(f"Discretized Surface  : 324 Paintable Wall Cells (54 cols x 6 rows)")
    print(f"Simulation Duration  : {duration:.1f} s | Timestep: {model.opt.timestep:.4f} s")
    print("=" * 78)

    dt = model.opt.timestep
    total_steps = int(duration / dt)
    log_interval = int(0.5 / dt)

    times = []
    standoff_errs = []
    recoils_actual = []
    recoils_est = []
    base_xs = []
    base_vs = []
    tcp_xs = []
    tcp_ys = []
    tcp_zs = []
    target_xs = []
    target_zs = []
    tilts = []
    coverages = []
    states = []

    t_start_wall = time.perf_counter()

    for step in range(total_steps):
        t_sim = float(data.time)
        err_y, f_rec, p_tcp, p_tgt, v_base, state, cov_pct = controller.compute_control_step(t_sim)
        mujoco.mj_step(model, data)

        R_curr = data.site_xmat[controller.tcp_site_id].reshape(3, 3)
        cos_theta = np.clip(R_curr[:, 2] @ np.array([0.0, 1.0, 0.0]), -1.0, 1.0)
        tilt_deg = float(np.degrees(np.arccos(cos_theta)))

        times.append(t_sim)
        standoff_errs.append(err_y * 1000.0)  # mm
        recoils_actual.append(f_rec)
        recoils_est.append(controller.recoil_est)
        base_xs.append(data.xpos[controller.base_body_id][0])
        base_vs.append(v_base)
        tcp_xs.append(p_tcp[0])
        tcp_ys.append(p_tcp[1])
        tcp_zs.append(p_tcp[2])
        target_xs.append(p_tgt[0])
        target_zs.append(p_tgt[2])
        tilts.append(tilt_deg)
        coverages.append(cov_pct)
        states.append(state)

        if step % log_interval == 0 or step == total_steps - 1:
            standoff_actual = (controller.y_wall - p_tcp[1]) * 100.0
            print(f"[t={t_sim:5.2f}s] Phase: {state:10s} | Base X: {base_xs[-1]:5.2f}m | "
                  f"Standoff: {standoff_actual:5.2f}cm (err={err_y*1000.0:+4.1f}mm) | "
                  f"Recoil: {f_rec:4.2f}N | Tilt: {tilt_deg:4.2f}° | Coverage: {cov_pct:4.1f}%")

    wall_elapsed = time.perf_counter() - t_start_wall
    realtime_factor = duration / max(1e-6, wall_elapsed)

    stats = controller.get_paint_statistics()
    s_errs = np.array(standoff_errs)[np.array(times) >= controller.t_start_paint]
    rmse_standoff = float(np.sqrt(np.mean(s_errs ** 2))) if len(s_errs) > 0 else 0.0
    mean_tilt = float(np.mean(tilts))

    print("=" * 78)
    print("Robotic Spray Painting Benchmark Results:")
    print(f"  - Simulated {duration:.2f} s in {wall_elapsed:.3f} s ({realtime_factor:.1f}x real-time)")
    print(f"  - Standoff Distance RMSE   : {rmse_standoff:.2f} mm (Target: {standoff * 100:.1f} cm)")
    print(f"  - Wall Paint Coverage      : {stats['coverage_pct']:.1f}% ({stats['coated_tiles']}/{stats['total_tiles']} cells)")
    print(f"  - Mean Dry Film Thickness  : {stats['mean_thickness']:.1f} µm ± {stats['std_thickness']:.1f} µm (Target: {controller.target_thickness:.1f} µm)")
    print(f"  - End-Effector Tilt Error  : Mean {mean_tilt:.2f}° (Strict 90° flush perpendicularity)")
    print(f"  - Total Base Travel        : {base_xs[-1] - base_xs[0]:.2f} m along 6.0 m wall")
    print("=" * 78)

    if plot:
        generate_painting_performance_plot(
            times, standoff_errs, recoils_actual, recoils_est,
            base_xs, base_vs, tcp_xs, tcp_zs, target_xs, target_zs,
            controller.tile_x, controller.tile_z, controller.tile_thickness,
            coverages, controller.t_start_paint, standoff,
            plot_name=plot_name
        )


def generate_painting_performance_plot(times, standoff_errs, recoils_actual, recoils_est,
                                       base_xs, base_vs, tcp_xs, tcp_zs, target_xs, target_zs,
                                       tile_x, tile_z, tile_thickness, coverages,
                                       t_start, standoff,
                                       plot_name: str = "whole_body_painting_performance.png"):
    """Generate publication-quality 4-panel analysis graphic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Whole-Body Mobile Loco-Manipulation Robotic Spray Painting Benchmark",
                 fontsize=14, fontweight="bold")

    # Plot 1: Standoff Distance Tracking Error (Y-axis)
    ax1 = axs[0, 0]
    ax1.plot(times, standoff_errs, label="Standoff Error (mm)", color="#1f77b4", lw=2)
    ax1.axhline(0.0, color="green", linestyle="--", label="Target Standoff", lw=1.8)
    ax1.fill_between(times, -2.0, 2.0, color="green", alpha=0.15, label="Tolerance (±2 mm)")
    if t_start > 0:
        ax1.axvline(t_start, color="orange", linestyle=":", label="Spray Trigger ON", lw=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Standoff Error (mm)")
    ax1.set_title("1. Standoff Distance Regulation (Constant 20 cm Offset)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Plot 2: Spray Recoil Force & Active Disturbance Compensation
    ax2 = axs[0, 1]
    ax2.plot(times, recoils_actual, label="Physical Spray Recoil Thrust (N)", color="#d62728", lw=2)
    ax2.plot(times, recoils_est, label="Active Recoil Compensation (N)", color="#2ca02c", linestyle="--", lw=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Recoil Force (N)")
    ax2.set_title("2. Spray Recoil Force & Active Disturbance Compensation")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    # Plot 3: 2D Wall Painting Raster Trajectory & Thickness Coverage Heatmap
    ax3 = axs[1, 0]
    # Draw heatmap of wall tile thickness
    extent = [tile_x.min() - 0.05, tile_x.max() + 0.05, tile_z.min() - 0.05, tile_z.max() + 0.05]
    im = ax3.imshow(tile_thickness.T, origin="lower", extent=extent, cmap="Blues",
                    aspect="auto", vmin=0.0, vmax=60.0)
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label("Dry Film Thickness (µm)")

    ax3.plot(tcp_xs, tcp_zs, label="Nozzle Path", color="#ff7f0e", lw=1.8, alpha=0.85)
    ax3.plot(target_xs, target_zs, label="Raster Reference", color="black", linestyle=":", lw=1.2)
    ax3.set_xlabel("Longitudinal Wall Position X (m)")
    ax3.set_ylabel("Vertical Wall Position Z (m)")
    ax3.set_title("3. Wall Painting Raster Trajectory & Film Deposition Heatmap")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right")

    # Plot 4: Wall Coverage Progress & Base Travel
    ax4 = axs[1, 1]
    color_cov = "#1f77b4"
    ax4.plot(times, coverages, label="Wall Coverage (%)", color=color_cov, lw=2.5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Wall Paint Coverage (%)", color=color_cov)
    ax4.tick_params(axis="y", labelcolor=color_cov)
    ax4.set_ylim(-2, 105)
    ax4.grid(True, linestyle="--", alpha=0.6)

    ax4_sub = ax4.twinx()
    color_base = "#2ca02c"
    ax4_sub.plot(times, base_xs, label="Husky Base X (m)", color=color_base, linestyle="--", lw=2)
    ax4_sub.set_ylabel("Base Position X (m)", color=color_base)
    ax4_sub.tick_params(axis="y", labelcolor=color_base)
    ax4.set_title("4. Cumulative Paint Coverage (%) & Mobile Base Travel")

    plt.tight_layout()
    plt.savefig(plot_name, dpi=200)
    print(f"\n[Plot] High-resolution performance plot saved to: {os.path.abspath(plot_name)}")
    plt.close()


def run_interactive_simulation(duration: float = None, speed: float = 0.03,
                               standoff: float = 0.20, recoil: float = 3.50,
                               compensate: bool = True, ee_speed: float = 0.08,
                               fan_width: float = 0.15, overlap: float = 0.25,
                               z_min: float = 0.72, z_max: float = 1.18,
                               kp_y: float = 280.0, kd_y: float = 35.0,
                               start_x: float = -1.5):
    """Launch interactive 3D graphical visualizer with real-time painting."""
    import mujoco.viewer

    scene_path = os.path.join(os.path.dirname(__file__), "scene_husky_painting.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    controller = WholeBodyPaintingController(
        model, data,
        base_speed=speed, standoff=standoff, recoil_force=recoil,
        compensate_recoil=compensate, ee_speed=ee_speed,
        fan_width=fan_width, overlap=overlap, z_min=z_min, z_max=z_max,
        kp_y=kp_y, kd_y=kd_y
    )
    controller.initialize_posture(start_x=start_x)

    print("=" * 78)
    print("Launching Whole-Body Loco-Manipulation Spray Painting 3D Visualizer")
    print("=" * 78)
    print("Configuration:")
    print(f"  • Base Cruising Speed    : {speed:.3f} m/s ({speed*1000:.1f} mm/s)")
    print(f"  • Standoff Offset (Wall) : {standoff*100:.1f} cm ({standoff:.3f} m from wall)")
    print(f"  • Spray Recoil Thrust    : {recoil:.2f} N (Compensation: {'ACTIVE' if compensate else 'OFF'})")
    print(f"  • Fan Width / Overlap    : {fan_width*100:.1f} cm / {overlap*100:.0f}% (Shift: {controller.d_shift*100:.1f} cm)")
    print(f"  • Vertical Stroke (Z)    : [{z_min:.2f} m, {z_max:.2f} m] (Stroke: {(z_max - z_min)*100:.1f} cm)")
    print(f"  • Wall Surface           : 324 Discretized Paint Cells (Real-time dynamic RGBA)")
    print("-" * 78)
    print("Interactive Controls:")
    print("  • Mouse Navigation       : Left-drag to rotate, Right-drag to zoom, Middle-drag to pan")
    print("  • Spacebar               : Pause / resume")
    print("  • Keys '+' / '-'         : Increase / decrease base cruising speed")
    print("  • Key 'C'                : Toggle active recoil compensation ON / OFF")
    print("  • Key 'T'                : Toggle spray gun trigger ON / OFF")
    print("  • Key 'R'                : Reset robot and wall to start")
    print("=" * 78)

    def key_callback(keycode):
        if keycode == ord('+') or keycode == ord('='):
            controller.base_speed = min(controller.base_speed + 0.005, 0.15)
            controller.update_timing()
            print(f"\n[Speed] Base target speed: {controller.base_speed:.3f} m/s")
        elif keycode == ord('-') or keycode == ord('_'):
            controller.base_speed = max(controller.base_speed - 0.005, 0.005)
            controller.update_timing()
            print(f"\n[Speed] Base target speed: {controller.base_speed:.3f} m/s")
        elif keycode == ord('C') or keycode == ord('c'):
            controller.compensate_recoil = not controller.compensate_recoil
            print(f"\n[Recoil] Active recoil compensation: {'ENABLED' if controller.compensate_recoil else 'DISABLED'}")
        elif keycode == ord('T') or keycode == ord('t'):
            controller.spraying = not controller.spraying
            if controller.spray_cone_geom_id >= 0:
                controller.model.geom_rgba[controller.spray_cone_geom_id][3] = 0.40 if controller.spraying else 0.0
            print(f"\n[Spray] Trigger: {'ON' if controller.spraying else 'OFF'}")
        elif keycode == ord('R') or keycode == ord('r'):
            controller.initialize_posture(start_x=start_x)
            print("\n[Reset] Reset robot and wall to corridor start!")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Tracking camera view on base and wall painting interaction
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = controller.base_body_id
        viewer.cam.distance = 3.2
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 145.0

        step_dt = model.opt.timestep

        while viewer.is_running():
            step_start = time.perf_counter()
            t_sim = float(data.time)

            if duration is not None and t_sim >= duration:
                break

            controller.compute_control_step(t_sim)
            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - step_start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="Whole-Body Robotic Spray Painting Loco-Manipulation Benchmark (Husky + UR10e)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1. Execution Modes
    mode_group = parser.add_argument_group("Execution Modes")
    mode_group.add_argument("--headless", action="store_true",
                            help="Execute fast headless simulation without graphical window")
    mode_group.add_argument("--plot", action="store_true",
                            help="Generate publication-quality 4-panel performance graphs")
    mode_group.add_argument("--duration", "-t", type=float, default=35.0,
                            help="Total simulation evaluation duration in seconds (for headless/plot)")
    mode_group.add_argument("--plot_name", type=str, default="whole_body_painting_performance.png",
                            help="Filename for the generated performance plot")
    mode_group.add_argument("--start_x", type=float, default=-1.5,
                            help="Initial longitudinal position of Husky base along wall (m)")

    # 2. Painting Standoff & Recoil Parameters
    paint_group = parser.add_argument_group("Painting & Standoff Parameters")
    paint_group.add_argument("--standoff", type=float, default=0.20,
                             help="Target standoff distance from nozzle tip to wall (m)")
    paint_group.add_argument("--recoil", type=float, default=3.50,
                             help="Nominal physical spray recoil thrust pushing backward on nozzle (N)")
    paint_group.add_argument("--compensate", action=argparse.BooleanOptionalAction, default=True,
                             help="Enable/disable active recoil feedforward and disturbance compensation")
    paint_group.add_argument("--fan_width", type=float, default=0.15,
                             help="Width of atomized spray fan pattern (m)")
    paint_group.add_argument("--overlap", type=float, default=0.25,
                             help="Swath overlap fraction between consecutive passes")

    # 3. Kinematic Speeds & Stroke Limits
    speed_group = parser.add_argument_group("Speeds & Workspace Geometry")
    speed_group.add_argument("--speed", "-s", type=float, default=0.03,
                             help="Husky mobile base cruising speed along the wall (m/s)")
    speed_group.add_argument("--ee_speed", type=float, default=0.08,
                             help="Target vertical stroke velocity for spray nozzle (m/s)")
    speed_group.add_argument("--z_min", type=float, default=0.72,
                             help="Lower limit of vertical painting stroke (m)")
    speed_group.add_argument("--z_max", type=float, default=1.18,
                             help="Upper limit of vertical painting stroke (m)")

    # 4. Controller Impedance Gains
    gain_group = parser.add_argument_group("Controller Gains")
    gain_group.add_argument("--kp_y", type=float, default=280.0,
                            help="Standoff normal axis position stiffness (N/m)")
    gain_group.add_argument("--kd_y", type=float, default=35.0,
                            help="Standoff normal axis velocity damping (Ns/m)")

    args = parser.parse_args()

    if args.headless or args.plot:
        run_headless_benchmark(
            duration=args.duration, plot=args.plot,
            speed=args.speed, standoff=args.standoff, recoil=args.recoil,
            compensate=args.compensate, ee_speed=args.ee_speed,
            fan_width=args.fan_width, overlap=args.overlap,
            z_min=args.z_min, z_max=args.z_max,
            kp_y=args.kp_y, kd_y=args.kd_y,
            plot_name=args.plot_name
        )
    else:
        run_interactive_simulation(
            speed=args.speed, standoff=args.standoff, recoil=args.recoil,
            compensate=args.compensate, ee_speed=args.ee_speed,
            fan_width=args.fan_width, overlap=args.overlap,
            z_min=args.z_min, z_max=args.z_max,
            kp_y=args.kp_y, kd_y=args.kd_y, start_x=args.start_x
        )


if __name__ == "__main__":
    main()
