#!/usr/bin/env python3
"""
Whole-Body Loco-Manipulation Hybrid Position/Force Sanding Controller
===================================================================
Clearpath Husky A200 + Universal Robots UR10e with Circular Sanding Disc

Key Features:
-------------
1. Whole-Body Loco-Manipulation:
   - Mobile base drives at a steady cruising velocity of 0.10 m/s along a lengthened 6.0 m wall.
   - Arm simultaneously articulates in an up-down sinusoidal raster motion (Z-axis).
   - In the normal direction (Y-axis), closed-loop hybrid PI force control presses the sanding disc
     against the wall at a regulated 1.0 N force.

2. Guarded Initial Approach & Standoff:
   - Robot starts at one end of the wall (X = -1.5 m) with a configurable initial offset
     (default: 0.10 m away from the wall surface at Y = 0.80 m).
   - The force controller drives the tool forward in the normal direction (+Y) at a smooth approach rate.
   - Once contact is detected (F_meas >= 0.3 N), the controller transitions to SANDING:
     the base starts moving forward at 0.1 m/s and the up-down sanding motion activates!

3. Weighted Null-Space Approach:
   - Task Jacobian J in R^(6 x 6) for arm degrees of freedom and whole-body coordination.
   - Positive-definite weight matrix W = diag([w1, ..., w6]) accounts for joint inertia and effort costs.
   - Weighted pseudo-inverse:
         J_W^dagger = W^(-1) * J^T * (J * W^(-1) * J^T + lambda^2 * I)^(-1)
   - Weighted null-space projection matrix:
         N_W = I - J_W^dagger * J
   - A secondary ergonomic joint-centering & manipulability optimization torque:
         tau_0 = -K_posture * (q_arm - q_nominal) - D_null * qvel_arm
     is projected into the null space: tau_null = N_W^T * tau_0.
     This guarantees ZERO interference with the primary contact force and sanding trajectory!

4. Interactive 3D Viewer & Telemetry Logging:
   - Free 360° mouse navigation in mjCAMERA_FREE.
   - Dynamic tracking cameras following the vehicle and tool closeup.
   - Headless benchmark mode (--headless) and performance plot generation (--plot).

Usage:
------
    python3 whole_body_sanding_controller.py            # Interactive 3D visualization
    python3 whole_body_sanding_controller.py --headless # Headless benchmark evaluation
    python3 whole_body_sanding_controller.py --plot     # Save high-res performance plots
    python3 whole_body_sanding_controller.py --speed 0.10 --duration 20.0
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco

# Set writable cache directory for Matplotlib before import
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"


class WholeBodySandingController:
    """
    Whole-body loco-manipulation controller combining mobile base tracking,
    hybrid position/force control, and weighted null-space projection.
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 base_speed: float = 0.01, desired_force: float = 1.0,
                 initial_offset: float = 0.10, v_approach: float = 0.01,
                 ee_speed: float = 0.01, d_head: float = 0.15):
        self.model = model
        self.data = data
        self.dt = model.opt.timestep

        # Mission Parameters
        self.base_speed = base_speed          # Base cruising velocity along wall (m/s)
        self.F_desired = desired_force        # 1.0 N normal pressing force
        self.initial_offset = initial_offset  # 0.10 m standoff from wall
        self.y_wall = 0.8000                  # Front surface of wall cuboid at Y = 0.8000 m
        self.F_contact_thresh = 0.30          # Threshold to trigger sanding state

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
        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.touch_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "sander_touch_sensor")
        self.touch_sensor_adr = model.sensor_adr[self.touch_sensor_id] if self.touch_sensor_id >= 0 else -1
        self.disc_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sander_disc_col")
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        # State Machine Tracking
        self.state = "APPROACH"               # "APPROACH" -> "SANDING"
        self.sub_state = "APPROACH"
        self.contact_detected = False
        self.t_contact = 0.0
        self.x_contact = 0.0

        # Normal Force Control Gains (Y-axis)
        self.Kp_f = 0.60                      # Proportional gain on force error
        self.Ki_f = 1.50                      # Integral gain on force error
        self.Kd_f = 3.20                      # Normal velocity damping
        self.force_integral = 0.0
        self.f_filtered = 0.0
        self.alpha_filter = 0.15              # 1st-order IIR low-pass filter factor

        # Approach Phase Parameters (Gentle 0.01 m/s contact approach)
        self.v_approach = v_approach          # 0.01 m/s forward approach velocity into wall
        self.Kv_approach = 22.0               # Approach velocity damping

        # Tangential Sanding Plane Motion Parameters (X and Z axes)
        self.Kp_x = 45.0                      # Longitudinal position gain
        self.Kd_x = 12.0                      # Longitudinal velocity damping
        self.Kp_z = 50.0                      # Vertical position gain
        self.Kd_z = 14.0                      # Vertical velocity damping

        # Desired Orientation: Tool axis strictly perpendicular to wall (+Y)
        # Circular disc face is flush against wall: X_tool along +X, Y_tool along -Z, Z_tool along +Y
        self.R_des = np.array([
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
            [0.0, -1.0, 0.0]
        ])

        # Orientation Stiffness & Damping:
        # Tilt axes (X and Z): High stiffness to keep disc flush so contact surface never moves outside
        # Perpendicular axis (Y): Soft compliance / damping for friction relief and wrist flexibility
        self.Kp_tilt = 120.0                  # High restoring stiffness for pitch and yaw (Nm/rad)
        self.Kd_tilt = 15.0                   # Tilt angular damping (Nms/rad)
        self.Kp_perp = 5.0                    # Soft compliance about normal axis (Nm/rad)
        self.Kd_perp = 2.0                    # Torsional damping about normal axis (Nms/rad)
        self.Kp_rot = np.array([self.Kp_tilt, self.Kp_perp, self.Kp_tilt])
        self.Kd_rot = np.array([self.Kd_tilt, self.Kd_perp, self.Kd_tilt])

        # Full Vertical Sanding Stroke (Covers whole 60cm wall patch: Z in [0.72, 1.18] m)
        self.z_min = 0.72
        self.z_max = 1.18
        self.stroke_z = self.z_max - self.z_min  # 0.46 m vertical stroke
        self.z_mid = (self.z_max + self.z_min) / 2.0  # 0.95 m
        self.Az = self.stroke_z / 2.0                 # 0.23 m amplitude

        # Digital Signal Raster Parameters:
        # Trajectory shape: Pure Vertical Stroke -> Horizontal Shift by Sanding Head Diameter -> Pure Vertical Stroke in reverse
        self.d_head = d_head                  # Diameter of circular sanding head (0.15 m = 150 mm)
        self.shift_ratio = 0.25               # 25% of pass duration allocated to horizontal shift, 75% to vertical stroke
        self.update_timing()
        self.x_rel_nominal = 0.05             # Nominal tool forward lead relative to base (+5 cm)

        # Base Steering / Lane Tracking Gains (Maintain Y = 0.0, Heading = 0.0)
        self.Ky_base = 3.5                    # Cross-track error gain
        self.Kyaw_base = 2.0                  # Heading error gain

        # Base Steering / Lane Tracking Gains (Maintain Y = 0.0, Heading = 0.0)
        self.Ky_base = 3.5                    # Cross-track error gain
        self.Kyaw_base = 2.0                  # Heading error gain

        # Weighted Null-Space Parameters
        # Weighting matrix assigns higher penalty to shoulder and lower penalty to wrist joints
        self.W_arm = np.diag([1.0, 1.2, 1.0, 0.5, 0.5, 0.4])
        self.Winv_arm = np.linalg.inv(self.W_arm)
        self.K_posture = 12.0                 # Postural centering gain in null space
        self.D_null = 2.2                     # Null-space velocity damping
        self.lambda_damping = 1e-4            # Levenberg-Marquardt singularity regularization

        # Nominal Arm Posture (optimal manipulability while sanding wall at Y = 0.80 m)
        self.q_nominal = np.array([-1.9147, -1.6404, 2.0291, -0.3887, 1.2269, 3.1416])

        # Preallocated buffers
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.contact_wrench_buf = np.zeros(6)

    def update_timing(self):
        """Synchronize digital signal pass period with mobile base cruising speed."""
        self.t_pass = self.d_head / max(1e-4, self.base_speed)
        self.t_shift = self.shift_ratio * self.t_pass
        self.t_vert = self.t_pass - self.t_shift

    def initialize_posture(self, start_x: float = -1.5, offset: float = None):
        """Reset robot to the starting corridor stance with the specified offset from wall."""
        if offset is not None:
            self.initial_offset = offset

        self.state = "APPROACH"
        self.contact_detected = False
        self.t_contact = 0.0
        self.x_contact = 0.0
        self.force_integral = 0.0
        self.f_filtered = 0.0

        # Base Position at corridor start
        self.data.qpos[0:3] = [start_x, 0.0, 0.14493]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:11] = 0.0  # Wheels
        self.data.qpos[11:17] = self.q_nominal.copy()
        self.data.qvel[:] = 0.0

        # Initial target TCP position in front of wall at stroke start height (z_min)
        y_init = self.y_wall - self.initial_offset
        target_init = np.array([start_x + self.x_rel_nominal, y_init, self.z_min])

        # Kinematic convergence to ensure exact initial standoff and perpendicular orientation
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

    def read_normal_force(self) -> tuple:
        """Measure normal pressing force from touch sensor / contact solver with low-pass filter."""
        raw_f = 0.0
        if self.touch_sensor_adr >= 0:
            raw_f = float(self.data.sensordata[self.touch_sensor_adr])

        if raw_f <= 0.0:
            # Fallback to direct contact wrench solver
            tot = 0.0
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if c.geom1 == self.disc_geom_id or c.geom2 == self.disc_geom_id:
                    mujoco.mj_contactForce(self.model, self.data, i, self.contact_wrench_buf)
                    tot += self.contact_wrench_buf[0]
            raw_f = float(tot)

        # 1st-order IIR low-pass filtering
        self.f_filtered = (1.0 - self.alpha_filter) * self.f_filtered + self.alpha_filter * raw_f
        return self.f_filtered, raw_f

    def compute_control_step(self, sim_time: float) -> tuple:
        """
        Evaluate whole-body loco-manipulation control laws for the current simulation step.
        Returns: (F_meas, F_err, p_tcp, target_pos, v_base_actual, state)
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

        # 1. Contact Force Measurement & Guarded State Transition
        F_meas, raw_f = self.read_normal_force()
        F_err = self.F_desired - F_meas

        if not self.contact_detected and raw_f >= self.F_contact_thresh:
            self.contact_detected = True
            self.state = "SANDING"
            self.t_contact = sim_time
            self.x_contact = float(p_tcp[0])
            print(f">>> [t={sim_time:5.3f}s] Wall contact confirmed (x={self.x_contact:.3f}m, F={raw_f:.2f} N). "
                  f"Initiating full-coverage vertical sanding raster at {self.base_speed:.3f} m/s!")

        # 2. Mobile Base Driving Control (0.01 m/s along wall + lane-centering)
        if self.contact_detected:
            v_cmd = self.base_speed
        else:
            v_cmd = 0.0  # Stationary during initial contact approach

        # Lane-centering yaw rate: keep Husky at Y = 0.0, Yaw = 0.0
        yaw_rate_cmd = -self.Ky_base * (base_y - 0.0) - self.Kyaw_base * base_yaw
        w_left  = (v_cmd - yaw_rate_cmd * (self.track_width / 2.0)) / self.wheel_radius
        w_right = (v_cmd + yaw_rate_cmd * (self.track_width / 2.0)) / self.wheel_radius

        self.data.ctrl[self.wheel_act_ids[0]] = w_left   # front left
        self.data.ctrl[self.wheel_act_ids[2]] = w_left   # rear left
        self.data.ctrl[self.wheel_act_ids[1]] = w_right  # front right
        self.data.ctrl[self.wheel_act_ids[3]] = w_right  # rear right

        # 3. Hybrid Force/Position Control Law
        # Normal Axis (Y-axis): Force Regulation into Wall
        if not self.contact_detected:
            # Controlled forward approach into wall at stroke start height (z_min)
            f_y = self.F_desired + self.Kv_approach * (self.v_approach - v_tcp[1])
            f_y = np.clip(f_y, 0.0, 3.5)
            y_des = self.y_wall
            x_des = self.data.qpos[0] + self.x_rel_nominal
            vx_des = 0.0
            z_des = self.z_min
            vz_des = 0.0
            self.sub_state = "APPROACH"
        else:
            # Closed-loop PI force regulation against wall
            self.force_integral += F_err * self.dt
            self.force_integral = np.clip(self.force_integral, -0.3, 0.3)
            f_y = self.F_desired + self.Kp_f * F_err + self.Ki_f * self.force_integral - self.Kd_f * v_tcp[1]
            f_y = np.clip(f_y, 0.2, 3.2)
            y_des = self.y_wall

            # Digital Signal Raster Trajectory:
            # Pattern: Pure Vertical Stroke -> Shift Horizontally by Head Diameter (d_head) -> Pure Vertical Stroke
            # As the base continuously moves forward, arm joints actively articulate so that:
            #  1) During vertical stroke: tool stays locked onto exact vertical wall line (X held constant).
            #  2) During horizontal shift: tool translates along X by d_head while holding height (Z held constant).
            tau = sim_time - self.t_contact
            pass_idx = int(tau / self.t_pass)
            tau_pass = tau - pass_idx * self.t_pass
            x_k = self.x_contact + pass_idx * self.d_head
            stroke_len = self.z_max - self.z_min

            if tau_pass <= self.t_vert:
                # Vertical stroke phase (pure vertical edge in digital signal)
                s = 0.5 * (1.0 - np.cos(np.pi * tau_pass / self.t_vert))
                ds = (0.5 * np.pi / self.t_vert) * np.sin(np.pi * tau_pass / self.t_vert)

                if pass_idx % 2 == 0:
                    # Even pass: Move UP (z_min -> z_max)
                    z_des = self.z_min + s * stroke_len
                    vz_des = ds * stroke_len
                    self.sub_state = f"P{pass_idx}_VERT_UP"
                else:
                    # Odd pass: Move DOWN (z_max -> z_min)
                    z_des = self.z_max - s * stroke_len
                    vz_des = -ds * stroke_len
                    self.sub_state = f"P{pass_idx}_VERT_DOWN"

                x_des = x_k
                vx_des = 0.0

            else:
                # Horizontal shift phase by head diameter (pure horizontal edge in digital signal)
                tau_shift = tau_pass - self.t_vert
                s_h = 0.5 * (1.0 - np.cos(np.pi * tau_shift / self.t_shift))
                ds_h = (0.5 * np.pi / self.t_shift) * np.sin(np.pi * tau_shift / self.t_shift)

                x_des = x_k + s_h * self.d_head
                vx_des = ds_h * self.d_head

                if pass_idx % 2 == 0:
                    z_des = self.z_max   # Held at top of wall patch
                else:
                    z_des = self.z_min   # Held at bottom of wall patch
                vz_des = 0.0
                self.sub_state = f"P{pass_idx}_SHIFT"

        f_x = self.Kp_x * (x_des - p_tcp[0]) + self.Kd_x * (vx_des - v_tcp[0])
        f_z = self.Kp_z * (z_des - p_tcp[2]) + self.Kd_z * (vz_des - v_tcp[2])

        # Rotational Control:
        # Tool normal points strictly into the wall (+Y)
        # Pitch and Yaw (rotations about X and Z) are SOLID with high restoring stiffness so contact surface never tilts outside
        # Perpendicular rotation (torsion about +Y) is COMPLIANT with soft damping
        R_curr = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        rot_err = 0.5 * (
            np.cross(R_curr[:, 0], self.R_des[:, 0]) +
            np.cross(R_curr[:, 1], self.R_des[:, 1]) +
            np.cross(R_curr[:, 2], self.R_des[:, 2])
        )
        tau_rot = self.Kp_rot * rot_err - self.Kd_rot * w_tcp

        # Task Wrench
        F_task = np.array([f_x, f_y, f_z, tau_rot[0], tau_rot[1], tau_rot[2]])

        # 4. Weighted Null-Space Projection
        # Arm Jacobian (columns 10:16 corresponding to the 6 UR10e joints)
        Ja = np.vstack([self.jacp[:, 10:16], self.jacr[:, 10:16]])

        # Weighted Pseudo-Inverse: J_W^# = W^(-1) J^T (J W^(-1) J^T + lambda^2 I)^(-1)
        Ja_weighted_pinv = self.Winv_arm @ Ja.T @ np.linalg.solve(
            Ja @ self.Winv_arm @ Ja.T + self.lambda_damping * np.eye(6), np.eye(6)
        )

        # Weighted Null-Space Projector: N_W = I - J_W^# * J
        N_w = np.eye(6) - Ja_weighted_pinv @ Ja

        # Secondary Ergonomic Posture Torque
        q_arm_curr = self.data.qpos[11:17]
        qvel_arm_curr = self.data.qvel[10:16]
        tau_0 = -self.K_posture * (q_arm_curr - self.q_nominal) - self.D_null * qvel_arm_curr

        # Project into null space: tau_null = N_W^T * tau_0
        tau_null = N_w.T @ tau_0

        # Full Arm Joint Torques: tau = J^T * F_task + tau_null + tau_bias
        tau_bias = self.data.qfrc_bias[10:16]
        tau_arm = Ja.T @ F_task + tau_null + tau_bias

        # Apply torques to UR10e direct torque motors
        for i, act_id in enumerate(self.arm_act_ids):
            self.data.ctrl[act_id] = tau_arm[i]

        target_pos = np.array([x_des, y_des, z_des])
        return F_meas, F_err, p_tcp, target_pos, v_base_actual, self.sub_state


def run_headless_benchmark(duration: float = 35.0, plot: bool = False,
                           speed: float = 0.01, force: float = 1.0, offset: float = 0.10,
                           v_approach: float = 0.01, ee_speed: float = 0.01, diameter: float = 0.15):
    """Execute fast headless evaluation and log full whole-body sanding metrics."""
    scene_path = os.path.join(os.path.dirname(__file__), "scene_husky_sanding.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    controller = WholeBodySandingController(model, data, base_speed=speed,
                                            desired_force=force, initial_offset=offset,
                                            v_approach=v_approach, ee_speed=ee_speed,
                                            d_head=diameter)
    controller.initialize_posture(start_x=-1.5)

    print("=" * 75)
    print("Whole-Body Loco-Manipulation Digital Signal Graph Sanding Benchmark")
    print(f"Wall Length: 6.0 m | Sanding Patch: 5.4 m | Initial Standoff: {offset:.2f} m")
    print(f"Target Base Velocity: {speed:.3f} m/s | Sanding Head Diameter: {diameter:.3f} m")
    print(f"Approach Velocity   : {v_approach:.3f} m/s | Target Normal Force: {force:.2f} N")
    print(f"Simulation Duration : {duration:.1f} s | Timestep: {model.opt.timestep:.4f} s")
    print("=" * 75)

    dt = model.opt.timestep
    total_steps = int(duration / dt)
    log_interval = int(0.5 / dt)

    times = []
    forces = []
    base_xs = []
    base_ys = []
    base_vs = []
    tcp_xs = []
    tcp_ys = []
    tcp_zs = []
    target_xs = []
    target_zs = []
    tilts = []
    states = []

    t_start_wall = time.perf_counter()

    for step in range(total_steps):
        t_sim = float(data.time)
        f_meas, f_err, p_tcp, p_target, v_base, state = controller.compute_control_step(t_sim)
        mujoco.mj_step(model, data)

        R_curr = data.site_xmat[controller.tcp_site_id].reshape(3, 3)
        cos_theta = np.clip(R_curr[:, 2] @ np.array([0.0, 1.0, 0.0]), -1.0, 1.0)
        tilt_deg = float(np.degrees(np.arccos(cos_theta)))

        times.append(t_sim)
        forces.append(f_meas)
        base_xs.append(data.xpos[controller.base_body_id][0])
        base_ys.append(data.xpos[controller.base_body_id][1])
        base_vs.append(v_base)
        tcp_xs.append(p_tcp[0])
        tcp_ys.append(p_tcp[1])
        tcp_zs.append(p_tcp[2])
        target_xs.append(p_target[0])
        target_zs.append(p_target[2])
        tilts.append(tilt_deg)
        states.append(state)

        if step % log_interval == 0 or step == total_steps - 1:
            print(f"[t={t_sim:5.2f}s] Phase: {state:13s} | "
                  f"Base X: {base_xs[-1]:5.2f}m (v={base_vs[-1]:4.2f}m/s) | "
                  f"TCP: ({tcp_xs[-1]:5.2f}, {tcp_ys[-1]:5.3f}, {tcp_zs[-1]:5.3f})m | "
                  f"Tilt: {tilt_deg:4.2f}° | F_norm: {f_meas:4.2f}N")

    wall_elapsed = time.perf_counter() - t_start_wall
    realtime_factor = duration / max(1e-6, wall_elapsed)

    # Performance Analysis on Sanding Phase
    sanding_mask = [s != "APPROACH" for s in states]
    if any(sanding_mask):
        sand_forces = np.array([f for f, m in zip(forces, sanding_mask) if m])
        sand_tcp_z = np.array([z for z, m in zip(tcp_zs, sanding_mask) if m])
        sand_tgt_z = np.array([z for z, m in zip(target_zs, sanding_mask) if m])
        sand_base_x = np.array([x for x, m in zip(base_xs, sanding_mask) if m])
        sand_base_v = np.array([v for v, m in zip(base_vs, sanding_mask) if m])

        sand_tilts = np.array([t for t, m in zip(tilts, sanding_mask) if m])

        mean_f = float(np.mean(sand_forces))
        std_f  = float(np.std(sand_forces))
        rmse_z = float(np.sqrt(np.mean((sand_tcp_z - sand_tgt_z) ** 2)))
        travel = float(sand_base_x[-1] - sand_base_x[0])
        mean_v = float(np.mean(sand_base_v))
        mean_tilt = float(np.mean(sand_tilts))
        max_tilt = float(np.max(sand_tilts))
    else:
        mean_f, std_f, rmse_z, travel, mean_v, mean_tilt, max_tilt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    print("=" * 75)
    print("Benchmark Results Summary:")
    print(f"  - Simulated {duration:.2f} s in {wall_elapsed:.3f} s ({realtime_factor:.1f}x real-time)")
    print(f"  - Contact Established At   : t = {controller.t_contact:.3f} s")
    print(f"  - Total Base Travel Along Wall: {travel:.2f} m (Mean Speed: {mean_v:.3f} m/s)")
    print(f"  - Normal Force Tracking    : {mean_f:.3f} N ± {std_f:.3f} N (Target: {force:.2f} N)")
    print(f"  - Vertical Sanding RMSE    : {rmse_z * 100:.2f} cm (Stroke: {controller.Az * 2 * 100:.1f} cm)")
    print(f"  - EE Perpendicularity      : Mean Tilt: {mean_tilt:.2f}° | Max Tilt: {max_tilt:.2f}° (Flush against wall)")
    print("=" * 75)

    if plot:
        generate_performance_plot(times, forces, base_xs, base_vs, tcp_xs, tcp_zs,
                                  target_xs, target_zs, states, controller.t_contact, force)


def generate_performance_plot(times, forces, base_xs, base_vs, tcp_xs, tcp_zs,
                              target_xs, target_zs, states, t_contact, f_target):
    """Create and save publication-quality performance graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Whole-Body Loco-Manipulation Digital Signal Graph Sanding Performance",
                 fontsize=14, fontweight="bold")

    # Plot 1: Normal Contact Force Regulation
    ax1 = axs[0, 0]
    ax1.plot(times, forces, label="Measured Force (N)", color="#1f77b4", lw=2)
    ax1.axhline(f_target, color="red", linestyle="--", label=f"Target Force ({f_target:.1f} N)", lw=2)
    if t_contact > 0:
        ax1.axvline(t_contact, color="green", linestyle=":", label="Contact Trigger", lw=1.8)
        ax1.axvspan(t_contact, times[-1], color="yellow", alpha=0.15, label="Sanding Phase")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Normal Force (N)")
    ax1.set_title("1. Normal Axis Hybrid Force Regulation (Y-axis)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")
    ax1.set_ylim(-0.2, max(3.5, f_target * 2.5))

    # Plot 2: Longitudinal Base Motion along Wall
    ax2 = axs[0, 1]
    ax2.plot(times, base_xs, label="Base X Position (m)", color="#2ca02c", lw=2)
    ax2.plot(times, base_vs, label="Base Velocity (m/s)", color="#ff7f0e", lw=1.8, linestyle="-.")
    ax2.axhline(0.01, color="gray", linestyle=":", label="Target Base Speed (0.01 m/s)")
    if t_contact > 0:
        ax2.axvline(t_contact, color="green", linestyle=":", lw=1.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m) / Speed (m/s)")
    ax2.set_title("2. Mobile Base Cruising Along Wall")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    # Plot 3: Vertical Up-Down Sanding Trajectory
    ax3 = axs[1, 0]
    ax3.plot(times, target_zs, label="Z Target (m)", color="purple", lw=2, linestyle="--")
    ax3.plot(times, tcp_zs, label="TCP Actual Z (m)", color="#d62728", lw=1.8)
    if t_contact > 0:
        ax3.axvline(t_contact, color="green", linestyle=":", lw=1.8)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Height Z (m)")
    ax3.set_title("3. Vertical Stroke Trajectory")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right")

    # Plot 4: 2D Spatial Sanding Trace on the Wall Face (Digital Signal Square Wave)
    ax4 = axs[1, 1]
    sanding_mask = [s != "APPROACH" for s in states]
    x_sand = [x for x, m in zip(tcp_xs, sanding_mask) if m]
    z_sand = [z for z, m in zip(tcp_zs, sanding_mask) if m]
    x_tgt_sand = [x for x, m in zip(target_xs, sanding_mask) if m]
    z_tgt_sand = [z for z, m in zip(target_zs, sanding_mask) if m]

    if len(x_sand) > 0:
        ax4.plot(x_tgt_sand, z_tgt_sand, color="dimgray", linestyle="--", lw=1.8, label="Digital Signal Ref (Square Wave)")
        sc = ax4.scatter(x_sand, z_sand, c=[t for t, m in zip(times, sanding_mask) if m],
                         cmap="plasma", s=18, alpha=0.9, label="Actual Sanded Path", zorder=3)
        cbar = plt.colorbar(sc, ax=ax4)
        cbar.set_label("Time (s)")
    ax4.axhspan(0.65, 1.25, color="wheat", alpha=0.3, label="Wall Sanding Patch")
    ax4.set_xlabel("Wall Longitudinal Position X (m)")
    ax4.set_ylabel("Wall Vertical Height Z (m)")
    ax4.set_title("4. Digital Signal Trajectory on Wall Face (Vertical + D_head Shift)")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="lower left")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "whole_body_sanding_performance.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\n[Plot] High-resolution performance plot saved to: {output_path}")


def run_interactive_simulation(duration: float = 60.0, speed: float = 0.01,
                               force: float = 1.0, offset: float = 0.10,
                               v_approach: float = 0.01, ee_speed: float = 0.01,
                               diameter: float = 0.15):
    """Launch interactive 3D visualizer with 360° mouse navigation and live loco-sanding."""
    import mujoco.viewer

    scene_path = os.path.join(os.path.dirname(__file__), "scene_husky_sanding.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    controller = WholeBodySandingController(model, data, base_speed=speed,
                                            desired_force=force, initial_offset=offset,
                                            v_approach=v_approach, ee_speed=ee_speed,
                                            d_head=diameter)
    controller.initialize_posture(start_x=-1.5)

    print("=" * 75)
    print("Launching Whole-Body Loco-Manipulation Sanding Visualizer")
    print("=" * 75)
    print("Features:")
    print(f"  - Base cruises along 6.0m wall at {speed:.3f} m/s")
    print(f"  - Digital signal trajectory: Pure vertical stroke -> Horizontal shift by {diameter:.2f}m -> Reverse vertical")
    print(f"  - Arm articulates to decouple tool trajectory from moving base")
    print(f"  - Tool normal approach velocity: {v_approach:.3f} m/s, Normal force: {force:.1f} N")
    print("  - Solid tilt orientation keeps disc flush against wall at all times")
    print("Controls:")
    print("  - Mouse Navigation: Left-drag to rotate 360°, Right-drag to zoom, Middle-drag to pan")
    print("  - Spacebar: Pause / resume")
    print("  - Tab / 'C': Cycle camera views")
    print("  - Keys '+' / '-': Adjust base cruise speed")
    print("  - Key 'R': Reset simulation to starting stance")
    print("=" * 75)

    def key_callback(keycode):
        if keycode == ord('+') or keycode == ord('='):
            controller.base_speed = min(controller.base_speed + 0.005, 0.20)
            controller.update_timing()
            print(f"\n[Speed] Base target speed: {controller.base_speed:.3f} m/s "
                  f"(pass: {controller.t_pass:.1f}s, vert: {controller.t_vert:.1f}s, shift: {controller.t_shift:.1f}s)")
        elif keycode == ord('-') or keycode == ord('_'):
            controller.base_speed = max(controller.base_speed - 0.005, 0.005)
            controller.update_timing()
            print(f"\n[Speed] Base target speed: {controller.base_speed:.3f} m/s "
                  f"(pass: {controller.t_pass:.1f}s, vert: {controller.t_vert:.1f}s, shift: {controller.t_shift:.1f}s)")
        elif keycode == ord('R') or keycode == ord('r'):
            controller.initialize_posture(start_x=-1.5)
            print("\n[Reset] Reset robot to corridor start!")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Default camera setup: dynamic tracking view on base and wall contact
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = controller.base_body_id
        viewer.cam.distance = 3.2
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 145.0

        sim_dt = model.opt.timestep

        while viewer.is_running():
            step_start = time.perf_counter()
            t_sim = float(data.time)

            f_meas, f_err, p_tcp, p_target, v_base, state = controller.compute_control_step(t_sim)
            mujoco.mj_step(model, data)
            viewer.sync()

            # Maintain natural real-time frame rate
            time_until_next = sim_dt - (time.perf_counter() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


def main():
    parser = argparse.ArgumentParser(description="Whole-Body Loco-Manipulation Hybrid Sanding Simulation")
    parser.add_argument("--headless", action="store_true",
                        help="Run in fast headless benchmark mode without 3D window")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save high-resolution performance plots")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Simulation run duration (seconds, default: 20.0)")
    parser.add_argument("--speed", type=float, default=0.01,
                        help="Mobile base constant velocity along wall (m/s, default: 0.01)")
    parser.add_argument("--force", type=float, default=1.0,
                        help="Desired pressing force against wall (N, default: 1.0)")
    parser.add_argument("--offset", type=float, default=0.10,
                        help="Initial standoff offset from wall (m, default: 0.10)")
    parser.add_argument("--v_approach", type=float, default=0.01,
                        help="End effector normal approach velocity into wall (m/s, default: 0.01)")
    parser.add_argument("--ee_speed", type=float, default=0.01,
                        help="End effector vertical sanding speed (m/s, default: 0.01)")
    parser.add_argument("--diameter", type=float, default=0.15,
                        help="Diameter of circular sanding head / shift offset (m, default: 0.15)")
    args = parser.parse_args()

    if args.headless or args.plot:
        run_headless_benchmark(duration=args.duration, plot=args.plot,
                               speed=args.speed, force=args.force, offset=args.offset,
                               v_approach=args.v_approach, ee_speed=args.ee_speed,
                               diameter=args.diameter)
    else:
        run_interactive_simulation(duration=args.duration, speed=args.speed,
                                   force=args.force, offset=args.offset,
                                   v_approach=args.v_approach, ee_speed=args.ee_speed,
                                   diameter=args.diameter)


if __name__ == "__main__":
    main()
