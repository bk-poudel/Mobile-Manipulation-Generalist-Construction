#!/usr/bin/env python3
"""
Hybrid Position/Force Control for UR10e Robot in MuJoCo
======================================================
This script implements a task-space Hybrid Position/Force Controller (Raibert & Craig)
for a Universal Robots UR10e manipulator equipped with a circular sanding mount end-effector.

Workflow & Control Architecture:
--------------------------------
1. Initial Offset & Approach Phase:
   - Robot initializes with an initial offset (default: 0.10 m) away from the tall wall.
   - Nominal wall surface is at Y = 0.8000 m; initial TCP is at Y = 0.7000 m.
   - Force controller drives the robot forward along the +Y normal axis at a smooth,
     controlled approach velocity (v_y ~ 0.04 m/s).
   - In the perpendicular sanding plane (X-Z), TCP holds steady at center (X=-0.15 m, Z=0.70 m).

2. Guarded Contact Transition:
   - A circular touch sensor and MuJoCo contact wrench monitor the sanding disc face.
   - When contact force F_meas >= 0.3 N is detected, the controller seamlessly transitions
     from APPROACH to SANDING state and notes the contact timestamp t_contact.

3. Sanding Phase (Normal Force Regulation + Perpendicular Sweeping):
   - Normal Force Control (Y-axis):
     Closed-loop PI force regulation maintains F_desired = 1.0 N:
     f_y = F_desired + Kp_f * (F_desired - F_meas) + Ki_f * integral(e_F) - Kd_f * v_y
   - Sweeping Sanding Motion (X-Z plane):
     Sinusoidal raster sweeping pattern activates upon contact:
     x_des(t) = x_center + Ax * sin(2*pi*fx * (t - t_contact))
     z_des(t) = z_center + Az * sin(2*pi*fz * (t - t_contact))
     Tracked via high-gain task-space PD control with feedforward velocity.
   - Orientation Compliance:
     Angular velocity damping allows the sanding pad to align flush against the wall.

4. Operational Space Dynamics & Torque Mapping:
   - Joint torques: tau = J^T * F_task + tau_bias + tau_null
   - tau_bias: Coriolis, centrifugal, and gravity compensation (data.qfrc_bias).
   - tau_null: Secondary nullspace velocity damping.

Usage:
------
    python3 hybrid_controller.py                 # Interactive 3D visualization
    python3 hybrid_controller.py --headless      # Headless benchmark / evaluation
    python3 hybrid_controller.py --plot          # Run and save performance plot
    python3 hybrid_controller.py --offset 0.10   # Custom initial offset (e.g. 0.10 m)
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco

# Ensure writable matplotlib cache dir
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"


class HybridForcePositionController:
    """
    Hybrid Position/Force Controller for UR10e sanding on a vertical wall.
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 initial_offset: float = 0.10, desired_force: float = 1.0,
                 v_approach: float = 0.01, f_thresh: float = 0.30,
                 freq_x: float = 0.04, freq_z: float = 0.02,
                 ax: float = 0.04, az: float = 0.02,
                 kp_f: float = 0.70, ki_f: float = 1.80, kd_f: float = 3.50,
                 kp_p: float = 40.0, kd_p: float = 10.0,
                 kp_tilt: float = 100.0, kd_tilt: float = 12.0):
        self.model = model
        self.data = data
        self.dt = model.opt.timestep
        self.initial_offset = initial_offset

        # Site & Sensor IDs
        self.tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.touch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "sander_touch_sensor")
        self.disc_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sander_disc_col")

        # Nominal wall contact plane (front surface of wall cuboid at Y = 0.8000 m)
        self.y_wall = 0.8000

        # Target Force & Contact Detection Threshold
        self.F_desired = desired_force        # Desired normal pressing force (N)
        self.F_contact_thresh = f_thresh      # Force threshold to detect wall contact (N)

        # State Tracking
        self.state = "APPROACH"               # "APPROACH" or "SANDING"
        self.contact_detected = False
        self.t_contact = 0.0

        # Approach Phase Parameters
        self.v_approach = v_approach          # Forward approach velocity into wall (m/s)
        self.Kv_approach = 25.0               # Velocity damping gain during approach

        # Force Control Gains (Normal axis Y during Sanding)
        self.Kp_f = kp_f                      # Proportional gain on force error
        self.Ki_f = ki_f                      # Integral gain on force error
        self.Kd_f = kd_f                      # Task-space normal velocity damping
        self.force_integral = 0.0

        # Low-pass filter for force sensor (1st order IIR, removes numerical contact chatter)
        self.f_filtered = 0.0
        self.alpha_filter = 0.15

        # Tangential Position Control Gains (X and Z axes)
        self.Kp_p = kp_p                      # Task-space position proportional gain (N/m)
        self.Kd_p = kd_p                      # Task-space velocity damping (N*s/m)

        # Orientation Stiffness & Damping:
        # Tilt axes (X and Z): High stiffness to keep disc flush so contact surface never moves outside
        # Perpendicular axis (Y): Soft compliance / damping for friction relief and wrist flexibility
        self.Kp_tilt = kp_tilt                # High restoring stiffness for pitch and yaw (Nm/rad)
        self.Kd_tilt = kd_tilt                # Tilt angular damping (Nms/rad)
        self.Kp_perp = 5.0                    # Soft compliance about normal axis (Nm/rad)
        self.Kd_perp = 2.0                    # Torsional damping about normal axis (Nms/rad)
        self.Kp_rot = np.array([self.Kp_tilt, self.Kp_perp, self.Kp_tilt])
        self.Kd_rot = np.array([self.Kd_tilt, self.Kd_perp, self.Kd_tilt])

        # Nullspace Joint Damping
        self.D_null = 0.20

        # Desired Orientation Matrix (Sanding disc normal aligned with +Y into wall)
        Rx = np.array([1.0, 0.0, 0.0])
        Rz = np.array([0.0, 1.0, 0.0])        # Tool Z points along +Y
        Ry = np.cross(Rz, Rx)                # Tool Y points along -Z
        self.R_desired = np.column_stack([Rx, Ry, Rz])

        # Sweeping Trajectory Settings
        self.x_center = -0.15
        self.z_center = 0.70
        self.Ax = ax                          # Horizontal sweep amplitude (m)
        self.Az = az                          # Vertical scan amplitude (m)
        self.freq_x = freq_x                  # Horizontal sweep frequency (Hz)
        self.freq_z = freq_z                  # Vertical scan frequency (Hz)

        # Preallocated buffers for Jacobian and contact wrench
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.contact_wrench_buf = np.zeros(6)

    def initialize_pose(self, offset: float = None):
        """
        Set robot initial configuration facing wall at the specified offset away from the wall.
        """
        if offset is not None:
            self.initial_offset = offset

        # Reset states
        self.state = "APPROACH"
        self.contact_detected = False
        self.t_contact = 0.0
        self.force_integral = 0.0
        self.f_filtered = 0.0

        # Nominal ready joint angles
        q_ready = np.array([-1.618, -1.672, 1.621, 0.05, 1.523, 3.142])
        self.data.qpos[:] = q_ready
        self.data.qvel[:] = 0.0

        # Target initial TCP position: offset in front of wall surface
        y_init = self.y_wall - self.initial_offset
        target_init = np.array([self.x_center, y_init, self.z_center])

        # Kinematic convergence to ensure exact initial standoff and perpendicular orientation
        for _ in range(250):
            mujoco.mj_forward(self.model, self.data)
            pos_err = target_init - self.data.site_xpos[self.tcp_id]
            R_curr = self.data.site_xmat[self.tcp_id].reshape(3, 3)
            rot_err = 0.5 * (
                np.cross(R_curr[:, 0], self.R_desired[:, 0]) +
                np.cross(R_curr[:, 1], self.R_desired[:, 1]) +
                np.cross(R_curr[:, 2], self.R_desired[:, 2])
            )
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.tcp_id)
            J = np.vstack([self.jacp, self.jacr])
            err_6d = np.concatenate([pos_err, rot_err])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err_6d)
            self.data.qpos[:] += dq * 0.4

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_measured_force(self) -> tuple:
        """
        Read normal contact force from touch sensor / contact solver and apply low-pass filtering.
        Returns (f_filtered, raw_force).
        """
        raw_f = 0.0
        if self.touch_id >= 0:
            raw_f = float(self.data.sensordata[0])

        if raw_f <= 0.0:
            # Fallback to direct constraint contact force calculation
            total_f = 0.0
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if c.geom1 == self.disc_geom_id or c.geom2 == self.disc_geom_id:
                    mujoco.mj_contactForce(self.model, self.data, i, self.contact_wrench_buf)
                    total_f += self.contact_wrench_buf[0]
            raw_f = float(total_f)

        # 1st order IIR low-pass filter
        self.f_filtered = (1.0 - self.alpha_filter) * self.f_filtered + self.alpha_filter * raw_f
        return self.f_filtered, raw_f

    def compute_torques(self, sim_time: float) -> tuple:
        """
        Compute hybrid position/force control torques for the current step.
        Returns (tau_command, F_meas, f_err, tcp_pos, target_pos, state).
        """
        # Forward kinematics and Jacobian
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.tcp_id)
        J = np.vstack([self.jacp, self.jacr])

        # Current state at TCP
        p_curr = self.data.site_xpos[self.tcp_id]
        v_curr = self.jacp @ self.data.qvel
        omega_curr = self.jacr @ self.data.qvel

        # 1. Measured Contact Force
        F_meas, raw_f = self.get_measured_force()
        f_err = self.F_desired - F_meas

        # 2. Guarded Contact State Machine
        if not self.contact_detected and raw_f >= self.F_contact_thresh:
            self.contact_detected = True
            self.state = "SANDING"
            self.t_contact = sim_time
            print(f">>> [t={sim_time:5.3f}s] Wall contact detected (F_meas={F_meas:.3f} N)! "
                  f"Transitioning to SANDING phase.")

        # 3. Normal Axis Control Law (Y-axis):
        if not self.contact_detected:
            # APPROACH PHASE:
            # Driven towards wall by force controller with velocity damping:
            f_y = self.F_desired + self.Kv_approach * (self.v_approach - v_curr[1])
            f_y = np.clip(f_y, 0.0, 4.0)
            y_des = self.y_wall
        else:
            # SANDING PHASE:
            # Closed-loop PI force regulation using force error e_F = F_desired - F_meas
            self.force_integral += f_err * self.dt
            self.force_integral = np.clip(self.force_integral, -0.4, 0.4)
            f_y = self.F_desired + self.Kp_f * f_err + self.Ki_f * self.force_integral - self.Kd_f * v_curr[1]
            f_y = np.clip(f_y, 0.2, 3.5)
            y_des = self.y_wall

        # 4. Tangential Sanding Plane Position Control Law (X and Z axes):
        if not self.contact_detected:
            # Hold stationary at center while approaching
            x_des = self.x_center
            vx_des = 0.0
            z_des = self.z_center
            vz_des = 0.0
        else:
            # Execute sweeping sanding pattern relative to contact time
            t_sand = sim_time - self.t_contact
            x_des = self.x_center + self.Ax * np.sin(2.0 * np.pi * self.freq_x * t_sand)
            vx_des = self.Ax * (2.0 * np.pi * self.freq_x) * np.cos(2.0 * np.pi * self.freq_x * t_sand)

            z_des = self.z_center + self.Az * np.sin(2.0 * np.pi * self.freq_z * t_sand)
            vz_des = self.Az * (2.0 * np.pi * self.freq_z) * np.cos(2.0 * np.pi * self.freq_z * t_sand)

        f_x = self.Kp_p * (x_des - p_curr[0]) + self.Kd_p * (vx_des - v_curr[0])
        f_z = self.Kp_p * (z_des - p_curr[2]) + self.Kd_p * (vz_des - v_curr[2])

        # 5. Orientation Control:
        # Tool normal points strictly into the wall (+Y)
        # Pitch and Yaw (rotations about X and Z) are SOLID with high restoring stiffness so contact surface never tilts outside
        # Perpendicular rotation (torsion about +Y) is COMPLIANT with soft damping
        R_curr = self.data.site_xmat[self.tcp_id].reshape(3, 3)
        rot_err = 0.5 * (
            np.cross(R_curr[:, 0], self.R_desired[:, 0]) +
            np.cross(R_curr[:, 1], self.R_desired[:, 1]) +
            np.cross(R_curr[:, 2], self.R_desired[:, 2])
        )
        tau_rot = self.Kp_rot * rot_err - self.Kd_rot * omega_curr

        # 6. Operational Space Hybrid Wrench:
        # F_task = [f_x (pos), f_y (force), f_z (pos), tau_rx, tau_ry, tau_rz]
        F_task = np.array([f_x, f_y, f_z, tau_rot[0], tau_rot[1], tau_rot[2]])

        # 7. Joint Torques via Transpose Jacobian + Dynamics Bias Compensation
        tau_task = J.T @ F_task
        tau_bias = self.data.qfrc_bias
        tau_null = -self.D_null * self.data.qvel

        tau_cmd = tau_task + tau_bias + tau_null
        target_pos = np.array([x_des, y_des, z_des])

        return tau_cmd, F_meas, f_err, p_curr.copy(), target_pos, self.state


def run_headless(duration: float = 12.0, plot: bool = False, offset: float = 0.10,
                 force: float = 1.0, v_approach: float = 0.01, f_thresh: float = 0.30,
                 freq_x: float = 0.04, freq_z: float = 0.02, ax: float = 0.04, az: float = 0.02,
                 kp_f: float = 0.70, ki_f: float = 1.80, kd_f: float = 3.50,
                 kp_p: float = 40.0, kd_p: float = 10.0,
                 kp_tilt: float = 100.0, kd_tilt: float = 12.0,
                 plot_name: str = "sanding_performance.png"):
    """Run headless simulation and print comprehensive performance metrics."""
    model_path = os.path.join(os.path.dirname(__file__), "scene_sanding.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    controller = HybridForcePositionController(
        model, data, initial_offset=offset, desired_force=force,
        v_approach=v_approach, f_thresh=f_thresh,
        freq_x=freq_x, freq_z=freq_z, ax=ax, az=az,
        kp_f=kp_f, ki_f=ki_f, kd_f=kd_f, kp_p=kp_p, kd_p=kd_p,
        kp_tilt=kp_tilt, kd_tilt=kd_tilt
    )
    controller.initialize_pose()

    total_steps = int(duration / model.opt.timestep)
    times = []
    forces = []
    errors = []
    x_positions = []
    y_positions = []
    z_positions = []
    x_targets = []
    y_targets = []
    z_targets = []
    states = []

    print(f"--- Running Hybrid Position/Force Sanding Simulation ({duration} s) ---")
    print(f"Initial Offset from Wall: {offset:.3f} m (Initial Y = {controller.y_wall - offset:.4f} m)")
    print(f"Target Normal Force:      {force:.2f} N (Kp={kp_f}, Ki={ki_f}, Kd={kd_f}) | Timestep: {model.opt.timestep} s")
    print(f"Sweeping Parameters:      fx={freq_x}Hz, fz={freq_z}Hz | Ax={ax*100:.1f}cm, Az={az*100:.1f}cm")

    start_wall_time = time.time()
    for step in range(total_steps):
        t = step * model.opt.timestep
        tau, f_meas, f_err, p_curr, p_targ, state = controller.compute_torques(t)

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        times.append(t)
        forces.append(f_meas)
        errors.append(f_err)
        x_positions.append(p_curr[0])
        y_positions.append(p_curr[1])
        z_positions.append(p_curr[2])
        x_targets.append(p_targ[0])
        y_targets.append(p_targ[1])
        z_targets.append(p_targ[2])
        states.append(state)

        if step % 500 == 0 and step > 0:
            print(f"t={t:4.1f}s [{state:8s}] | Force: {f_meas:5.3f} N | Err: {f_err:+5.3f} N | "
                  f"TCP: [X={p_curr[0]:+.3f}, Y={p_curr[1]:.4f}, Z={p_curr[2]:.3f}]")

    elapsed_wall = time.time() - start_wall_time
    times = np.array(times)
    forces = np.array(forces)
    errors = np.array(errors)

    # Steady-state statistics (during active sanding phase, t >= t_contact + 0.5s)
    if controller.contact_detected:
        t_contact = controller.t_contact
        settled_mask = times >= (t_contact + 0.5)
        sanding_forces = forces[settled_mask]
        sanding_errors = errors[settled_mask]
    else:
        t_contact = 0.0
        sanding_forces = forces
        sanding_errors = errors

    print("\n================== Performance Summary ==================")
    print(f"Simulation Speed:        {duration / elapsed_wall:.1f}x real-time")
    print(f"Initial Offset:          {offset:.3f} m (Wall at Y={controller.y_wall:.4f} m)")
    print(f"Contact Established At:  t = {t_contact:.3f} s")
    print(f"Desired Normal Force:    {controller.F_desired:.3f} N")
    if len(sanding_forces) > 0:
        print(f"Mean Sanding Force:      {np.mean(sanding_forces):.4f} N (target: {controller.F_desired:.3f} N)")
        print(f"Force Error (Mean | RMS):{np.mean(sanding_errors):+.4f} N | {np.sqrt(np.mean(sanding_errors**2)):.4f} N")
        print(f"Force Std Dev:           {np.std(sanding_forces):.4f} N")
        print(f"Min / Max Sanding Force: {np.min(sanding_forces):.4f} N / {np.max(sanding_forces):.4f} N")
        print(f"Contact Duty Cycle:      {np.sum(sanding_forces > 0.2) / len(sanding_forces) * 100:.1f} %")
    print("=========================================================\n")

    if plot:
        generate_performance_plot(times, forces, errors, x_positions, y_positions, z_positions,
                                  x_targets, z_targets, t_contact, force, offset, plot_name=plot_name)


def generate_performance_plot(times, forces, errors, x_pos, y_pos, z_pos, x_tar, z_tar, t_contact, f_target, offset,
                              plot_name: str = "sanding_performance.png"):
    """Plot force tracking, normal approach, and sweeping pattern to a high-resolution PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("UR10e Hybrid Position/Force Sanding Performance Evaluation", fontsize=15, fontweight="bold")

    # 1. Normal Force Tracking (Top-Left)
    axs[0, 0].plot(times, forces, color="#1f77b4", linewidth=2.0, label=r"Measured Force ($F_y$)")
    axs[0, 0].axhline(f_target, color="#d62728", linestyle="--", linewidth=1.8, label=f"Target Force ({f_target:.1f} N)")
    axs[0, 0].fill_between(times, f_target - 0.2, f_target + 0.2, color="#2ca02c", alpha=0.15, label=r"$\pm 0.2$ N Band")
    if t_contact > 0:
        axs[0, 0].axvline(t_contact, color="purple", linestyle=":", linewidth=1.8, label=f"Contact (t={t_contact:.2f}s)")
        axs[0, 0].axvspan(t_contact, times[-1], color="yellow", alpha=0.15, label="Sanding Phase")
    axs[0, 0].set_ylabel("Normal Force [N]", fontsize=11, fontweight="bold")
    axs[0, 0].set_title(f"Normal Force Regulation ({f_target:.1f} N Target)", fontsize=12, fontweight="bold")
    axs[0, 0].set_ylim(-0.1, max(3.5, f_target * 2.5))
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)
    axs[0, 0].legend(loc="upper right", framealpha=0.9, fontsize=9)

    # 2. Normal Axis Approach Trajectory Y(t) (Top-Right)
    axs[0, 1].plot(times, y_pos, color="#2ca02c", linewidth=2.0, label="TCP Y(t) Actual")
    axs[0, 1].axhline(0.8000, color="darkred", linestyle="--", linewidth=1.8, label="Wall Surface (Y=0.8000 m)")
    axs[0, 1].axhline(0.8000 - offset, color="gray", linestyle=":", linewidth=1.5, label=f"Initial Offset ({offset:.2f} m)")
    if t_contact > 0:
        axs[0, 1].axvline(t_contact, color="purple", linestyle=":", linewidth=1.8)
    axs[0, 1].set_ylabel("Normal Position Y [m]", fontsize=11, fontweight="bold")
    axs[0, 1].set_title(f"Normal Approach: {offset:.2f} m Offset to Wall Surface", fontsize=12, fontweight="bold")
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)
    axs[0, 1].legend(loc="lower right", framealpha=0.9, fontsize=9)

    # 3. Tangential Sanding Sweeping Pattern (Bottom-Left)
    axs[1, 0].plot(times, x_pos, color="#9467bd", linewidth=1.8, label="X(t) Actual")
    axs[1, 0].plot(times, x_tar, color="#c5b0d5", linestyle="--", linewidth=1.5, label="X(t) Desired")
    axs[1, 0].plot(times, z_pos, color="#8c564b", linewidth=1.8, label="Z(t) Actual")
    axs[1, 0].plot(times, z_tar, color="#c49c94", linestyle="--", linewidth=1.5, label="Z(t) Desired")
    if t_contact > 0:
        axs[1, 0].axvline(t_contact, color="purple", linestyle=":", linewidth=1.8, label="Sanding Begins")
    axs[1, 0].set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    axs[1, 0].set_ylabel("Tangential Position [m]", fontsize=11, fontweight="bold")
    axs[1, 0].set_title("Tangential Sanding Trajectory (X & Z Sweeps)", fontsize=12, fontweight="bold")
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)
    axs[1, 0].legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=2)

    # 4. Force Error Delta F = F_des - F_meas (Bottom-Right)
    axs[1, 1].plot(times, errors, color="#ff7f0e", linewidth=1.6, label=r"Force Error $\Delta F = F_{des} - F_{meas}$")
    axs[1, 1].axhline(0.0, color="gray", linestyle="-", linewidth=1.0)
    if t_contact > 0:
        axs[1, 1].axvline(t_contact, color="purple", linestyle=":", linewidth=1.8)
    axs[1, 1].set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    axs[1, 1].set_ylabel("Force Error [N]", fontsize=11, fontweight="bold")
    axs[1, 1].set_title(r"Force Regulation Error ($\Delta F \rightarrow 0$)", fontsize=12, fontweight="bold")
    axs[1, 1].set_ylim(-1.5, 1.5)
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)
    axs[1, 1].legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), plot_name)
    plt.savefig(output_path, dpi=200)
    print(f"Performance plot saved successfully to: {output_path}")


def run_gui(duration: float = None, offset: float = 0.10,
            force: float = 1.0, v_approach: float = 0.01, f_thresh: float = 0.30,
            freq_x: float = 0.04, freq_z: float = 0.02, ax: float = 0.04, az: float = 0.02,
            kp_f: float = 0.70, ki_f: float = 1.80, kd_f: float = 3.50,
            kp_p: float = 40.0, kd_p: float = 10.0,
            kp_tilt: float = 100.0, kd_tilt: float = 12.0):
    """Launch interactive 3D viewer demonstrating the approach and sanding task in real time."""
    import mujoco.viewer

    model_path = os.path.join(os.path.dirname(__file__), "scene_sanding.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    controller = HybridForcePositionController(
        model, data, initial_offset=offset, desired_force=force,
        v_approach=v_approach, f_thresh=f_thresh,
        freq_x=freq_x, freq_z=freq_z, ax=ax, az=az,
        kp_f=kp_f, ki_f=ki_f, kd_f=kd_f, kp_p=kp_p, kd_p=kd_p,
        kp_tilt=kp_tilt, kd_tilt=kd_tilt
    )
    controller.initialize_pose()

    print("=" * 65)
    print("           Interactive MuJoCo 3D Sanding Viewer")
    print("=" * 65)
    print(f"Parameters:")
    print(f"  • Starting Standoff       : {offset:.2f} m (Y = {controller.y_wall - offset:.4f} m)")
    print(f"  • Target Normal Force     : {force:.2f} N (Kp={kp_f}, Ki={ki_f}, Kd={kd_f})")
    print(f"  • Approach Velocity       : {v_approach:.3f} m/s")
    print(f"  • Sweeping Trajectory     : fx={freq_x}Hz, fz={freq_z}Hz | Ax={ax*100:.1f}cm, Az={az*100:.1f}cm")
    print(f"  • Stiffness Gains         : Task Kp={kp_p}, Kd={kd_p} | Tilt Kp={kp_tilt}, Kd={kd_tilt}")
    print("\nMouse Camera Controls (Free Orbit Enabled):")
    print("  • Left-click + drag       : Rotate / Orbit camera 360° around target")
    print("  • Right-click + drag      : Zoom in / out")
    print("  • Shift + Right / Middle  : Pan camera horizontally / vertically")
    print("\nKeyboard Shortcuts:")
    print("  • Space                   : Pause / Unpause simulation")
    print("  • '[' and ']'             : Cycle between Free Camera & Preset Cameras")
    print("  • Backspace               : Reset simulation")
    print("  • Esc / Close window      : Exit")
    print("=" * 65)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [-0.15, 0.75, 0.70]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -15.0

        sim_time = 0.0

        while viewer.is_running():
            if duration is not None and sim_time >= duration:
                break

            step_start = time.time()
            tau, f_meas, f_err, p_curr, _, state = controller.compute_torques(sim_time)
            data.ctrl[:] = tau

            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep

            viewer.sync()
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(
        description="UR10e Hybrid Position/Force Sanding Control in MuJoCo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Execution & Display
    mode_group = parser.add_argument_group("Execution & Display")
    mode_group.add_argument("--headless", action="store_true",
                            help="Run in headless benchmark mode without GUI")
    mode_group.add_argument("--plot", action="store_true",
                            help="Generate performance graph and save to PNG")
    mode_group.add_argument("--plot_name", type=str, default="sanding_performance.png",
                            help="Filename for the generated performance plot")
    mode_group.add_argument("--duration", type=float, default=None,
                            help="Simulation duration in seconds (default: 12.0s for headless, window-close for GUI)")

    # Force & Approach
    force_group = parser.add_argument_group("Force & Contact Parameters")
    force_group.add_argument("--force", "-f", type=float, default=1.0,
                             help="Target normal pressing force against wall (N)")
    force_group.add_argument("--offset", type=float, default=0.10,
                             help="Initial standoff offset distance from wall in meters")
    force_group.add_argument("--v_approach", type=float, default=0.01,
                             help="Normal approach velocity towards wall (m/s)")
    force_group.add_argument("--f_thresh", type=float, default=0.30,
                             help="Force threshold to detect wall contact (N)")
    force_group.add_argument("--kp_f", type=float, default=0.70,
                             help="Force PI controller proportional gain")
    force_group.add_argument("--ki_f", type=float, default=1.80,
                             help="Force PI controller integral gain")
    force_group.add_argument("--kd_f", type=float, default=3.50,
                             help="Force controller normal velocity damping gain")

    # Sweeping Pattern & Gains
    sweep_group = parser.add_argument_group("Sweeping Trajectory & Stiffness")
    sweep_group.add_argument("--freq_x", type=float, default=0.04,
                             help="Horizontal sweep frequency (Hz)")
    sweep_group.add_argument("--freq_z", type=float, default=0.02,
                             help="Vertical scan frequency (Hz)")
    sweep_group.add_argument("--ax", type=float, default=0.04,
                             help="Horizontal sweep amplitude (m)")
    sweep_group.add_argument("--az", type=float, default=0.02,
                             help="Vertical scan amplitude (m)")
    sweep_group.add_argument("--kp_p", type=float, default=40.0,
                             help="Tangential position proportional gain (N/m)")
    sweep_group.add_argument("--kd_p", type=float, default=10.0,
                             help="Tangential velocity damping (Ns/m)")
    sweep_group.add_argument("--kp_tilt", type=float, default=100.0,
                             help="Rotational tilt stiffness for pitch and yaw (Nm/rad)")
    sweep_group.add_argument("--kd_tilt", type=float, default=12.0,
                             help="Rotational tilt damping (Nms/rad)")

    args = parser.parse_args()

    dur = args.duration if args.duration is not None else 12.0

    if args.headless or args.plot:
        run_headless(
            duration=dur, plot=args.plot, offset=args.offset,
            force=args.force, v_approach=args.v_approach, f_thresh=args.f_thresh,
            freq_x=args.freq_x, freq_z=args.freq_z, ax=args.ax, az=args.az,
            kp_f=args.kp_f, ki_f=args.ki_f, kd_f=args.kd_f,
            kp_p=args.kp_p, kd_p=args.kd_p, kp_tilt=args.kp_tilt, kd_tilt=args.kd_tilt,
            plot_name=args.plot_name
        )
    else:
        try:
            run_gui(
                duration=args.duration, offset=args.offset,
                force=args.force, v_approach=args.v_approach, f_thresh=args.f_thresh,
                freq_x=args.freq_x, freq_z=args.freq_z, ax=args.ax, az=args.az,
                kp_f=args.kp_f, ki_f=args.ki_f, kd_f=args.kd_f,
                kp_p=args.kp_p, kd_p=args.kd_p, kp_tilt=args.kp_tilt, kd_tilt=args.kd_tilt
            )
        except Exception as e:
            print(f"GUI launch exception (falling back to headless benchmark): {e}")
            run_headless(
                duration=dur, plot=True, offset=args.offset,
                force=args.force, v_approach=args.v_approach, f_thresh=args.f_thresh,
                freq_x=args.freq_x, freq_z=args.freq_z, ax=args.ax, az=args.az,
                kp_f=args.kp_f, ki_f=args.ki_f, kd_f=args.kd_f,
                kp_p=args.kp_p, kd_p=args.kd_p, kp_tilt=args.kp_tilt, kd_tilt=args.kd_tilt,
                plot_name=args.plot_name
            )


if __name__ == "__main__":
    main()
