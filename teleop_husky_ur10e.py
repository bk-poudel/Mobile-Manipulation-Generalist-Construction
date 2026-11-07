#!/usr/bin/env python3
"""
Clearpath Husky + Universal Robots UR10e Mobile Manipulator in MuJoCo
====================================================================
This script provides an interactive simulation, teleoperation, and demonstration
runner for the unified Clearpath Husky A200 + UR10e 6-DOF mobile manipulator.

Key Capabilities:
-----------------
1. Mobile Base Skid-Steer Kinematics:
   Translates linear velocity (v) and angular rate (omega) into 4 wheel velocities:
       omega_left  = (v - omega * (track_width / 2)) / wheel_radius
       omega_right = (v + omega * (track_width / 2)) / wheel_radius

2. UR10e 6-DOF Manipulation & Smooth Trajectory Interpolation:
   Integrated industrial pedestal mounting with pre-programmed postures:
   - STOW: Compact fold over rear deck for stable high-speed transit
   - READY: Upright forward-facing manipulation stance
   - REACH_FORWARD: Extended inspection / reach in front of chassis
   - REACH_SIDE: Lateral manipulation stance (90° yaw)
   - SCAN_SWEEP: Continuous coordinated mobile manipulation trajectory

3. Interactive 360° Mouse Navigation:
   Default camera is mjCAMERA_FREE so you can freely rotate, pan, and zoom:
   - Left-click drag: Rotate 360°
   - Right-click drag: Zoom
   - Middle-click / Shift+Left drag: Pan

4. Multi-Phase Autonomous Demonstration or Keyboard Teleoperation:
   - Unfolds from stowed posture
   - Mobile base cruise with forward arm articulation
   - Skid-steer pivot turn with lateral side reach
   - Coordinated mobile scanning sweep
   - Safe re-stowing and braking

5. Headless Benchmark & Verification Mode:
   Run with `--headless` to verify physics, kinematics, and telemetry in CI/terminal.

Usage:
------
    python3 teleop_husky_ur10e.py                # Interactive 3D visualization & demo
    python3 teleop_husky_ur10e.py --headless     # Headless test & telemetry verification
    python3 teleop_husky_ur10e.py --duration 10  # Set run duration (seconds)
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco


# Pre-configured UR10e Postures (in radians: [pan, lift, elbow, wrist1, wrist2, wrist3])
ARM_POSTURES = {
    "stow": np.array([np.pi, -2.40, 2.40, -1.5708, -1.5708, 0.0]),
    "ready": np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
    "reach_forward": np.array([0.0, -0.785, 1.5708, -0.785, -1.5708, 0.0]),
    "reach_side": np.array([np.pi / 2.0, -1.0, 1.5708, -1.0, -1.5708, 0.0]),
    "scan_high": np.array([0.0, -1.2, 0.8, -1.2, -1.5708, 0.0]),
}


class HuskyUR10eController:
    """
    Unified controller for the Clearpath Husky mobile base and UR10e robotic arm.
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Physical Parameters (Clearpath Husky A200)
        self.wheel_radius = 0.17775  # 13-inch pneumatic tire
        self.track_width  = 0.5708   # Lateral wheel center separation

        # Actuator IDs (4 Wheels)
        self.fl_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_left_wheel_ctrl")
        self.fr_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_right_wheel_ctrl")
        self.rl_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_wheel_ctrl")
        self.rr_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_wheel_ctrl")
        self.wheel_act_ids = [self.fl_act_id, self.fr_act_id, self.rl_act_id, self.rr_act_id]

        # Actuator IDs (6 Arm Servos)
        arm_act_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        self.arm_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in arm_act_names]

        # Joint IDs (Arm)
        arm_jnt_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.arm_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in arm_jnt_names]
        self.arm_qpos_indices = [model.jnt_qposadr[jid] for jid in self.arm_jnt_ids]

        # Sensor IDs
        self.accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        self.gyro_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.tcp_pos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tcp_pos")
        self.tcp_quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "tcp_quat")

        # Body & Site IDs
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.tcp_site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        # Internal state for smooth arm trajectory interpolation
        self.current_arm_target = np.array(ARM_POSTURES["stow"], dtype=np.float64)
        self.traj_start_q = np.array(ARM_POSTURES["stow"], dtype=np.float64)
        self.traj_goal_q  = np.array(ARM_POSTURES["stow"], dtype=np.float64)
        self.traj_t0      = 0.0
        self.traj_duration = 1.0
        self.in_transition = False

    def set_base_twist(self, linear_x: float, angular_z: float):
        """
        Convert forward velocity linear_x [m/s] and angular yaw rate angular_z [rad/s]
        to 4 independent wheel angular speeds [rad/s] via skid-steer kinematics.
        """
        w_left  = (linear_x - angular_z * (self.track_width / 2.0)) / self.wheel_radius
        w_right = (linear_x + angular_z * (self.track_width / 2.0)) / self.wheel_radius

        self.data.ctrl[self.fl_act_id] = w_left
        self.data.ctrl[self.rl_act_id] = w_left
        self.data.ctrl[self.fr_act_id] = w_right
        self.data.ctrl[self.rr_act_id] = w_right

    def stop_base(self):
        """Zero wheel velocity setpoints."""
        for act_id in self.wheel_act_ids:
            self.data.ctrl[act_id] = 0.0

    def command_arm_posture(self, posture_name: str, duration: float = 2.0, current_sim_time: float = 0.0):
        """Trigger smooth minimum-jerk trajectory transition to a named posture."""
        if posture_name in ARM_POSTURES:
            target = ARM_POSTURES[posture_name]
            self.start_arm_trajectory(target, duration, current_sim_time)
        else:
            print(f"[HuskyUR10e] Unknown posture '{posture_name}'")

    def start_arm_trajectory(self, goal_q: np.ndarray, duration: float, current_sim_time: float):
        """Initialize smooth trajectory transition from current position."""
        self.traj_start_q = self.get_arm_qpos()
        self.traj_goal_q  = np.array(goal_q, dtype=np.float64)
        self.traj_t0      = current_sim_time
        self.traj_duration = max(0.1, duration)
        self.in_transition = True

    def update_arm(self, current_sim_time: float):
        """Evaluate trajectory interpolation and apply to arm position actuators."""
        if self.in_transition:
            tau = (current_sim_time - self.traj_t0) / self.traj_duration
            if tau >= 1.0:
                self.current_arm_target = self.traj_goal_q.copy()
                self.in_transition = False
            else:
                # Quintic polynomial (smooth velocity and acceleration at endpoints)
                s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
                self.current_arm_target = self.traj_start_q + s * (self.traj_goal_q - self.traj_start_q)

        # Apply target to UR10e actuators
        for i, act_id in enumerate(self.arm_act_ids):
            self.data.ctrl[act_id] = self.current_arm_target[i]

    def set_arm_posture_instant(self, q: np.ndarray):
        """Set arm target without trajectory smoothing."""
        self.current_arm_target = np.array(q, dtype=np.float64)
        self.in_transition = False
        for i, act_id in enumerate(self.arm_act_ids):
            self.data.ctrl[act_id] = self.current_arm_target[i]

    def get_arm_qpos(self) -> np.ndarray:
        """Return current arm joint positions [6]."""
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_indices])

    def get_base_pose(self) -> tuple:
        """Return (position [3], orientation quat [w, x, y, z], yaw [deg])."""
        pos = self.data.xpos[self.base_body_id].copy()
        quat = self.data.xquat[self.base_body_id].copy()
        qw, qx, qy, qz = quat
        yaw_rad = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return pos, quat, float(np.degrees(yaw_rad))

    def get_tcp_pos(self) -> np.ndarray:
        """Return end-effector / TCP 3D position in world frame."""
        return self.data.site_xpos[self.tcp_site_id].copy()

    def get_telemetry(self) -> dict:
        """Collect comprehensive telemetry for logging and verification."""
        pos, quat, yaw_deg = self.get_base_pose()
        v_world = self.data.qvel[0:3]
        speed = float(np.linalg.norm(v_world))
        tcp_pos = self.get_tcp_pos()
        arm_q = self.get_arm_qpos()

        accel = self.data.sensordata[0:3].copy()
        gyro  = self.data.sensordata[3:6].copy()

        return {
            "time": float(self.data.time),
            "base_x": pos[0],
            "base_y": pos[1],
            "base_z": pos[2],
            "yaw_deg": yaw_deg,
            "speed_mps": speed,
            "tcp_x": tcp_pos[0],
            "tcp_y": tcp_pos[1],
            "tcp_z": tcp_pos[2],
            "arm_q": arm_q,
            "imu_accel": accel,
            "imu_gyro": gyro,
        }


def get_autonomous_demo_command(t: float) -> tuple:
    """
    Multi-phase demonstration generator exercising base navigation & arm articulation:
    - [0.0 to  2.0s]: Arm unfolds from STOW to READY pose while base is stationary.
    - [2.0 to  5.0s]: Husky cruises forward at 0.7 m/s; arm extends forward (REACH_FORWARD).
    - [5.0 to  8.0s]: Husky performs skid-steer pivot turn; arm rotates to side (REACH_SIDE).
    - [8.0 to 11.0s]: Husky maneuvers in coordinated arc while arm executes a scanning sweep.
    - [11.0 to 14.0s]: Husky comes to a halt; arm smoothly re-stows (STOW).
    """
    cycle_time = 14.0
    phase_t = t % cycle_time

    if phase_t < 2.0:
        # Phase 0: Unfold to Ready
        v, omega = 0.0, 0.0
        posture = "ready"
    elif phase_t < 5.0:
        # Phase 1: Forward Cruise + Reach Forward
        v, omega = 0.7, 0.0
        posture = "reach_forward"
    elif phase_t < 8.0:
        # Phase 2: In-Place Skid-Steer Pivot + Reach Side
        v, omega = 0.0, 0.6
        posture = "reach_side"
    elif phase_t < 11.0:
        # Phase 3: Coordinated Arc Maneuver + High Scan
        v, omega = 0.5, 0.4
        posture = "scan_high"
    else:
        # Phase 4: Stop and Return to Stow
        v, omega = 0.0, 0.0
        posture = "stow"

    return v, omega, posture


def run_headless_benchmark(scene_path: str, duration: float = 5.0):
    """Headless simulation mode for physics verification and telemetry logging."""
    print("=" * 70)
    print("Clearpath Husky + UR10e Headless Benchmark Mode")
    print(f"Loading scene: {scene_path}")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    controller = HuskyUR10eController(model, data)

    # Initialize at stow posture
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    controller.set_arm_posture_instant(ARM_POSTURES["stow"])
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    total_steps = int(duration / dt)
    log_interval = int(0.5 / dt)

    t_start = time.perf_counter()
    last_posture = "stow"

    for step in range(total_steps):
        t_sim = float(data.time)

        # Autonomous command update
        v, omega, posture = get_autonomous_demo_command(t_sim)
        controller.set_base_twist(v, omega)

        if posture != last_posture:
            controller.command_arm_posture(posture, duration=1.5, current_sim_time=t_sim)
            last_posture = posture

        controller.update_arm(t_sim)
        mujoco.mj_step(model, data)

        # Telemetry Log
        if step % log_interval == 0 or step == total_steps - 1:
            telem = controller.get_telemetry()
            print(f"[t={telem['time']:5.2f}s] "
                  f"Base Pos: ({telem['base_x']:6.2f}, {telem['base_y']:6.2f}, {telem['base_z']:5.2f})m | "
                  f"Yaw: {telem['yaw_deg']:6.1f}° | "
                  f"Speed: {telem['speed_mps']:4.2f}m/s | "
                  f"TCP: ({telem['tcp_x']:6.2f}, {telem['tcp_y']:6.2f}, {telem['tcp_z']:5.2f})m | "
                  f"Target: {posture}")

    wall_time = time.perf_counter() - t_start
    realtime_factor = duration / max(1e-6, wall_time)
    print("=" * 70)
    print(f"Benchmark Complete: {duration:.2f}s simulated in {wall_time:.3f}s ({realtime_factor:.1f}x real-time)")
    print(f"Final Base Position : ({telem['base_x']:.3f}, {telem['base_y']:.3f}, {telem['base_z']:.3f}) m")
    print(f"Final TCP Position  : ({telem['tcp_x']:.3f}, {telem['tcp_y']:.3f}, {telem['tcp_z']:.3f}) m")
    print("=" * 70)


def run_interactive_simulation(scene_path: str):
    """
    Launch interactive 3D simulation with 360° mouse navigation, passive viewer,
    and automated demonstration with keyboard interaction support.
    """
    import mujoco.viewer

    print("=" * 70)
    print("Launching Clearpath Husky + UR10e Mobile Manipulator Viewer")
    print("=" * 70)
    print("Controls:")
    print("  - Mouse Navigation: Left-drag to rotate 360°, Right-drag to zoom, Middle-drag to pan")
    print("  - Spacebar: Pause / resume simulation")
    print("  - Tab: Cycle viewpoint cameras (isometric, tracking, closeup, front, side)")
    print("  - Keys 1-4: Manual Arm Posture (1: Stow, 2: Ready, 3: Reach Forward, 4: Reach Side)")
    print("  - Keys W/S: Manual Drive Forward/Back | A/D: Turn Left/Right | X: Stop")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    controller = HuskyUR10eController(model, data)

    # Initialize at stow posture
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    controller.set_arm_posture_instant(ARM_POSTURES["stow"])
    mujoco.mj_forward(model, data)

    # Keyboard / user override variables
    manual_override = False
    manual_override_timeout = 0.0
    manual_v, manual_omega = 0.0, 0.0
    last_posture = "stow"

    def key_callback(keycode):
        nonlocal manual_override, manual_override_timeout, manual_v, manual_omega, last_posture
        t_sim = float(data.time)

        # Posture triggers: '1', '2', '3', '4'
        if keycode == ord('1'):
            print("\n[Teleop] Arm Posture -> STOW")
            controller.command_arm_posture("stow", duration=1.8, current_sim_time=t_sim)
            last_posture = "stow"
        elif keycode == ord('2'):
            print("\n[Teleop] Arm Posture -> READY")
            controller.command_arm_posture("ready", duration=1.8, current_sim_time=t_sim)
            last_posture = "ready"
        elif keycode == ord('3'):
            print("\n[Teleop] Arm Posture -> REACH_FORWARD")
            controller.command_arm_posture("reach_forward", duration=1.8, current_sim_time=t_sim)
            last_posture = "reach_forward"
        elif keycode == ord('4'):
            print("\n[Teleop] Arm Posture -> REACH_SIDE")
            controller.command_arm_posture("reach_side", duration=1.8, current_sim_time=t_sim)
            last_posture = "reach_side"
        # Driving keys: 'W', 'S', 'A', 'D', 'X'
        elif keycode == ord('W') or keycode == ord('w'):
            manual_override = True
            manual_override_timeout = t_sim + 2.0
            manual_v = min(manual_v + 0.3, 1.2)
            print(f"\n[Teleop] Base Drive: v={manual_v:.2f} m/s, omega={manual_omega:.2f} rad/s")
        elif keycode == ord('S') or keycode == ord('s'):
            manual_override = True
            manual_override_timeout = t_sim + 2.0
            manual_v = max(manual_v - 0.3, -1.2)
            print(f"\n[Teleop] Base Drive: v={manual_v:.2f} m/s, omega={manual_omega:.2f} rad/s")
        elif keycode == ord('A') or keycode == ord('a'):
            manual_override = True
            manual_override_timeout = t_sim + 2.0
            manual_omega = min(manual_omega + 0.4, 1.5)
            print(f"\n[Teleop] Base Drive: v={manual_v:.2f} m/s, omega={manual_omega:.2f} rad/s")
        elif keycode == ord('D') or keycode == ord('d'):
            manual_override = True
            manual_override_timeout = t_sim + 2.0
            manual_omega = max(manual_omega - 0.4, -1.5)
            print(f"\n[Teleop] Base Drive: v={manual_v:.2f} m/s, omega={manual_omega:.2f} rad/s")
        elif keycode == ord('X') or keycode == ord('x'):
            manual_override = False
            manual_v, manual_omega = 0.0, 0.0
            controller.stop_base()
            print("\n[Teleop] Base Stopped. Resuming automated demo.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Set default view to free 360° mouse navigation
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -22.0
        viewer.cam.azimuth = 135.0
        viewer.cam.lookat = [0.0, 0.0, 0.6]

        sim_dt = model.opt.timestep

        while viewer.is_running():
            step_start = time.perf_counter()
            t_sim = float(data.time)

            # Determine control inputs: Manual teleop vs Autonomous demo
            if manual_override:
                if t_sim > manual_override_timeout:
                    manual_override = False
                    manual_v, manual_omega = 0.0, 0.0
                else:
                    controller.set_base_twist(manual_v, manual_omega)
            else:
                v, omega, posture = get_autonomous_demo_command(t_sim)
                controller.set_base_twist(v, omega)

                if posture != last_posture:
                    controller.command_arm_posture(posture, duration=1.6, current_sim_time=t_sim)
                    last_posture = posture

            controller.update_arm(t_sim)
            mujoco.mj_step(model, data)
            viewer.sync()

            # Maintain natural real-time frame rate
            time_until_next_step = sim_dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main():
    parser = argparse.ArgumentParser(description="Clearpath Husky + UR10e Mobile Manipulator Simulation")
    parser.add_argument("--scene", type=str, default="scene_husky_ur10e.xml",
                        help="Path to the scene XML file (default: scene_husky_ur10e.xml)")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless benchmark mode without 3D window")
    parser.add_argument("--duration", type=float, default=6.0,
                        help="Duration to simulate in headless mode (seconds, default: 6.0)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(script_dir, args.scene) if not os.path.isabs(args.scene) else args.scene

    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found at '{scene_path}'")
        sys.exit(1)

    if args.headless:
        run_headless_benchmark(scene_path, duration=args.duration)
    else:
        run_interactive_simulation(scene_path)


if __name__ == "__main__":
    main()
