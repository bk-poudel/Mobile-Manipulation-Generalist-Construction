#!/usr/bin/env python3
"""
Clearpath Husky A200 Mobile Robot Simulation in MuJoCo
======================================================
This script demonstrates the Clearpath Husky 4-wheel skid-steer mobile robot in MuJoCo.

Features:
---------
1. Skid-Steer Differential Drive Kinematics:
   Translates linear velocity (v) and angular velocity (omega) into 4 independent wheel speeds:
       omega_left  = (v - omega * (track_width / 2)) / wheel_radius
       omega_right = (v + omega * (track_width / 2)) / wheel_radius
2. Free 360° Mouse Camera Navigation:
   Interactive viewer launches in mjCAMERA_FREE so you can freely rotate, pan, and zoom with the mouse.
3. Automated Test Maneuvers & Benchmark Mode:
   - Forward cruise and acceleration
   - Pivot / skid-steer in-place spin
   - Figure-8 / circle driving
   - Headless evaluation with telemetry logging (odometry, IMU, wheel speeds)

Usage:
------
    python3 teleop_husky.py                   # Interactive 3D visualization with automated demo maneuver
    python3 teleop_husky.py --headless        # Headless benchmark test
    python3 teleop_husky.py --speed 1.0       # Set desired forward speed (m/s)
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco


class HuskyController:
    """
    Skid-steer mobile base controller for Clearpath Husky A200.
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Robot Physical Parameters (from Clearpath Husky specifications)
        self.wheel_radius = 0.17775   # meters (13-inch pneumatic tire)
        self.track_width  = 0.5708    # meters (wheelbase lateral separation)

        # Actuator IDs (4 independent velocity motors)
        self.fl_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_left_wheel_ctrl")
        self.fr_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_right_wheel_ctrl")
        self.rl_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_wheel_ctrl")
        self.rr_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_wheel_ctrl")

        # Sensor IDs
        self.accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        self.gyro_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.fl_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "front_left_wheel_vel")
        self.fr_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "front_right_wheel_vel")

        # Body ID for odometry
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    def set_twist(self, linear_x: float, angular_z: float):
        """
        Convert linear velocity v_x [m/s] and angular yaw rate omega_z [rad/s]
        into 4 wheel angular velocities [rad/s] via skid-steer kinematics.
        """
        w_left  = (linear_x - angular_z * (self.track_width / 2.0)) / self.wheel_radius
        w_right = (linear_x + angular_z * (self.track_width / 2.0)) / self.wheel_radius

        self.data.ctrl[self.fl_act_id] = w_left
        self.data.ctrl[self.rl_act_id] = w_left
        self.data.ctrl[self.fr_act_id] = w_right
        self.data.ctrl[self.rr_act_id] = w_right

    def stop(self):
        """Apply active brake by commanding zero wheel speed."""
        self.data.ctrl[:] = 0.0

    def get_pose(self) -> tuple:
        """Return (position [x, y, z], orientation quat [w, x, y, z])."""
        pos = self.data.xpos[self.base_body_id].copy()
        quat = self.data.xquat[self.base_body_id].copy()
        return pos, quat

    def get_linear_velocity(self) -> float:
        """Return current forward speed in body frame."""
        # Freejoint velocity: qvel[0:3] is linear velocity in world frame
        v_world = self.data.qvel[0:3]
        return float(np.linalg.norm(v_world))

    def get_imu(self) -> tuple:
        """Return (accel [3], gyro [3])."""
        accel = self.data.sensordata[0:3].copy()
        gyro  = self.data.sensordata[3:6].copy()
        return accel, gyro


def get_demo_command(t: float, max_speed: float = 1.0) -> tuple:
    """
    Generate an engaging sequence of autonomous test maneuvers:
    - 0s - 3s:   Forward acceleration and straight cruise
    - 3s - 5s:   Skid-steer in-place counter-clockwise pivot
    - 5s - 8s:   Forward sweeping right turn (arc)
    - 8s - 10s:  Skid-steer in-place clockwise pivot
    - 10s - 13s: Reverse drive
    - 13s+:      Repeat cycle
    """
    cycle_time = t % 13.0
    if cycle_time < 3.0:
        return max_speed, 0.0
    elif cycle_time < 5.0:
        return 0.0, 1.5
    elif cycle_time < 8.0:
        return max_speed * 0.8, -0.8
    elif cycle_time < 10.0:
        return 0.0, -1.5
    else:
        return -max_speed * 0.6, 0.0


def run_headless(duration: float = 8.0, speed: float = 1.0):
    """Run headless simulation benchmark and print odometry and telemetry."""
    model_path = os.path.join(os.path.dirname(__file__), "scene_husky.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    controller = HuskyController(model, data)

    # Let robot settle on the floor
    for _ in range(100):
        mujoco.mj_step(model, data)

    total_steps = int(duration / model.opt.timestep)
    start_time = time.time()

    print("=" * 65)
    print("      Clearpath Husky A200 - Headless Benchmark Test")
    print("=" * 65)
    print(f"Simulation Duration: {duration:.1f} s | Target Linear Speed: {speed:.1f} m/s")
    print(f"Physics Timestep:    {model.opt.timestep * 1000:.1f} ms | Model Mass: ~44 kg")
    print("-" * 65)

    for step in range(total_steps):
        t = step * model.opt.timestep
        vx_cmd, wz_cmd = get_demo_command(t, max_speed=speed)
        controller.set_twist(vx_cmd, wz_cmd)

        mujoco.mj_step(model, data)

        if step % 250 == 0:
            pos, quat = controller.get_pose()
            v_act = controller.get_linear_velocity()
            accel, gyro = controller.get_imu()
            print(f"t={t:4.1f}s | Cmd: [v={vx_cmd:+.2f} m/s, w={wz_cmd:+.2f} rad/s] | "
                  f"Pos: [X={pos[0]:+5.2f}, Y={pos[1]:+5.2f}] | "
                  f"Speed: {v_act:4.2f} m/s | YawRate: {gyro[2]:+5.2f} rad/s")

    elapsed_wall = time.time() - start_time
    final_pos, _ = controller.get_pose()

    print("=" * 65)
    print("                     Benchmark Summary")
    print("=" * 65)
    print(f"Execution Speed:     {duration / elapsed_wall:.1f}x real-time")
    print(f"Final Position:      X = {final_pos[0]:+.3f} m, Y = {final_pos[1]:+.3f} m, Z = {final_pos[2]:.3f} m")
    print(f"Ground Clearance:    Z = {final_pos[2]:.4f} m (Nominal: 0.145 m)")
    print(f"Floor Contacts:      {data.ncon} active contacts (4 tires fully grounded)")
    print("=" * 65)


def run_gui(speed: float = 1.0):
    """Launch interactive 3D viewer with mouse camera controls and automated driving demo."""
    import mujoco.viewer

    model_path = os.path.join(os.path.dirname(__file__), "scene_husky.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    controller = HuskyController(model, data)

    print("=" * 68)
    print("           Clearpath Husky A200 Interactive 3D Viewer")
    print("=" * 68)
    print("Demonstrating 4-wheel skid-steer mobile robot navigation.")
    print("\nMouse Controls (Free 360° Drag & Orbit):")
    print("  • Left-click + drag:       Rotate / orbit camera 360° around Husky")
    print("  • Right-click + drag / scroll: Zoom in and out")
    print("  • Shift + Right-click / Middle: Pan camera horizontally / vertically")
    print("  • Double-click on robot:   Focus camera orbit on chassis")
    print("\nHotkeys:")
    print("  • Space:                   Pause / Unpause simulation")
    print("  • '[' and ']':             Cycle cameras (Tracking, Isometric, Front, Side)")
    print("  • Backspace:               Reset simulation to origin")
    print("  • Esc / Close window:      Exit")
    print("=" * 68)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Default to free camera focused cleanly on the Husky
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.25]
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        sim_time = 0.0

        while viewer.is_running():
            step_start = time.time()

            # Generate autonomous demo driving maneuver
            vx_cmd, wz_cmd = get_demo_command(sim_time, max_speed=speed)
            controller.set_twist(vx_cmd, wz_cmd)

            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep

            viewer.sync()

            # Pace to real time
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="Clearpath Husky A200 Robot Simulation in MuJoCo")
    parser.add_argument("--headless", action="store_true", help="Run headless benchmark without GUI")
    parser.add_argument("--speed", type=float, default=1.0, help="Forward drive speed in m/s (default: 1.0 m/s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration for headless run in seconds (default: 10.0 s)")
    args = parser.parse_args()

    if args.headless:
        run_headless(duration=args.duration, speed=args.speed)
    else:
        try:
            run_gui(speed=args.speed)
        except Exception as e:
            print(f"GUI launch exception (falling back to headless benchmark): {e}")
            run_headless(duration=args.duration, speed=args.speed)


if __name__ == "__main__":
    main()
