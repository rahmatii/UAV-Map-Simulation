import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
import os


class PositionEstimator:
    def __init__(self, acc_body, ground_truth_orientation, gyro_data, dt,initial_param_path, gravity = 9.80665):
        self.acc_body = acc_body
        self.gyro_data = gyro_data
        self.orientation = ground_truth_orientation
        self.dt = dt
        self.initial_param_path = initial_param_path
        self.gravity = gravity
        self.n_samples = len(acc_body)

    def load_initial_velocity(self,csv_path):
        df = pd.read_csv(csv_path)
        return df[['speed_E_mps', 'speed_N_mps', 'speed_U_mps']].values   # (N, 3)

    def integrate_accelerationG(self):
        vel_all = self.load_initial_velocity(self.initial_param_path)
      
        velocity =vel_all[0] #np.zeros(3)
        position = np.zeros(3)
        positions = []

        theta = self.orientation[0]
        R = Rot.from_rotvec(theta).as_matrix()

        for i in range(self.n_samples - 2):
            wx, wy, wz = self.gyro_data[i]
            omega = np.array([wx, wy, wz])
            dR = Rot.from_rotvec(omega * self.dt).as_matrix()
            R = R @ dR

            global_acc = R @ self.acc_body[i]
            global_acc[2] -= self.gravity

            velocity += global_acc * self.dt
            position += velocity * self.dt
            positions.append(position.copy())

        return np.array(positions)

    def estimate_position(self):
        return self.integrate_accelerationG()

    def plot_results(self, true_path, clean_path, noisy_path):
        fig = plt.figure(figsize=(16, 7))

        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(true_path[:, 0], true_path[:, 1], true_path[:, 2],
                 'b-', linewidth=1.5, label='Ground Truth')
        ax1.plot(clean_path[:, 0], clean_path[:, 1], clean_path[:, 2],
                 'g--', linewidth=1.2, label='Clean IMU')
        ax1.plot(noisy_path[:, 0], noisy_path[:, 1], noisy_path[:, 2],
                 'r:', linewidth=1.2, label='Noisy IMU')
        ax1.scatter(*true_path[0], color='black', s=30, zorder=5, label='Start')
        ax1.set_title('3D Path Comparison')
        ax1.set_xlabel('East (m)')
        ax1.set_ylabel('North (m)')
        ax1.set_zlabel('Up (m)')
        ax1.legend(fontsize=8)

        # XY top-down view
        ax2 = fig.add_subplot(122)
        ax2.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=1.5, label='Ground Truth')
        ax2.plot(clean_path[:, 0], clean_path[:, 1], 'g--', linewidth=1.2, label='Clean IMU')
        ax2.plot(noisy_path[:, 0], noisy_path[:, 1], 'r:', linewidth=1.2, label='Noisy IMU')
        ax2.scatter(true_path[0, 0], true_path[0, 1], color='black', s=30, zorder=5, label='Start')
        ax2.set_title('Top-Down View (East–North Plane)')
        ax2.set_xlabel('East (m)')
        ax2.set_ylabel('North (m)')
        ax2.legend(fontsize=8)
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imu_path_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to: {save_path}")


def lla_to_enu(lat, lon, alt):
    """Convert lat/lon/alt (degrees, meters) to local ENU XYZ in meters,
    using the first point as the reference origin."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    lat0, lon0, alt0 = lat_r[0], lon_r[0], alt[0]

    R_earth = 6378137.0  # WGS-84 semi-major axis in meters
    e2 = 0.00669437999014  # WGS-84 first eccentricity squared
    N = R_earth / np.sqrt(1 - e2 * np.sin(lat0) ** 2)

    dlat = lat_r - lat0
    dlon = lon_r - lon0
    dalt = alt - alt0

    x = (N + alt0) * np.cos(lat0) * dlon          # East
    y = (N * (1 - e2) + alt0) * dlat              # North
    z = dalt                                        # Up

    return np.column_stack([x, y, z])


def load_data(filepath):
    df = pd.read_excel(filepath)

    required_cols = ['lat_deg', 'lon_deg', 'alt_m', 'roll_rad', 'pitch_rad', 'yaw_rad',
                     'Ax', 'Ay', 'Az', 'Wx', 'Wy', 'Wz',
                     'Ax_noise', 'Ay_noise', 'Az_noise',
                     'Wx_noise', 'Wy_noise', 'Wz_noise']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in file: {missing}")

    # Ground truth: convert GPS lat/lon/alt → local ENU meters
    true_path = lla_to_enu(df['lat_deg'].values, df['lon_deg'].values, df['alt_m'].values)

    # Build rotation vectors from Euler angles (roll, pitch, yaw)
    euler_angles = df[['roll_rad', 'pitch_rad', 'yaw_rad']].values
    rot_vecs = []
    for rpy in euler_angles:
        roll, pitch, yaw = rpy
        r = Rot.from_euler('xyz', [roll, pitch, yaw])
        rot_vecs.append(r.as_rotvec())
    ground_truth_orientation = np.array(rot_vecs)

    acc_clean = df[['Ax', 'Ay', 'Az']].values
    gyro_clean = df[['Wx', 'Wy', 'Wz']].values
    acc_noisy = df[['Ax_noise', 'Ay_noise', 'Az_noise']].values
    gyro_noisy = df[['Wx_noise', 'Wy_noise', 'Wz_noise']].values

    return true_path, ground_truth_orientation, acc_clean, gyro_clean, acc_noisy, gyro_noisy


def main():
    initial_param_path = "test7/trajectory.csv"
    input_file = "test7/Path_IMU.xlsx"
    dt = 1/100  

    print(f"Loading data from {input_file} ...")
    true_path, orientation, acc_clean, gyro_clean, acc_noisy, gyro_noisy = load_data(input_file)
    print(f"  Loaded {len(true_path)} samples.")

    print("Integrating clean IMU data ...")
    estimator_clean = PositionEstimator(acc_clean, orientation, gyro_clean, dt,initial_param_path)
    clean_path = estimator_clean.estimate_position()

    print("Integrating noisy IMU data ...")
    estimator_noisy = PositionEstimator(acc_noisy, orientation, gyro_noisy, dt , initial_param_path)
    noisy_path = estimator_noisy.estimate_position()

    # Align lengths for plotting
    n = min(len(true_path), len(clean_path), len(noisy_path))
    true_path = true_path[:n]
    clean_path = clean_path[:n]
    noisy_path = noisy_path[:n]

    print("Plotting results ...")
    estimator_clean.plot_results(true_path, clean_path, noisy_path)


if __name__ == "__main__":
    main()