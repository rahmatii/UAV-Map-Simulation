import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
import math

from pywmm import WMMv2

from datetime import datetime
from typing import List, Optional

def decimal_year(date_string):
    """Convert 'YYYY-MM-DD' to decimal year"""
    dt = datetime.strptime(date_string, "%Y-%m-%d")
    year = dt.year
    start_of_year = datetime(year, 1, 1)
    end_of_year = datetime(year + 1, 1, 1)
    days_in_year = (end_of_year - start_of_year).days
    day_of_year = (dt - start_of_year).days
    return year + day_of_year / days_in_year


def mag_gen(ref_mag, mag_err):
    """
    Add error to magnetic data.
    Args:
        ref_mag: nx3 true magnetic data, uT.
        mag_err: Magnetometer error parameters.
            'si': 3x3 soft iron matrix
            'hi': hard iron array, [ox, oy, oz], uT
            'std': RMS of magnetometer noise, uT
    Returns:
        mag_mea: ref_mag with error, mag_mea = si * (ref_mag + hi) + noise
    """
    # total data count
    n = ref_mag.shape[0]
    # add error
    mag_mea = ref_mag + mag_err['hi']
    mag_mea = mag_mea.dot(mag_err['si'].T)
    mag_noise = mag_err['std'] * np.random.randn(n, 3)
    return mag_mea + mag_noise


def magnetometer_reading_batch(
    latitudes: List[float],
    longitudes: List[float],
    altitudes: List[float],
    yaws_rad: List[float],
    pitches_rad: List[float],
    rolls_rad: List[float],
    date: Optional[str] = None
) -> np.ndarray:
    """
    محاسبه دسته‌ای میدان مغناطیسی با مدل WMM برای چندین لحظه مختلف
    
    خروجی:
        آرایه سه‌بعدی با شکل (N, 3) شامل میدان مغناطیسی در دستگاه بدنه پهپاد
    """
    mag_low_accuracy = {'si': np.eye(3) + np.random.randn(3, 3)*0.0,
                        'hi': np.array([10.0, 10.0, 10.0])*0.0,
                        'std': np.array([0.1, 0.1, 0.1])}
    
    n = len(latitudes)
    if not (len(longitudes) == n and len(altitudes) == n and 
            len(yaws_rad) == n and len(pitches_rad) == n and len(rolls_rad) == n):
        raise ValueError("تمامی لیست‌های ورودی باید هم اندازه باشند")
    
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    decimal_year_value = decimal_year(date)
    wmm = WMMv2()
    
    results = np.zeros((n, 3))
    
    for i in range(n):
        alt_km = altitudes[i] / 1000.0
        
        B_N = wmm.get_north_intensity(latitudes[i], longitudes[i], decimal_year_value, alt_km)
        B_E = wmm.get_east_intensity(latitudes[i], longitudes[i], decimal_year_value, alt_km)
        B_D = wmm.get_vertical_intensity(latitudes[i], longitudes[i], decimal_year_value, alt_km)
        
        B_ned = np.array([B_N, B_E, B_D])
        
        yaw = yaws_rad[i]
        pitch = pitches_rad[i]
        roll = rolls_rad[i]
        
        R_yaw = np.array([
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R_pitch = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])
        
        R_NED_to_body = R_roll @ R_pitch @ R_yaw
        results[i] = R_NED_to_body @ B_ned
    
    # ✅ اعمال نویز و برگرداندن مقدار تصحیح شده
    mag_mea = mag_gen(results, mag_low_accuracy)
    return mag_mea

def bias_drift_gm(corr_time, drift, n, fs):
     sensor_bias_drift = np.zeros((n, 3))
     for i in range(0, 3):
         if not math.isinf(corr_time[i]):
             a = 1 - 1/fs/corr_time[i]
             b = drift[i] * np.sqrt(1.0 - np.exp(-2/(fs * corr_time[i])))
             drift_noise = np.random.randn(n, 3)
             for j in range(1, n):
                 sensor_bias_drift[j, i] = a*sensor_bias_drift[j-1, i] + b*drift_noise[j-1, i]
         else:
             sensor_bias_drift[:, i] = drift[i] * np.random.randn(n)
     return sensor_bias_drift

class IMUSimulator:
    def __init__(self, path, orientation,dt):
        self.path = path
        self.orientation = orientation
        self.dt = dt
    
   


    def generate_imu_data(self):
        
        
        
        N = len(self.orientation)
        
        gyro = []
        rotations = [Rot.from_euler('xyz', rpy) for rpy in self.orientation]

        for i in range(len(rotations) - 1):
            R1 = rotations[i]
            R2 = rotations[i + 1]

            delta_R = R1.inv() * R2 
            omega = delta_R.as_rotvec() / self.dt  
            gyro.append(omega)
            
        gyro.append(omega)
        gyro = np.array(gyro)

        velocity = np.diff(self.path, axis=0) / self.dt
        velocity = np.vstack([velocity,velocity[-1]])
        acc = np.diff(velocity, axis=0) / self.dt
        acc = np.vstack([acc,acc[-1]])
        g = 9.80665
        acc[:,2] += g  #  گرانش
        
        acc_body = np.zeros_like(acc)
        
        for i in range(N-1):
            roll, pitch, yaw = self.orientation[i]

            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])

            Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])

            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
            R = Rz @ Ry @ Rx  

            
            acc_body[i] = R.T @ acc[i]
        
 
        
        gyro_bias = np.array([0.0, 0, 0])
        random_walk_std =  np.array([3.6e-6, 3.6e-6, 3.6e-6])  # انحراف معیار Random Walk

        random_walk_acc = bias_drift_gm(np.array([100.0, 100.0, 100.0]), random_walk_std, N, N/10)
        accel_noise_std = 2.5 / 60
        acc_noise = np.random.randn(N, 3)
        acc_noise = accel_noise_std / math.sqrt(10/N) * acc_noise
        accel_bias = np.array([0.0, -0.0, 0.0])
        
        acc_bodyN = acc_body + acc_noise + random_walk_acc + accel_bias
        
        
        D2R = math.pi/180
        gyro_noise_std = 0.01# انحراف معیار نویز سفید
        gyro_noise = np.random.randn(N, 3)
        gyro_noise[:, 0] = gyro_noise_std / math.sqrt(10/N) * gyro_noise[:, 0]
        gyro_noise[:, 1] = gyro_noise_std / math.sqrt(10/N) * gyro_noise[:, 1]
        gyro_noise[:, 2] = gyro_noise_std / math.sqrt(10/N) * gyro_noise[:, 2]
        random_walk_std_gyro = np.array([0.1, 0.1, 0.1]) * D2R/3600.0
        random_walk_gyro = bias_drift_gm(np.array([100.0, 100.0, 100.0]), random_walk_std_gyro, N, N/10)
        gyroN = gyro + gyro_noise + gyro_bias+ random_walk_gyro
        
        return acc_body, gyro , acc_bodyN , gyroN 


def gps_to_meters(gps_trajectory: list[np.ndarray]) -> np.ndarray:
    """
    تبدیل تراژکتوری GPS به متر نسبت به اولین نقطه (مبدا)
 
    Args:
        gps_trajectory: لیستی از آرایه‌های [lat, lon, alt] بر حسب درجه و متر
 
    Returns:
        آرایه‌ای به شکل (N, 3) بر حسب متر [x(east), y(north), z(up)]
    """
    traj = np.array(gps_trajectory)  # شکل: (N, 3)
 
    lat = traj[:, 0]
    lon = traj[:, 1]
    alt = traj[:, 2]

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    lat0, lon0, alt0 = lat_r[0], lon_r[0], alt[0]

    R_earth = 6378137.0          # شعاع بزرگ WGS-84 (متر)
    e2 = 0.00669437999014        # مربع خروج از مرکز اول WGS-84
    N = R_earth / np.sqrt(1 - e2 * np.sin(lat0) ** 2)

    dlat = lat_r - lat0
    dlon = lon_r - lon0
    dalt = alt - alt0

    x = (N + alt0) * np.cos(lat0) * dlon      # East
    y = (N * (1 - e2) + alt0) * dlat          # North
    z = dalt                                # بالا/پایین
 
    return np.column_stack([x, y, z])
 
      
def main(positions , ground_truth_orientation , step_size , output_dir,t):

    positions = np.array(positions)                        # شکل: (N, 3)
    ground_truth_orientation = np.array(ground_truth_orientation)  # شکل: (N, 3)
    
    true_path = gps_to_meters(positions)
       # محاسبه میدان مغناطیسی

    today = datetime.now().strftime("%Y-%m-%d")
    mag_data = magnetometer_reading_batch(
        latitudes=positions[:, 0].tolist(),
        longitudes=positions[:, 1].tolist(),
        altitudes=positions[:, 2].tolist(),
        yaws_rad=ground_truth_orientation[:, 2].tolist(),
        pitches_rad=ground_truth_orientation[:, 1].tolist(),
        rolls_rad=ground_truth_orientation[:, 0].tolist(),
        date=today
    )
    
    simulator = IMUSimulator(true_path, ground_truth_orientation,step_size)
    acc_data, gyro_data,accN_data, gyroN_data = simulator.generate_imu_data()
    
    output = np.column_stack([t,positions,ground_truth_orientation,acc_data,gyro_data,accN_data,gyroN_data,mag_data])

    columns = ['ts','lat_deg','lon_deg','alt_m', 'roll_rad', 'pitch_rad', 'yaw_rad','Ax','Ay','Az','Wx','Wy','Wz','Ax_noise'
               ,'Ay_noise','Az_noise','Wx_noise','Wy_noise','Wz_noise','mag_x','mag_y','mag_z']
    output_df = pd.DataFrame(output, columns=columns)
    output_df.to_excel(output_dir+"/Path_IMU.xlsx", index=False)
