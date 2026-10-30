import numpy as np
import os
import rasterio
import cv2
from scipy.spatial.transform import Rotation as Rot
from rasterio.warp import transform
import time
import math
from rasterio.windows import Window
import GenSensorData


def calculate_pixel_size(fov_degrees, image_size_pixels, altitude_meters):
    """
    محاسبه اندازه هر پیکسل تصویر بر حسب متر
    
    پارامترها:
    fov_degrees: میدان دید دوربین بر حسب درجه (مربعی فرض شده)
    image_size_pixels: اندازه تصویر بر حسب پیکسل (مربعی)
    altitude_meters: ارتفاع پرواز پهباد بر حسب متر
    
    بازگشت:
    اندازه هر پیکسل بر حسب متر
    """
    # تبدیل زاویه FOV از درجه به رادیان
    fov_radians = math.radians(fov_degrees)
    
    # محاسبه عرض میدان دید واقعی در سطح زمین
    # با فرض اینکه دوربین عمود بر زمین است
    fov_meters = 2 * altitude_meters * math.tan(fov_radians / 2)
    
    # محاسبه اندازه هر پیکسل
    pixel_size_meters = fov_meters / image_size_pixels
    
    return pixel_size_meters


# =========================================================
# UTM ↔ GPS  (بدون pyproj / utm)
# =========================================================

def _utm_zone_from_lon(lon: float) -> int:
    return int((lon + 180) / 6) + 1


def gps_to_utm_manual(lat: float, lon: float):
    """
    تبدیل WGS-84 (درجه) به UTM (متر)
    بازگشت: (easting, northing, zone_number, hemisphere)
    """
    a = 6378137.0          # نیم‌محور بزرگ
    f = 1 / 298.257223563  # تخت‌شدگی
    b = a * (1 - f)
    e2 = 1 - (b / a) ** 2
    e_prime2 = e2 / (1 - e2)

    k0 = 0.9996
    E0 = 500000.0

    zone = _utm_zone_from_lon(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)  # مرکز zone

    lat_r = math.radians(lat)
    lon_r = math.radians(lon)

    N = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
    T = math.tan(lat_r) ** 2
    C = e_prime2 * math.cos(lat_r) ** 2
    A = math.cos(lat_r) * (lon_r - lon0)

    M = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_r
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_r)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_r)
        - (35 * e2**3 / 3072) * math.sin(6 * lat_r)
    )

    easting = k0 * N * (
        A
        + (1 - T + C) * A**3 / 6
        + (5 - 18 * T + T**2 + 72 * C - 58 * e_prime2) * A**5 / 120
    ) + E0

    northing = k0 * (
        M
        + N * math.tan(lat_r) * (
            A**2 / 2
            + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
            + (61 - 58 * T + T**2 + 600 * C - 330 * e_prime2) * A**6 / 720
        )
    )

    hemisphere = 'north' if lat >= 0 else 'south'
    if hemisphere == 'south':
        northing += 10_000_000.0  # false northing جنوب

    return easting, northing, zone, hemisphere


def utm_to_gps_manual(easting: float, northing: float, zone: int, hemisphere: str):
    """
    تبدیل UTM (متر) به WGS-84 (درجه)
    بازگشت: (lat, lon)
    """
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 1 - (b / a) ** 2
    e_prime2 = e2 / (1 - e2)

    k0 = 0.9996
    E0 = 500000.0

    if hemisphere == 'south':
        northing -= 10_000_000.0

    x = easting - E0
    M = northing / k0

    mu = M / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))

    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
            + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
            + (151 * e1**3 / 96) * math.sin(6 * mu)
            + (1097 * e1**4 / 512) * math.sin(8 * mu))

    N1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    T1 = math.tan(phi1) ** 2
    C1 = e_prime2 * math.cos(phi1) ** 2
    R1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    D = x / (N1 * k0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D**2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1**2 - 9 * e_prime2) * D**4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1**2 - 252 * e_prime2 - 3 * C1**2) * D**6 / 720
    )

    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    lon = lon0 + (
        D
        - (1 + 2 * T1 + C1) * D**3 / 6
        + (5 - 2 * C1 + 28 * T1 - 3 * C1**2 + 8 * e_prime2 + 24 * T1**2) * D**5 / 120
    ) / math.cos(phi1)

    return math.degrees(lat), math.degrees(lon)


# =========================================================
# UAVCameraSimulator
# =========================================================

class UAVCameraSimulator:

    def __init__(self, geotiff_path):
        self.geotiff_path = geotiff_path
        with rasterio.open(geotiff_path) as src:
            self.H = src.height
            self.W = src.width
            self.count = src.count
            self.transform = src.transform
            self.crs = src.crs

            raw_pixel_size = abs(self.transform.a)
            if self.crs.is_geographic:
                center_row = src.height / 2
                center_col = src.width / 2
                center_x, center_y = src.transform * (center_col, center_row)
                self.pixel_size = raw_pixel_size * 111320 * math.cos(math.radians(center_y))
                print(f"pixel_size: {raw_pixel_size:.8f} deg → {self.pixel_size:.4f} m/pixel")
            else:
                self.pixel_size = raw_pixel_size
                print(f"pixel_size: {self.pixel_size:.4f} m/pixel")

            print("Number of bands:", src.count)
            print("Map shape:", (self.H, self.W, self.count if self.count >= 3 else 3))
            print(f"Memory if loaded: {self.H * self.W * 3 / (1024**3):.1f} GB (uint8)")

            if self.count >= 3:
                self.band_mins = []
                self.band_maxs = []
                for band in [1, 2, 3]:
                    stats = src.statistics(band)
                    self.band_mins.append(stats.min)
                    self.band_maxs.append(stats.max)
            else:
                stats = src.statistics(1)
                self.band_mins = [stats.min]
                self.band_maxs = [stats.max]

        # تشخیص zone و hemisphere از مرکز نقشه
        center_lat, center_lon = self.pixel_to_gps(self.H / 2, self.W / 2)
        self._utm_zone = _utm_zone_from_lon(center_lon)
        self._utm_hemisphere = 'north' if center_lat >= 0 else 'south'

        epsg_utm = (32600 if self._utm_hemisphere == 'north' else 32700) + self._utm_zone
        print(f"UTM zone: {self._utm_zone} ({self._utm_hemisphere}), EPSG:{epsg_utm}")

    # ------------------------------------------------------------------
    # تبدیل مختصات
    # ------------------------------------------------------------------

    def pixel_to_gps(self, row, col):
        with rasterio.open(self.geotiff_path) as src:
            x, y = src.transform * (col, row)
            lon, lat = transform(src.crs, 'EPSG:4326', [x], [y])
            return float(lat[0]), float(lon[0])

    def gps_to_pixel(self, lat, lon):
        with rasterio.open(self.geotiff_path) as src:
            x, y = transform('EPSG:4326', src.crs, [lon], [lat])
            col, row = ~src.transform * (x[0], y[0])
            return row, col

    def gps_to_utm(self, lat, lon, alt):
        """GPS (درجه) → UTM (متر)  — بدون pyproj"""
        easting, northing, _, _ = gps_to_utm_manual(lat, lon)
        return easting, northing,alt   # x=easting, y=northing

    def utm_to_gps(self, x_utm, y_utm):
        """UTM (متر) → GPS (درجه)  — بدون pyproj"""
        lat, lon = utm_to_gps_manual(x_utm, y_utm,
                                     self._utm_zone, self._utm_hemisphere)
        return lat, lon

    # ------------------------------------------------------------------
    # ابزار هندسی
    # ------------------------------------------------------------------

    def get_ground_distance_between_pixels(self, row1, col1, row2, col2):
        lat1, lon1 = self.pixel_to_gps(row1, col1)
        lat2, lon2 = self.pixel_to_gps(row2, col2)

        R = 6371000
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def calculate_trajectory_length(self, start_row, start_col, end_row, end_col):
        return self.get_ground_distance_between_pixels(start_row, start_col, end_row, end_col)

    # ------------------------------------------------------------------
    # خواندن patch از GeoTIFF
    # ------------------------------------------------------------------

    def read_patch(self, x_min, y_min, x_max, y_max):
        with rasterio.open(self.geotiff_path) as src:
            x_min = max(0, min(x_min, self.W - 1))
            x_max = max(x_min + 1, min(x_max, self.W))
            y_min = max(0, min(y_min, self.H - 1))
            y_max = max(y_min + 1, min(y_max, self.H))

            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                return None

            window = Window(x_min, y_min, width, height)

            if self.count >= 3:
                patch = src.read([1, 2, 3], window=window)
                patch = np.transpose(patch, (1, 2, 0))
            else:
                gray = src.read(1, window=window)
                patch = np.stack([gray, gray, gray], axis=2)

            for c in range(patch.shape[2]):
                mn = self.band_mins[c] if c < len(self.band_mins) else 0
                mx = self.band_maxs[c] if c < len(self.band_maxs) else 255
                if mx > mn:
                    patch[:, :, c] = (patch[:, :, c] - mn) * 255.0 / (mx - mn)

            return np.clip(patch, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # دوربین
    # ------------------------------------------------------------------

    def create_camera_matrix(self, width, height, fov_deg):
        f = width / (2 * np.tan(np.radians(fov_deg) / 2))
        return np.array([
            [f, 0, width / 2],
            [0, f, height / 2],
            [0, 0, 1]
        ])

    def get_four_corners_on_ground(self, position_gps, orientation, width, height, fov):
        K = self.create_camera_matrix(width, height, fov)
        R = Rot.from_euler('zyx', orientation).as_matrix()

        position_utm = position_gps

        corners_pixel = []
        for u, v in [(0, 0), (width, 0), (width, height), (0, height)]:
            ray_cam = np.array([
                (u - width / 2) / K[0, 0],
                (v - height / 2) / K[1, 1],
                1.0
            ])
            ray_cam /= np.linalg.norm(ray_cam)
            R_cb = Rot.from_euler('x', 180, degrees=True).as_matrix() 
            ray_world = R @ R_cb @ ray_cam
            

            if ray_world[2] >= 0:
                return None

            t = -position_utm[2] / ray_world[2]
            gp_utm = position_utm + t * ray_world

            corner_lat, corner_lon = self.utm_to_gps(gp_utm[0], gp_utm[1])
            corner_row, corner_col = self.gps_to_pixel(corner_lat, corner_lon)
            corners_pixel.append([corner_col, corner_row])

        return np.array(corners_pixel, dtype=np.float32)

    def render_fast(self, position_gps, orientation, width=512, height=512, fov=90):
        SCALE = 1
        corners_ground = self.get_four_corners_on_ground(
            position_gps, orientation, width, height, fov
        )

        if corners_ground is None:
            print("Camera looking upward!")
            return np.zeros((height, width, 3), dtype=np.uint8)

        min_x = np.min(corners_ground[:, 0])
        max_x = np.max(corners_ground[:, 0])
        min_y = np.min(corners_ground[:, 1])
        max_y = np.max(corners_ground[:, 1])

        margin = max(50, int(0.1 * max(max_x - min_x, max_y - min_y)))
        min_x = max(0, min_x - margin)
        max_x = min(self.W, max_x + margin)
        min_y = max(0, min_y - margin)
        max_y = min(self.H, max_y + margin)

        if min_x >= max_x or min_y >= max_y:
            print("Camera view is outside the map")
            return np.zeros((height, width, 3), dtype=np.uint8)

        map_patch = self.read_patch(
            int(np.floor(min_x)), int(np.floor(min_y)),
            int(np.ceil(max_x)), int(np.ceil(max_y))
        )
        if map_patch is None:
            return np.zeros((height, width, 3), dtype=np.uint8)

        corners_rel = corners_ground - [min_x, min_y]
        corners_out = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]],
            dtype=np.float32
        )

        H_mat, _ = cv2.findHomography(corners_rel, corners_out)
        if H_mat is None:
            return np.zeros((height, width, 3), dtype=np.uint8)

        H_large = H_mat  # همان محتوا، در فضای 1024
        try:
            output_large = cv2.warpPerspective(
                map_patch, H_large, (width * SCALE, height * SCALE),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            return output_large
        except Exception as e:
            print(f"Warp perspective failed: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)


# =========================================================
# MAIN
# =========================================================

def main(gps_trajectory , roll_rad, pitch_rad, yaw_rad,t,fps,flight_height_meters,num_points,output_dir,image_size,fov,geoPath):
    fov = fov  # درجه
    image_size = image_size  # پیکسل (تصویر مربعی 1024x1024)
    sim = UAVCameraSimulator(geoPath)
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("🛸 INITIAL POSITION CALCULATION")
    print("=" * 70)
   
    GenSensorData.main(gps_trajectory, np.column_stack([roll_rad, pitch_rad, yaw_rad]), 1/fps , output_dir,t)
    alt_array = np.array([p[2] for p in gps_trajectory])
    

    print(f"✅ Generated {len(gps_trajectory)} GPS points")
    print(f"  Start GPS: ({gps_trajectory[0][0]:.8f}, {gps_trajectory[0][1]:.8f}, {gps_trajectory[0][2]:.1f}m)")
    print(f"  End GPS:   ({gps_trajectory[-1][0]:.8f}, {gps_trajectory[-1][1]:.8f}, {gps_trajectory[-1][2]:.1f}m)")
    print(f"  Altitude range: [{alt_array.min():.1f}, {alt_array.max():.1f}] m")
    print("\n🔄 Converting GPS trajectory to pixel coordinates...")


    # pixel_trajectory = []
    # last_row, last_col = 0.0, 0.0

    # for i, gps_pos in enumerate(gps_trajectory):
    #     if i % 100 == 0:
    #         last_row, last_col = sim.gps_to_pixel(gps_pos[0], gps_pos[1])
    #     pixel_trajectory.append((float(last_row), float(last_col)))


    print("\n🎬 Starting frame generation...")
    print("=" * 70)

    start_time_total = time.time()

    for i in range(0, num_points, 10):#int(fps/30)
        
        gps_pos = sim.gps_to_utm(gps_trajectory[i][0],gps_trajectory[i][1],gps_trajectory[i][2] ) 
        ori = np.array([yaw_rad[i], pitch_rad[i], roll_rad[i]])

        if i % max(1, num_points // 20) == 0:
            progress = (i / num_points) * 100
            print(f"\n📸 Frame: {i:05d}/{num_points} ({progress:.1f}%)")
            print(f"  GPS Position: ({gps_pos[0]:.8f}, {gps_pos[1]:.8f}, {gps_pos[2]:.1f}m)")

        start_time = time.time()
        sim._utm_zone = _utm_zone_from_lon(gps_trajectory[i][1])
        img = sim.render_fast(
            position_gps=gps_pos,
            orientation=ori,
            width=image_size, height=image_size, fov=fov
        )
  

        
        elapsed_time = time.time() - start_time


        filename = os.path.join(output_dir, f"frame_{(i+120000):05d}.png")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if i % max(1, num_points // 20) == 0:
            print(f"  Render time: {elapsed_time:.3f}s  ->  Saved: {filename}")

    total_time = time.time() - start_time_total

    print("\n" + "=" * 70)
    print("✅ ALL FRAMES GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total frames: {int(num_points)/10}")
    print(f"⏱️  Total render time: {total_time:.2f}s")
    print(f"⏱️  Average time per frame: {total_time/num_points:.3f}s")
    print(f"🎬 Effective FPS: {num_points/total_time:.2f}")
    print(f"\n📍 Start GPS: ({gps_trajectory[0][0]:.8f}, {gps_trajectory[0][1]:.8f}, {gps_trajectory[0][2]:.1f}m)")
    print(f"📍 End GPS:   ({gps_trajectory[-1][0]:.8f}, {gps_trajectory[-1][1]:.8f}, {gps_trajectory[-1][2]:.1f}m)")
