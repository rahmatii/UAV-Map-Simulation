"""
flight_path.py
══════════════
ورودی : waypoint‌های GPS + سرعت پایه + dt نمونه‌برداری
خروجی : مسیر C²-smooth با نرخ نمونه‌برداری ثابت روی بیضوی WGS-84

جریان کلی:
  1. GPS waypoints → CubicSpline C² روی (lat, lon, alt)
  2. arc-length واقعی روی بیضوی → flight_duration = path_len / speed
  3. t = [0, dt, 2dt, ...] با گام ثابت
  4. سرعت ENU از مشتق (lat,lon,alt) × شعاع‌های M و N بیضوی
"""

import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
from scipy.interpolate import CubicSpline

# ══════════════════════════════════════════════════════════════════
#  ثوابت WGS-84
# ══════════════════════════════════════════════════════════════════
_a  = 6_378_137.0           # نیم‌محور بزرگ (m)
_f  = 1.0 / 298.257223563   # تخت‌شدگی
_b  = _a * (1.0 - _f)
_e2 = 1.0 - (_b / _a) ** 2  # خروج‌از‌مرکز مربع


# ══════════════════════════════════════════════════════════════════
#  شعاع‌های انحنای بیضوی
# ══════════════════════════════════════════════════════════════════


def _M(lat_rad: float) -> float:
    """
    شعاع انحنا در راستای meridian (شمال-جنوب).
      M = a(1-e²) / (1 - e²sin²φ)^1.5
    رابطه با سرعت:
      v_N = dφ/dt · (M + h)
    """
    s2 = math.sin(lat_rad) ** 2
    return _a * (1.0 - _e2) / (1.0 - _e2 * s2) ** 1.5


def _N(lat_rad: float) -> float:
    """
    شعاع انحنا در راستای prime vertical (شرق-غرب).
      N = a / sqrt(1 - e²sin²φ)
    رابطه با سرعت:
      v_E = dλ/dt · (N + h) · cos(φ)
    """
    s2 = math.sin(lat_rad) ** 2
    return _a / math.sqrt(1.0 - _e2 * s2)


# ══════════════════════════════════════════════════════════════════
#  فاصله haversine — فقط برای knot spacing
# ══════════════════════════════════════════════════════════════════

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R  = 6_371_000.0
    f1 = math.radians(lat1); f2 = math.radians(lat2)
    df = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(df/2)**2 + math.cos(f1)*math.cos(f2)*math.sin(dl/2)**2
    return R * 2.0 * math.asin(math.sqrt(max(a, 0.0)))


# ══════════════════════════════════════════════════════════════════
#  CubicSpline C² روی (lat, lon, alt)
# ══════════════════════════════════════════════════════════════════

def _build_spline(wps: np.ndarray):
    """
    wps: (N, 3) — [lat°, lon°, alt_m]

    پارامتر u متناسب با فاصله واقعی بین waypoint‌ها:
      u[0] = 0
      u[i] = u[i-1] + hypot(haversine_i, dalt_i)

    این باعث می‌شود spline در نقاط نزدیک‌به‌هم overshoot نکند.
    bc_type='not-a-knot' → پایدارترین شرط مرزی برای interpolation.
    """
    n = len(wps)
    u = np.zeros(n)
    for i in range(1, n):
        d2d = _haversine(wps[i-1,0], wps[i-1,1], wps[i,0], wps[i,1])
        dh  = wps[i,2] - wps[i-1,2]
        u[i] = u[i-1] + math.hypot(d2d, dh)
    return CubicSpline(u, wps, bc_type='not-a-knot'), u


# ══════════════════════════════════════════════════════════════════
#  arc-length روی بیضوی و reparameterization
# ══════════════════════════════════════════════════════════════════

def _arclen_and_resample(cs, u_knots: np.ndarray,
                         n_fine: int, n_out: int):
    """
    مرحله ۱ — نمونه‌برداری متراکم از spline (n_fine نقطه).
    مرحله ۲ — محاسبه arc-length تجمعی روی بیضوی:
        ds = sqrt( (dφ·(M+h))² + (dλ·cosφ·(N+h))² + dh² )
    مرحله ۳ — نمونه‌برداری یکنواخت روی arc-length با np.interp.

    خروجی: (pts_out, total_len)
      pts_out: (n_out, 3) با فاصله یکنواخت arc-length
    """
    # مرحله ۱
    u_fine = np.linspace(u_knots[0], u_knots[-1], n_fine)
    pts    = cs(u_fine)   # (n_fine, 3)

    # مرحله ۲
    arc = np.zeros(n_fine)
    for i in range(1, n_fine):
        lat_r  = math.radians(pts[i-1, 0])
        h      = pts[i-1, 2]
        dlat_r = math.radians(pts[i,0] - pts[i-1,0])
        dlon_r = math.radians(pts[i,1] - pts[i-1,1])
        dh     = pts[i,2] - pts[i-1,2]
        ds = math.sqrt(
            (dlat_r * (_M(lat_r) + h)) ** 2 +
            (dlon_r * math.cos(lat_r) * (_N(lat_r) + h)) ** 2 +
            dh ** 2
        )
        arc[i] = arc[i-1] + ds

    total_len = arc[-1]

    # مرحله ۳
    s_uni   = np.linspace(0.0, total_len, n_out)
    pts_out = np.column_stack([
        np.interp(s_uni, arc, pts[:, k]) for k in range(3)
    ])
    return pts_out, total_len


# ══════════════════════════════════════════════════════════════════
#  تابع اصلی
# ══════════════════════════════════════════════════════════════════

def genPath_waypoints(
    waypoints_gps,            # list of (lon°, lat°)
    flight_height_meters,     # ارتفاع پایه h (متر)
    base_speed,               # سرعت پایه |V| (m/s)
    dt_sec       = 0.003,     # گام نمونه‌برداری زمانی (ثانیه)
    height_variation = 5.0,   # ±نوسان ارتفاع سینوسی (متر)
    n_fine       = 100_000,   # تراکم نقاط برای arc-length دقیق
):
    """
    خروجی (همه آرایه‌ها به طول num_points):
      gps_trajectory : list of np.array([lat°, lon°, alt_m])
      roll_rad       : np.ndarray — رادیان
      pitch_rad      : np.ndarray — رادیان
      yaw_rad        : np.ndarray — رادیان
      speed_E        : np.ndarray — سرعت شرقی  (m/s)
      speed_N        : np.ndarray — سرعت شمالی (m/s)
      speed_U        : np.ndarray — سرعت عمودی (m/s)
      t              : np.ndarray — زمان با گام ثابت dt_sec (s)
      speed_mag      : np.ndarray — قدر مطلق سرعت (m/s)
    """

    # ── ۱. آماده‌سازی waypoint‌ها ─────────────────────────────────
    wps = np.array([[lat, lon, float(flight_height_meters)]
                    for lon, lat in waypoints_gps])

    # ── ۲. ساخت spline C² ────────────────────────────────────────
    cs, u_knots = _build_spline(wps)

    # ── ۳. محاسبه طول مسیر (arc-length روی بیضوی) ────────────────
    # برای تعیین flight_duration و num_points به path_len نیاز داریم.
    # ابتدا با یک n_out موقت طول مسیر را می‌گیریم.
    _, path_len = _arclen_and_resample(cs, u_knots, n_fine, n_out=2)

    # ── ۴. زمان پرواز و تعداد نقاط از dt ─────────────────────────
    flight_duration = path_len / base_speed
    num_points      = int(round(flight_duration / dt_sec)) + 1
    t               = np.linspace(0.0, flight_duration, num_points)
    # گام واقعی (ممکن است کمی با dt_sec تفاوت داشته باشد)
    dt_actual = t[1] - t[0]

    # ── ۵. نمونه‌برداری نهایی با num_points واقعی ─────────────────
    sampled_pts, _ = _arclen_and_resample(cs, u_knots, n_fine, num_points)

    # ── ۶. ارتفاع با نوسان سینوسی ────────────────────────────────
    alt_profile = (flight_height_meters
                   + height_variation * np.sin(2.0 * np.pi * t / flight_duration))
    sampled_pts[:, 2] = alt_profile

    lats = sampled_pts[:, 0]
    lons = sampled_pts[:, 1]
    alts = sampled_pts[:, 2]

    # ── ۷. سرعت ENU ───────────────────────────────────────────────
    # مشتق مرکزی O(dt²) روی lat/lon/alt
    dlat_dt = np.gradient(lats, t)   # °/s
    dlon_dt = np.gradient(lons, t)   # °/s
    dalt_dt = np.gradient(alts, t)   # m/s

    speed_E = np.empty(num_points)
    speed_N = np.empty(num_points)
    speed_U = np.empty(num_points)

    for i in range(num_points):
        lat_r = math.radians(lats[i])
        h     = alts[i]
        # v_E = dλ/dt [rad/s] × (N+h) × cosφ
        speed_E[i] = math.radians(dlon_dt[i]) * (_N(lat_r) + h) * math.cos(lat_r)
        # v_N = dφ/dt [rad/s] × (M+h)
        speed_N[i] = math.radians(dlat_dt[i]) * (_M(lat_r) + h)
        # v_U = dh/dt
        speed_U[i] = dalt_dt[i]

    speed_mag = np.sqrt(speed_E**2 + speed_N**2 + speed_U**2)

    # ── ۸. orientation (مثل کد اصلی) ─────────────────────────────
    # ── ۸. orientation — واقعاً از روی مسیر ────────────────────────
    g = 9.80665

    speed_h = np.sqrt(speed_E**2 + speed_N**2)          # سرعت افقی

    # yaw: جهت حرکت (heading نسبت به شمال، رادیان، unwrap شده تا پرش ±π نداشته باشیم)
    yaw_rad = np.unwrap(np.arctan2(speed_E, speed_N))

    # pitch: زاویه بین بردار سرعت و صفحه افق
    pitch_rad = np.arctan2(speed_U, speed_h)

    # نرخ چرخش yaw برای محاسبه roll (coordinated turn assumption)
    yaw_rate = np.gradient(yaw_rad, t)
    roll_rad = np.arctan2(speed_h * yaw_rate, g)

    # ── ۹. گزارش ─────────────────────────────────────────────────
    print(f"Path length     : {path_len:.1f} m  ({path_len/1000:.3f} km)")
    print(f"Flight duration : {flight_duration:.2f} s  ({flight_duration/60:.2f} min)")
    print(f"dt actual       : {dt_actual*1000:.4f} ms  (requested {dt_sec*1000:.1f} ms)")
    print(f"num_points      : {num_points}")
    print(f"|V| range       : [{speed_mag.min():.3f}, {speed_mag.max():.3f}] m/s")
    print(f"Alt range       : [{alts.min():.1f}, {alts.max():.1f}] m")
    print("Waypoint proximity:")
    for i, (wlon, wlat) in enumerate(waypoints_gps):
        dists = [_haversine(wlat, wlon, lats[k], lons[k]) for k in range(num_points)]
        print(f"  WP{i+1}: {min(dists):.2f} m")

    gps_trajectory = [np.array([lats[i], lons[i], alts[i]])
                      for i in range(num_points)]
    return (gps_trajectory, 
            roll_rad, pitch_rad, yaw_rad,
            speed_E, speed_N, speed_U,
            t, speed_mag,num_points)


# ══════════════════════════════════════════════════════════════════
#  پلات نتایج
# ══════════════════════════════════════════════════════════════════

def plot_results(result, waypoints_gps, save_path="flight_path.png"):
    (gps_traj,
     roll_rad, pitch_rad, yaw_rad,
     speed_E, speed_N, speed_U,
     t, speed_mag,num_points) = result

    lats = np.array([p[0] for p in gps_traj])
    lons = np.array([p[1] for p in gps_traj])
    alts = np.array([p[2] for p in gps_traj])
    accel = np.gradient(speed_mag, t)
    jerk  = np.gradient(accel,     t)

    C = {'bg':'#0d1117','panel':'#161b22','text':'#e6edf3','grid':'#30363d',
         'blue':'#58a6ff','green':'#3fb950','red':'#f78166',
         'orange':'#ffa657','purple':'#d2a8ff','wp':'#f0883e'}

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(C['panel'])
        ax.set_title(title,   color=C['text'], fontsize=9, pad=5)
        ax.set_xlabel(xlabel, color=C['text'], fontsize=8)
        ax.set_ylabel(ylabel, color=C['text'], fontsize=8)
        ax.tick_params(colors=C['text'], labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(C['grid'])
        ax.grid(color=C['grid'], lw=0.5, alpha=0.6)

    fig = plt.figure(figsize=(18, 13), facecolor=C['bg'])
    fig.suptitle(f"C²-Smooth Path — WGS-84 — dt={t[1]-t[0]:.4f} s — {len(t)} pts",
                 color=C['text'], fontsize=14, fontweight='bold', y=0.98)
    gs = mgrid.GridSpec(3, 3, figure=fig,
                        hspace=0.50, wspace=0.36,
                        left=0.07, right=0.97, top=0.93, bottom=0.06)

    # نقشه مسیر
    ax1 = fig.add_subplot(gs[0:2, 0:2], facecolor=C['panel'])
    style(ax1, "Flight Path (colored by |V|)", "Longitude (°)", "Latitude (°)")
    sc = ax1.scatter(lons, lats,   c=speed_mag, cmap='plasma', s=2, linewidths=0)
    cb = fig.colorbar(sc, ax=ax1, pad=0.01)
    cb.set_label("|V| (m/s)", color=C['text'], fontsize=8)
    cb.ax.yaxis.set_tick_params(color=C['text'])
    plt.setp(cb.ax.get_yticklabels(), color=C['text'])
    for i, (wlon, wlat) in enumerate(waypoints_gps):
        ax1.plot(  wlon,wlat, 'o', color=C['wp'], ms=9, zorder=5,
                 mec='white', mew=0.8)
        ax1.annotate(f"WP{i+1}", (wlat , wlon), xytext=(5, 4),
                     textcoords='offset points', color=C['wp'],
                     fontsize=8, fontweight='bold')

    # ارتفاع
    ax2 = fig.add_subplot(gs[0, 2], facecolor=C['panel'])
    style(ax2, "Altitude vs Time", "t (s)", "Alt (m)")
    ax2.plot(t, alts, color=C['blue'], lw=1.2)
    ax2.fill_between(t, alts, alpha=0.15, color=C['blue'])

    # سرعت ENU
    ax3 = fig.add_subplot(gs[1, 2], facecolor=C['panel'])
    style(ax3, "ENU Speed Components", "t (s)", "m/s")
    ax3.plot(t, speed_E,   color=C['green'],  lw=0.9, label='East')
    ax3.plot(t, speed_N,   color=C['blue'],   lw=0.9, label='North')
    ax3.plot(t, speed_U,   color=C['purple'], lw=0.9, label='Up')
    ax3.plot(t, speed_mag, color=C['red'],    lw=1.5, label='|V|')
    ax3.axhline(0, color=C['grid'], lw=0.5)
    ax3.legend(fontsize=7, framealpha=0.3, labelcolor=C['text'])

    # شتاب
    ax4 = fig.add_subplot(gs[2, 0], facecolor=C['panel'])
    style(ax4, "Acceleration d|V|/dt (m/s²)", "t (s)", "m/s²")
    ax4.plot(t, accel, color=C['orange'], lw=1.0)
    ax4.axhline(0, color=C['grid'], lw=0.5)
    ax4.fill_between(t, accel, alpha=0.12, color=C['orange'])

    # jerk
    ax5 = fig.add_subplot(gs[2, 1], facecolor=C['panel'])
    style(ax5, "Jerk d²|V|/dt² (m/s³)", "t (s)", "m/s³")
    ax5.plot(t, jerk, color=C['red'], lw=0.8)
    ax5.axhline(0, color=C['grid'], lw=0.5)

    # orientation
    ax6 = fig.add_subplot(gs[2, 2], facecolor=C['panel'])
    style(ax6, "Camera Orientation", "t (s)", "deg")
    ax6.plot(t, np.degrees(roll_rad),  color=C['green'],  lw=0.9, label='Roll')
    ax6.plot(t, np.degrees(pitch_rad), color=C['blue'],   lw=0.9, label='Pitch')
    ax6.plot(t, np.degrees(yaw_rad),   color=C['purple'], lw=0.9, label='Yaw')
    ax6.legend(fontsize=7, framealpha=0.3, labelcolor=C['text'])

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {save_path}")


# ══════════════════════════════════════════════════════════════════
#  خروجی CSV
# ══════════════════════════════════════════════════════════════════

def export_csv(result, save_path="trajectory.csv"):
    """
    ستون‌ها:
      index, time_s,
      lat_deg, lon_deg, alt_m,
      speed_E_mps, speed_N_mps, speed_U_mps, speed_mag_mps,
      roll_rad, pitch_rad, yaw_rad
    """
    (gps_traj,
     roll_rad, pitch_rad, yaw_rad,
     speed_E, speed_N, speed_U,
     t, speed_mag , num_points) = result

    fields = ["index","time_s",
              "lat_deg","lon_deg","alt_m",
              "speed_E_mps","speed_N_mps","speed_U_mps","speed_mag_mps",
              "roll_rad_abs","pitch_rad_abs","yaw_rad_abs"]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, pt in enumerate(gps_traj):
            w.writerow({
                "index":         i,
                "time_s":        f"{t[i]:.6f}",
                "lat_deg":       f"{pt[0]:.8f}",
                "lon_deg":       f"{pt[1]:.8f}",
                "alt_m":         f"{pt[2]:.4f}",
                "speed_E_mps":   f"{speed_E[i]:.6f}",
                "speed_N_mps":   f"{speed_N[i]:.6f}",
                "speed_U_mps":   f"{speed_U[i]:.6f}",
                "speed_mag_mps": f"{speed_mag[i]:.6f}",
                "roll_rad_abs":      f"{roll_rad[i]:.6f}",
                "pitch_rad_abs":     f"{pitch_rad[i]:.6f}",
                "yaw_rad_abs":       f"{yaw_rad[i]:.6f}",
            })
    print(f"CSV  saved → {save_path}  ({len(gps_traj)} rows, dt={t[1]-t[0]:.4f} s)")


# ══════════════════════════════════════════════════════════════════
#  مثال اجرا
# ══════════════════════════════════════════════════════════════════

def main(WAYPOINTS,flight_height_meters,base_speed,freq,outdir):


    result = genPath_waypoints(
        waypoints_gps        = WAYPOINTS,
        flight_height_meters = flight_height_meters,
        base_speed           = base_speed,    # m/s
        dt_sec               = 1/freq,   # 333 Hz
        height_variation     = 5.0,
    )

    plot_results(result, WAYPOINTS, save_path=outdir +"/flight_path.png")
    export_csv(result,              save_path=outdir +"/trajectory.csv")
    return result
