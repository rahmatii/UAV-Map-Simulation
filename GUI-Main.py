

import sys, json, os, urllib.request
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QListWidget, QPushButton,
    QListWidgetItem, QFrame, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import waypointSmooth
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import Simulation
import parameter_input
import MapApp.my_load_GeoTif as my_load_GeoTif


# ─── مسیر فایل‌ها ────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
WORLD_GEOJSON    = os.path.join(BASE_DIR, "countries.geojson")
PROVINCE_GEOJSON = os.path.join(BASE_DIR, "iran_provinces.geojson")

WORLD_URL    = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
PROVINCE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson"

# ─── شهرهای ایران (lon, lat, نام فارسی, نام انگلیسی) ─────────────────────────
IRAN_CITIES = [
    (51.4215, 35.6944, "تهران",         "Tehran"),
    (59.5706, 36.2981, "مشهد",          "Mashhad"),
    (51.6660, 32.6539, "اصفهان",        "Isfahan"),
    (46.2919, 38.0800, "تبریز",         "Tabriz"),
    (52.5310, 29.5918, "شیراز",         "Shiraz"),
    (48.6842, 31.3183, "اهواز",         "Ahvaz"),
    (49.5892, 37.2808, "رشت",           "Rasht"),
    (47.0650, 34.3142, "کرمانشاه",      "Kermanshah"),
    (57.0788, 30.2839, "کرمان",         "Kerman"),
    (54.3680, 31.8974, "یزد",           "Yazd"),
    (50.8649, 35.8040, "قم",            "Qom"),
    (48.5146, 35.1174, "همدان",         "Hamadan"),
    (58.7960, 37.4895, "سبزوار",        "Sabzevar"),
    (56.2646, 27.1832, "بندرعباس",      "Bandar Abbas"),
    (50.0041, 36.5168, "قزوین",         "Qazvin"),
    (45.0759, 37.5527, "ارومیه",        "Urmia"),
    (48.3165, 30.9225, "خرم‌آباد",      "Khorramabad"),
    (49.6958, 36.8013, "کرج",           "Karaj"),
    (60.6195, 36.7007, "نیشابور",       "Nishapur"),
    (55.1689, 36.4143, "شاهرود",        "Shahrud"),
    (53.5791, 35.5729, "سمنان",         "Semnan"),
    (54.4268, 30.2868, "بم",            "Bam"),
    (47.7939, 32.4942, "ایلام",         "Ilam"),
    (47.6263, 39.6481, "اردبیل",        "Ardabil"),
    (50.8799, 32.8837, "شهرکرد",        "Shahrekord"),
    (51.9240, 34.6416, "اراک",          "Arak"),
    (54.5691, 30.5156, "سیرجان",        "Sirjan"),
    (61.8608, 29.4963, "زاهدان",        "Zahedan"),
    (50.3288, 30.4387, "بوشهر",         "Bushehr"),
    (55.8385, 37.2835, "گرگان",         "Gorgan"),
    (52.6558, 36.5633, "ساری",          "Sari"),
    (46.7560, 34.3172, "سنندج",         "Sanandaj"),
    (56.9666, 29.5021, "جیرفت",         "Jiroft"),
    (49.3152, 33.4871, "خمین",          "Khomein"),
    (59.2183, 36.8312, "قوچان",         "Quchan"),
    (48.7463, 38.5552, "خوی",           "Khoy"),
    (63.2174, 31.3308, "ایرانشهر",      "Iranshahr"),
    (52.6813, 32.2645, "یاسوج",         "Yasuj"),
    (56.3739, 36.5280, "بهشهر",         "Behshahr"),
]

# ─── رنگ‌های تم ──────────────────────────────────────────────────────────────
OCEAN   = "#0d1b2a"
LAND    = "#1b3a4b"
IRAN_F  = "#1a4a6a"
IRAN_B  = "#e74c3c"
PROV_B  = "#f39c12"   # مرز استان‌ها
CLICK   = "#00ffcc"
CITY_C  = "#f5c842"
SIDE_BG = "#0a1828"

# ─── تبدیل geometry به Patch ──────────────────────────────────────────────────
def geometry_to_patches(geometry, facecolor, edgecolor, lw=0.4, alpha=1.0, zorder=2):
    patches = []
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    rings = []
    if gtype == "Polygon":
        rings = coords
    elif gtype == "MultiPolygon":
        for poly in coords:
            rings.extend(poly)
    for ring in rings:
        if len(ring) < 3:
            continue
        verts = [(x, y) for x, y in ring]
        verts.append(verts[0])
        codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
        patch = PathPatch(Path(verts, codes),
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=lw, alpha=alpha, zorder=zorder)
        patches.append(patch)
    return patches

# ─── Canvas نقشه ─────────────────────────────────────────────────────────────
class WorldMapCanvas(FigureCanvas):
    def __init__(self, world_data, province_data, parent=None, on_click=None):
        self.fig = Figure(figsize=(13, 7.5), dpi=100, facecolor=OCEAN)
        super().__init__(self.fig)
        self.setParent(parent)
        self.on_click       = on_click
        self.world_data     = world_data
        self.province_data  = province_data
        self.click_markers  = []
        self.click_texts    = []
        self.clicked_points = []
        self._draw_map()
        self.mpl_connect("button_press_event", self._mouse_click)

    def _draw_map(self):
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(OCEAN)
        self.fig.patch.set_facecolor(OCEAN)

        # ── 1. رسم همه کشورها ──
        for feature in self.world_data["features"]:
            name = feature["properties"].get("name", "")
            if name == "Iran":
                continue
            for p in geometry_to_patches(feature["geometry"], LAND, "#2e5a70", lw=0.3):
                self.ax.add_patch(p)

        # ── 2. رسم ایران با رنگ ویژه ──
        iran_feature = next((f for f in self.world_data["features"]
                             if f["properties"].get("name") == "Iran"), None)
        if iran_feature:
            for p in geometry_to_patches(iran_feature["geometry"],
                                         IRAN_F, IRAN_B, lw=1.5, zorder=3):
                self.ax.add_patch(p)

        # ── 3. رسم مرز استان‌های ایران ──
        if self.province_data:
            for feature in self.province_data["features"]:
                for p in geometry_to_patches(feature["geometry"],
                                             "none", PROV_B, lw=0.65, alpha=0.85, zorder=4):
                    self.ax.add_patch(p)

            # ── 4. برچسب نام استان‌ها ──
            for feature in self.province_data["features"]:
                props = feature["properties"]
                name_fa = props.get("name_en", "").replace("استان ", "")
                lon = props.get("longitude", 0)
                lat = props.get("latitude", 0)
                if lon and lat:
                    self.ax.text(lon, lat, name_fa,
                                 color="#f39c12", fontsize=5.5,
                                 ha="center", va="center",
                                 fontfamily="DejaVu Sans",
                                 zorder=6, alpha=0.85,
                                 bbox=dict(boxstyle="round,pad=0.15",
                                           facecolor=OCEAN, edgecolor="none",
                                           alpha=0.5))

        # ── 5. رسم شهرهای ایران ──
        for lon, lat, name_fa, name_en in IRAN_CITIES:
            # نقطه شهر
            self.ax.plot(lon, lat, "o", color=CITY_C,
                         markersize=3.2, zorder=7,
                         markeredgecolor="#00000055", markeredgewidth=0.4)
            # نام شهر
            self.ax.text(lon + 0.18, lat + 0.18, name_en,
                         color=CITY_C, fontsize=5.0,
                         fontfamily="DejaVu Sans",
                         zorder=8, alpha=0.9,
                         bbox=dict(boxstyle="round,pad=0.1",
                                   facecolor=OCEAN, edgecolor="none",
                                   alpha=0.4))

        # ── 6. خطوط مدار/نصف‌النهار ──
        for lat in range(-60, 91, 30):
            self.ax.axhline(lat, color="#1a3a4a", lw=0.35, ls="--", alpha=0.5, zorder=1)
        for lon in range(-180, 181, 60):
            self.ax.axvline(lon, color="#1a3a4a", lw=0.35, ls="--", alpha=0.5, zorder=1)
        self.ax.axhline(0, color="#2e6b8a", lw=0.7, alpha=0.7, zorder=1)
        self.ax.axvline(0, color="#2e6b8a", lw=0.7, alpha=0.7, zorder=1)

        self.ax.set_xlim(-180, 180)
        self.ax.set_ylim(-90, 90)
        self.ax.set_aspect("equal")

        self.ax.set_xlabel("Longitude", color="#7fa8c0", fontsize=9, labelpad=6)
        self.ax.set_ylabel("Latitude",  color="#7fa8c0", fontsize=9, labelpad=6)
        self.ax.set_title("نقشه تعاملی جهان | مرز استان‌های ایران",
                          color="#e0eaf0", fontsize=13, fontweight="bold", pad=12)
        self.ax.tick_params(colors="#5a8aaa", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#1a3a4a")

        # راهنما
        h = [
            mpatches.Patch(facecolor=IRAN_F,  edgecolor=IRAN_B, lw=1.5, label="IRAN"),
           
        ]
        self.ax.legend(handles=h, facecolor="#0a1828", edgecolor="#2e6b8a",
                       labelcolor="#c0d8e8", fontsize=8, loc="lower left",
                       framealpha=0.9)
        self.fig.tight_layout(pad=1.2)

    def _mouse_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        lon = round(event.xdata, 6)
        lat = round(event.ydata, 6)
        num = len(self.clicked_points) + 1
        m, = self.ax.plot(lon, lat, "*", color=CLICK, markersize=11,
                          zorder=10, markeredgecolor="white", markeredgewidth=0.5)
        t = self.ax.text(lon+1.2, lat+1.2, str(num),
                         color=CLICK, fontsize=7.5, fontweight="bold", zorder=11)
        self.click_markers.append(m)
        self.click_texts.append(t)
        self.clicked_points.append((lon, lat))
        self.draw()
        if self.on_click:
            self.on_click(lon, lat, num)

    def remove_last(self):
        if self.click_markers:
            self.click_markers.pop().remove()
            self.click_texts.pop().remove()
            self.clicked_points.pop()
            self.draw()

    def clear_all(self):
        for m in self.click_markers: m.remove()
        for t in self.click_texts:   t.remove()
        self.click_markers.clear()
        self.click_texts.clear()
        self.clicked_points.clear()
        self.draw()

# ─── Thread دانلود ────────────────────────────────────────────────────────────
class DownloadThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, urls_paths):
        super().__init__()
        self.urls_paths = urls_paths   # [(url, path, label), ...]

    def run(self):
        for url, path, label in self.urls_paths:
            if os.path.exists(path):
                self.progress.emit(f"✅ {label} از قبل موجود است")
                continue
            self.progress.emit(f"⏬ در حال دانلود {label}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read()

                # اگر داده استان‌ها بود، فقط ایران را نگه‌دار
                if "admin_1_states" in url:
                    data = json.loads(raw)
                    iran = {"type": "FeatureCollection",
                            "features": [f for f in data["features"]
                                         if f["properties"].get("adm0_a3") == "IRN"]}
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(iran, f, ensure_ascii=False)
                else:
                    with open(path, "wb") as f:
                        f.write(raw)

                self.progress.emit(f"✅ {label} دانلود شد")
            except Exception as e:
                self.finished.emit(False, f"خطا در دانلود {label}:\n{e}")
                return
        self.finished.emit(True, "")

# ─── پنجره بارگذاری ───────────────────────────────────────────────────────────
class LoadWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("در حال بارگذاری نقشه...")
        self.setFixedSize(420, 140)
        self.setStyleSheet(f"background:{OCEAN}; color:#80c0d8;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)
        self.label = QLabel("در حال آماده‌سازی...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size:13px;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setStyleSheet(
            "QProgressBar{background:#0f3060;border-radius:5px;height:8px;}"
            "QProgressBar::chunk{background:#e74c3c;border-radius:5px;}")
        note = QLabel("فایل‌ها یک‌بار دانلود و برای همیشه ذخیره می‌شوند")
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet("font-size:10px; color:#3a6080;")
        lay.addWidget(self.label)
        lay.addWidget(self.bar)
        lay.addWidget(note)

    def set_status(self, txt):
        self.label.setText(txt)
        QApplication.processEvents()

# ─── پنجره اصلی ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, world_data, province_data):
        super().__init__()
        self.setWindowTitle("نقشه تعاملی جهان — GPS Clicker")
        self.setMinimumSize(QSize(1250, 720))
        self._build_ui(world_data, province_data)
        self._apply_style()

    def _build_ui(self, world_data, province_data):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # نقشه
        map_w = QWidget(); map_w.setObjectName("mapWidget")
        mv = QVBoxLayout(map_w)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        self.canvas = WorldMapCanvas(world_data, province_data, on_click=self._on_click)
        toolbar = NavigationToolbar2QT(self.canvas, map_w)
        toolbar.setStyleSheet("background:#0a1828;border:none;color:#7fa8c0;")
        mv.addWidget(toolbar)
        mv.addWidget(self.canvas)

        # پانل کناری
        side = QWidget(); side.setObjectName("sidePanel"); side.setFixedWidth(282)
        s = QVBoxLayout(side)
        s.setContentsMargins(14, 18, 14, 18); s.setSpacing(10)

        title = QLabel("📍 مختصات GPS")
        title.setObjectName("panelTitle"); title.setAlignment(Qt.AlignCenter)
        s.addWidget(title)

        cf = QFrame(); cf.setObjectName("coordFrame")
        cl = QVBoxLayout(cf); cl.setContentsMargins(10,8,10,8); cl.setSpacing(4)
        self.lat_lbl     = QLabel("عرض (Lat): —")
        self.lon_lbl     = QLabel("طول (Lon): —")
        self.dms_lbl     = QLabel("DMS: —")
        self.country_lbl = QLabel("کشور / استان: —")
        for lb in [self.lat_lbl, self.lon_lbl, self.dms_lbl, self.country_lbl]:
            lb.setObjectName("coordLabel"); lb.setWordWrap(True); cl.addWidget(lb)
        s.addWidget(cf)

        lh = QLabel("نقاط ثبت‌شده:")
        lh.setObjectName("sectionTitle"); s.addWidget(lh)
        self.list_w = QListWidget(); self.list_w.setObjectName("pointsList")
        s.addWidget(self.list_w)

        for label, obj_name, slot in [
            ("↩  حذف آخرین نقطه", "undoBtn",  self._undo),
            ("🗑  پاک کردن همه",   "clearBtn", self._clear),
            ("📋  کپی همه مختصات","copyBtn",  self._copy),
            ("ساخت مسیر","tragectoryBtn",  self._GenTrajectory),
        ]:
            btn = QPushButton(label); btn.setObjectName(obj_name)
            btn.clicked.connect(slot); s.addWidget(btn)

        hint = QLabel("💡 اسکرول = زوم  |  ✋ Pan = جابجایی\n🔍 Zoom = کادر زوم  |  🏠 Home = ریست")
        hint.setObjectName("hintLabel"); hint.setAlignment(Qt.AlignCenter)
        s.addWidget(hint)

        root.addWidget(map_w, stretch=1)
        root.addWidget(side)

    # ─── رویدادها ────────────────────────────────────────────────────────────
    def _on_click(self, lon, lat, num):
        self.lat_lbl.setText(f"عرض (Lat): {lat:.6f}°")
        self.lon_lbl.setText(f"طول (Lon): {lon:.6f}°")
        self.dms_lbl.setText(
            f"DMS: {self._dms(lat,'N','S')}  {self._dms(lon,'E','W')}")
        location = self._find_location(lon, lat)
        self.country_lbl.setText(f"موقعیت: {location}")
        item = QListWidgetItem(f"#{num}  {lat:.4f}°, {lon:.4f}°  [{location}]")
        item.setForeground(QColor(CLICK))
        self.list_w.addItem(item); self.list_w.scrollToBottom()

    def _find_location(self, lon, lat):
        try:
            from shapely.geometry import Point, shape
            pt = Point(lon, lat)
            # اول استان‌های ایران
            if self.canvas.province_data:
                for f in self.canvas.province_data["features"]:
                    if shape(f["geometry"]).contains(pt):
                        fa = f["properties"].get("name_en","").replace("استان ","")
                        en = f["properties"].get("name_en","")
                        return f"{fa} ({en}) — ایران"
            # بعد کشورها
            for f in self.canvas.world_data["features"]:
                if shape(f["geometry"]).contains(pt):
                    return f["properties"].get("name","نامشخص")
        except Exception:
            pass
        return "اقیانوس / دریا"

    def _dms(self, dd, pos, neg):
        d = pos if dd >= 0 else neg
        dd = abs(dd)
        deg = int(dd); mn = int((dd-deg)*60)
        sec = ((dd-deg)*60 - mn)*60
        return f"{deg}°{mn}′{sec:.1f}″{d}"

    def _undo(self):
        self.canvas.remove_last()
        if self.list_w.count():
            self.list_w.takeItem(self.list_w.count()-1)
        if not self.canvas.clicked_points:
            for lb in [self.lat_lbl, self.lon_lbl, self.dms_lbl, self.country_lbl]:
                lb.setText(lb.text().split(":")[0]+": —")

    def _clear(self):
        if not self.canvas.clicked_points: return
        if QMessageBox.question(self,"تأیید","همه نقاط پاک شوند؟",
                QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            self.canvas.clear_all(); self.list_w.clear()
            for lb in [self.lat_lbl, self.lon_lbl, self.dms_lbl, self.country_lbl]:
                lb.setText(lb.text().split(":")[0]+": —")
    
    def _GenTrajectory(self):
        
        pts = self.canvas.clicked_points
        
        
        if ( len(pts) == 0 or len(pts) == 1):
            
            QMessageBox.information(self,"خطا","حداقل دو نقطه انتخاب کنید")
            return
        if (len(pts) >=6):
            QMessageBox.information(self,"خطا","حداکثر 5 نقطه انتخاب کنید")
            return
        x, y = zip(*pts)

        xmin = min(x)
        xmax = max(x)

        ymin = min(y)
        ymax = max(y)

       

        params = parameter_input.main()
        print("params ",params)
        pixel_size = Simulation.calculate_pixel_size(params['fov'], params['image_size'], params['altitude'])
        if pixel_size < 0.36:
            zoom = 19
        elif pixel_size < 0.72 :
            zoom = 18
        elif pixel_size < 1.46 :
            zoom = 17 
        else :
            zoom = 16
        
      
        gps_trajectory, roll_rad, pitch_rad, yaw_rad, speed_E, speed_N, speed_U, t, speed_mag,num_points = waypointSmooth.main(pts,params['altitude'],params['speed'],params['freq'],params['output_dir'])
        #gps_trajectory, num_points, t, yaw_rad, pitch_rad, roll_rad , lat_min , lat_max , lon_min , lon_max =rauof.rauof()
   
        my_load_GeoTif.main(ymax + 0.1 , ymin - 0.1 , xmax + 0.1 , xmin - 0.1 , params['output_dir'] , zoom)
        print("numpoint ",num_points)
        geoPath = params['output_dir'] + "/geotif.tif"
        Simulation.main(gps_trajectory, roll_rad, pitch_rad, yaw_rad,t,params['freq'],params['altitude'],num_points,params['output_dir'],params['image_size'],params['fov'],geoPath)
        sys.exit(QApplication.exec_())  

    def _copy(self):
        pts = self.canvas.clicked_points
        if not pts: return
        txt = "\n".join(f"#{i}: lat={la:.6f}, lon={lo:.6f}"
                        for i,(lo,la) in enumerate(pts,1))
        QApplication.clipboard().setText(txt)
        QMessageBox.information(self,"کپی شد",f"{len(pts)} نقطه کپی شد.")

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{OCEAN};}}
            #mapWidget{{background:{OCEAN};}}
            #sidePanel{{background:{SIDE_BG};border-left:1px solid #1a3a4a;}}
            #panelTitle{{color:#c8e0f0;font-size:15px;font-weight:bold;padding:4px 0;}}
            #coordFrame{{background:#0d2030;border:1px solid {IRAN_B};border-radius:7px;}}
            #coordLabel{{color:#80c8e0;font-size:11px;}}
            #sectionTitle{{color:#5a8aaa;font-size:10px;font-weight:bold;}}
            #pointsList{{background:#060f1a;color:{CLICK};border:1px solid #1a3a4a;
                border-radius:5px;font-size:10.5px;selection-background-color:#0d2030;}}
            QPushButton{{border-radius:6px;padding:8px;font-size:12px;font-weight:bold;border:none;}}
            #undoBtn{{background:#1a3050;color:#80aacc;}}
            #undoBtn:hover{{background:#243d60;}}
            #clearBtn{{background:#3d1010;color:#ff9090;}}
            #clearBtn:hover{{background:#4d1a1a;}}
            #copyBtn{{background:#0d3020;color:#80ffcc;}}
            #copyBtn:hover{{background:#1a4030;}}
            #hintLabel{{color:#3a6080;font-size:9px;padding:4px;}}
        """)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    downloads_needed = []
    if not os.path.exists(WORLD_GEOJSON):
        downloads_needed.append((WORLD_URL, WORLD_GEOJSON, "نقشه جهان"))
    if not os.path.exists(PROVINCE_GEOJSON):
        downloads_needed.append((PROVINCE_URL, PROVINCE_GEOJSON, "مرز استان‌های ایران"))

    if downloads_needed:
        loader = LoadWindow(); loader.show(); QApplication.processEvents()
        result = {"ok": False, "msg": ""}

        thread = DownloadThread(downloads_needed)
        thread.progress.connect(loader.set_status)
        thread.finished.connect(lambda ok, msg: result.update({"ok": ok, "msg": msg}))
        thread.start()
        while thread.isRunning():
            QApplication.processEvents()

        loader.close()
        if not result["ok"]:
            QMessageBox.critical(None, "خطا", result["msg"])
            sys.exit(1)

    # بارگذاری داده‌ها
    with open(WORLD_GEOJSON, encoding="utf-8") as f:
        world_data = json.load(f)

    province_data = None
    if os.path.exists(PROVINCE_GEOJSON):
        with open(PROVINCE_GEOJSON, encoding="utf-8") as f:
            province_data = json.load(f)

    win = MainWindow(world_data, province_data)
    win.show()
    print("✅ نقشه آماده است. روی هر نقطه کلیک کنید.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
