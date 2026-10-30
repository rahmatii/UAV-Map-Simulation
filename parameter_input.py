"""
رابط کاربری ساده برای دریافت پارامترهای شبیه‌سازی:
FOV, output_dir, image_size, altitude, speed
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os


class ParameterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("تنظیم پارامترهای شبیه‌سازی")
        self.root.geometry("500x400")
        self.root.resizable(False, False)

        self.result = None  # پارامترهای نهایی پس از تایید

        pad = {"padx": 10, "pady": 8}
        frame = ttk.Frame(root)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        row = 0

        # ---------------- FOV ----------------
        ttk.Label(frame, text="FOV (درجه):").grid(row=row, column=0, sticky="w", **pad)
        self.fov_var = tk.StringVar(value="90")
        ttk.Entry(frame, textvariable=self.fov_var, width=20).grid(row=row, column=1, **pad)
        row += 1

        
        # ---------------- FOV ----------------
        ttk.Label(frame, text="Freq (داده های سنسور):").grid(row=row, column=0, sticky="w", **pad)
        self.Freq = tk.StringVar(value="100")
        ttk.Entry(frame, textvariable=self.Freq, width=20).grid(row=row, column=1, **pad)
        row += 1
        # ---------------- output_dir ----------------
        ttk.Label(frame, text="output_dir:").grid(row=row, column=0, sticky="w", **pad)
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        entry_out = ttk.Entry(frame, textvariable=self.output_dir_var, width=20)
        entry_out.grid(row=row, column=1, **pad)
        ttk.Button(frame, text="choose", command=self.browse_output_dir).grid(row=row, column=2, **pad)
        row += 1

        # ---------------- image_size ----------------
        ttk.Label(frame, text="image_size (پیکسل):").grid(row=row, column=0, sticky="w", **pad)
        self.image_size_var = tk.StringVar(value="512")
        ttk.Entry(frame, textvariable=self.image_size_var, width=20).grid(row=row, column=1, **pad)
        row += 1

        # ---------------- altitude ----------------
        ttk.Label(frame, text="ارتفاع (متر):").grid(row=row, column=0, sticky="w", **pad)
        self.altitude_var = tk.StringVar(value="200")
        ttk.Entry(frame, textvariable=self.altitude_var, width=20).grid(row=row, column=1, **pad)
        row += 1

        # ---------------- speed ----------------
        ttk.Label(frame, text="سرعت (m/s):").grid(row=row, column=0, sticky="w", **pad)
        self.speed_var = tk.StringVar(value="50")
        ttk.Entry(frame, textvariable=self.speed_var, width=20).grid(row=row, column=1, **pad)
        row += 1

        # ---------------- دکمه‌ها ----------------
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=20)
        ttk.Button(btn_frame, text="تایید", command=self.on_submit).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="انصراف", command=self.root.destroy).pack(side="left", padx=10)

    def browse_output_dir(self):
        selected = filedialog.askdirectory(initialdir=self.output_dir_var.get() or os.getcwd())
        if selected:
            self.output_dir_var.set(selected)

    def on_submit(self):
        try:
            fov = float(self.fov_var.get())
            Freq = float(self.Freq.get())
            image_size = int(self.image_size_var.get())
            altitude = float(self.altitude_var.get())
            speed = float(self.speed_var.get())
            output_dir = self.output_dir_var.get().strip()

            if not output_dir:
                raise ValueError("output_dir نمی‌تواند خالی باشد")
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            if not (0 < fov <= 360):
                raise ValueError("FOV باید بین 0 و 360 درجه باشد")
            if image_size <= 0:
                raise ValueError("image_size باید عددی مثبت باشد")
            if speed <= 0:
                raise ValueError("سرعت باید عددی مثبت باشد")

        except ValueError as e:
            messagebox.showerror("خطا در ورودی", f"لطفاً مقادیر را بررسی کنید:\n{e}")
            return

        self.result = {
            "fov": fov,
            "freq": Freq,
            "output_dir": output_dir,
            "image_size": image_size,
            "altitude": altitude,
            "speed": speed,
        }

        self.root.destroy()


def get_parameters() -> dict | None:
    """
    رابط کاربری را باز می‌کند و پس از تایید کاربر، دیکشنری پارامترها را برمی‌گرداند.
    اگر کاربر پنجره را ببندد یا لغو کند، None برگردانده می‌شود.
    """
    root = tk.Tk()
    app = ParameterUI(root)
    root.mainloop()
    return app.result


def main():
    params = get_parameters()
    if params is not None:
        print("پارامترهای دریافت‌شده:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    else:
        print("کاربر عملیات را لغو کرد یا پنجره بسته شد.")
    return params