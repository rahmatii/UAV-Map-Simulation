import argparse
import os
from pathlib import Path

import cv2


def images_to_video_fast(image_folder, output_video, fps=30, extension='.png'):
    """
    نسخه بسیار سریع - فقط بر اساس اسم فایل (بدون پردازش اضافی)
    """
    import glob
    
    # گرفتن لیست فایل‌ها با glob (سریع‌تر از pathlib)
    images = glob.glob(os.path.join(image_folder, f'*{extension}'))
    
    if not images:
        print(f"❌ No images found!")
        return False
    
    # مرتب‌سازی ساده (برای اسم‌هایی که با صفر padding شدن خوب کار میکنه)
    images.sort()
    
    print(f"📁 Found {len(images)} images")
    
    # خوندن اولین فریم
    frame = cv2.imread(images[0])
    if frame is None:
        print("❌ Error reading first image")
        return False
    
    height, width = frame.shape[:2]
    
    # ساخت ویدیو
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"🎬 Processing {len(images)} frames...")
    
    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is not None:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
        
        if (i + 1) % 200 == 0:
            print(f"Progress: {i+1}/{len(images)}")
    
    out.release()
    print(f"✅ Done: {output_video}")
    
    return True


# راه حل حرفه‌ای: شماره‌گذاری مجدد فریم‌ها
def renumber_frames(image_folder, extension='.png', digits=5):
    """
    تغییر نام فریم‌ها به فرمت یکسان (مثلاً frame_00001.png)
    این کار باعث میشه مرتب‌سازی ساده به درستی کار کنه
    """
    import shutil
    
    images = sorted(Path(image_folder).glob(f'*{extension}'))
    
    if not images:
        print("No images found")
        return
    
    print(f"Renaming {len(images)} files...")
    
    temp_folder = Path(image_folder) / "temp_rename"
    temp_folder.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(images, 1):
        # نام جدید با تعداد رقم مشخص
        new_name = f"frame_{i:0{digits}d}{extension}"
        temp_path = temp_folder / new_name
        
        # کپی به پوشه موقت با نام جدید
        shutil.copy2(img_path, temp_path)
    
    # پاک کردن فایل‌های اصلی و جایگزینی
    for img_path in images:
        img_path.unlink()
    
    for temp_file in temp_folder.glob(f'*{extension}'):
        shutil.move(str(temp_file), str(Path(image_folder) / temp_file.name))
    
    temp_folder.rmdir()
    print(f"✅ Renamed {len(images)} files with {digits} digits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to video - Fast version')
    parser.add_argument('-input_folder', default='test7')
    parser.add_argument('-o', '--output', default='test7.mp4')
    parser.add_argument('-fps', '--fps', type=int, default=30)
    parser.add_argument('-ext', '--extension', default='.png')
    parser.add_argument('-rename', '--renumber', action='store_true', 
                       help='Renumber frames first (fixes sorting issues)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"❌ Folder not found: {args.input_folder}")
        exit(1)
    
    # اگر مشکل مرتب‌سازی دارید، اول شماره‌گذاری مجدد کنید
    if args.renumber:
        renumber_frames(args.input_folder, args.extension)
    
    # استفاده از نسخه سریع
    images_to_video_fast(args.input_folder, args.output, args.fps, args.extension)