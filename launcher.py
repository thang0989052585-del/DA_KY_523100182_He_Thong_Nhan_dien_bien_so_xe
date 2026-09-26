"""
launcher.py — Entry Point cho PyInstaller
=========================================
Script này được PyInstaller đóng gói thành file .exe (BienSoAI.exe).
Khi chạy, nó sẽ:
  1. Xác định đường dẫn thực tế của app và tài nguyên (_internal hoặc source)
  2. Khởi động Streamlit server
  3. Tự động mở trình duyệt web sau 3.5 giây
  4. Giữ cửa sổ terminal hiển thị log, khi tắt app sẽ thông báo
"""

import os
import sys
import time
import threading
import webbrowser
import subprocess

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_DIR = BASE_DIR

APP_PY = os.path.join(BASE_DIR, "app.py")
if not os.path.isfile(APP_PY):
    APP_PY = os.path.join(EXE_DIR, "app.py")

PORT = 8501
URL = f"http://localhost:{PORT}"


def open_browser_after_delay(delay: float = 3.5):
    """Mở trình duyệt sau delay giây để Streamlit kịp khởi động."""
    time.sleep(delay)
    print(f"[launcher] Đang mở trình duyệt: {URL}")
    webbrowser.open(URL)


def main():
    print("=" * 60)
    print("  🚗 HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE THÔNG MINH (AI)")
    print("=" * 60)
    print(f"[launcher] BASE_DIR : {BASE_DIR}")
    print(f"[launcher] EXE_DIR  : {EXE_DIR}")
    print(f"[launcher] app.py   : {APP_PY}")
    print(f"[launcher] Web URL  : {URL}")
    print()

    if not os.path.isfile(APP_PY):
        print(f"[ERROR] Không tìm thấy file app.py tại: {APP_PY}")
        print("Vui lòng đảm bảo app.py nằm cùng thư mục hoặc trong _internal!")
        input("Nhấn Enter để thoát...")
        sys.exit(1)

    browser_thread = threading.Thread(target=open_browser_after_delay, args=(3.5,), daemon=True)
    browser_thread.start()

    streamlit_exe = os.path.join(EXE_DIR, "streamlit.exe")
    if not os.path.isfile(streamlit_exe):
        internal_st = os.path.join(BASE_DIR, "streamlit.exe")
        if os.path.isfile(internal_st):
            streamlit_exe = internal_st
        else:
            streamlit_exe = "streamlit"

    cmd = [
        streamlit_exe,
        "run",
        APP_PY,
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "light",
    ]

    print(f"[launcher] Lệnh khởi động: {' '.join(cmd)}")
    print("[launcher] Streamlit đang khởi động, vui lòng chờ trong giây lát...")
    print()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            print(f"[streamlit] {line}", end="")
        process.wait()
    except FileNotFoundError:
        print("[launcher] Không tìm thấy lệnh ngoài, đang chuyển sang chế độ module nội bộ...")
        try:
            import streamlit.web.cli as stcli
            sys.argv = [
                "streamlit",
                "run",
                APP_PY,
                "--server.port",
                str(PORT),
                "--server.headless",
                "true",
                "--server.fileWatcherType",
                "none",
                "--browser.gatherUsageStats",
                "false",
                "--theme.base",
                "light",
            ]
            stcli.main()
        except Exception as e:
            print(f"[ERROR] Lỗi khi chạy Streamlit nội bộ: {e}")
            input("Nhấn Enter để thoát...")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[launcher] Đã nhận tín hiệu dừng (Ctrl+C).")
    finally:
        print("\n[launcher] Ứng dụng đã đóng. Cảm ơn bạn đã sử dụng!")
        try:
            input("Nhấn Enter để thoát...")
        except Exception:
            pass


if __name__ == "__main__":
    main()
