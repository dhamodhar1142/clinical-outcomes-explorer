import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    port = "8510"
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        port,
    ]
    screenshot_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_demo_screenshots.py"),
        "--url",
        f"http://localhost:{port}",
    ]

    process = subprocess.Popen(streamlit_cmd, cwd=ROOT)
    try:
        time.sleep(10)
        result = subprocess.run(screenshot_cmd, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
