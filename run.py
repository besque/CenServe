"""
CenServe Launcher
Starts the Flask server then opens the browser.
Build:  pyinstaller --onefile --noconsole --name CenServe run.py
"""

import sys
import os
import time
import threading
import webbrowser


def _patch_paths():
    """
    When running from a PyInstaller bundle, sys._MEIPASS points to the
    temp extraction directory. Ensure the project root and package are
    importable from there.
    """
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    if base not in sys.path:
        sys.path.insert(0, base)
    os.chdir(base)


def _start_server():
    from censerve.web.server import app
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)


def main():
    _patch_paths()

    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()

    time.sleep(2.5)
    webbrowser.open('http://localhost:5000')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
