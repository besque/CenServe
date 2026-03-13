"""
CenServe Screen Capture
Captures a monitor or a specific application window as a numpy frame.
Same read() interface as cv2.VideoCapture so it's a drop-in for the pipeline.

Windows only — uses mss for capture and win32gui for window enumeration.

Important note on hardware-accelerated windows (Chrome, Electron apps):
  These render via GPU and mss will capture a black frame by default.
  Fix: call SetWindowDisplayAffinity before capturing.
  This is handled automatically in set_source().
"""

import numpy as np
import cv2
import mss
import win32gui
import win32con


# ── Source enumeration ────────────────────────────────────────────────────────

def list_sources() -> list:
    """
    Returns all capturable sources — monitors and visible application windows.
    Each source is a dict with:
      id      str   unique identifier
      label   str   human-readable name shown in the UI
      type    str   'monitor' or 'window'

    For monitors, also includes:
      index   int   mss monitor index (1-based)

    For windows, also includes:
      hwnd    int   Win32 window handle
    """
    sources = []

    # ── Monitors ──
    with mss.mss() as sct:
        for i, mon in enumerate(sct.monitors[1:], 1):
            sources.append({
                'id':    f'monitor_{i}',
                'label': f'Monitor {i}  ({mon["width"]}×{mon["height"]})',
                'type':  'monitor',
                'index': i,
            })

    # ── Application windows ──
    def _enum_cb(hwnd, out):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title or len(title) < 2:
            return
        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if w < 200 or h < 150:
            return
        out.append({
            'id':    f'window_{hwnd}',
            'label': title,
            'type':  'window',
            'hwnd':  hwnd,
        })

    wins = []
    win32gui.EnumWindows(_enum_cb, wins)
    sources.extend(wins[:20])

    return sources


# ── Capture class ─────────────────────────────────────────────────────────────

class ScreenCapture:
    """
    Captures frames from a selected monitor or application window.

    Usage:
        sc = ScreenCapture()
        sc.set_source({'type': 'monitor', 'index': 1})
        ok, frame = sc.read()   # frame is BGR numpy array, 1280×720
    """

    def __init__(self):
        self.sct     = mss.mss()
        self._source = None

    def set_source(self, source: dict):
        """
        Set which monitor or window to capture.
        For windows, also calls SetWindowDisplayAffinity to force
        GPU-accelerated windows (Chrome etc) to be capturable.
        """
        self._source = source

        # Force hardware-accelerated windows to be capturable by mss
        # Without this, Chrome and most Electron apps capture as black frames
        if source.get('type') == 'window':
            hwnd = source.get('hwnd')
            if hwnd:
                try:
                    # WDA_NONE = 0 disables display affinity protection
                    win32gui.SetWindowDisplayAffinity(hwnd, 0)
                except Exception as e:
                    print(f'[ScreenCapture] SetWindowDisplayAffinity failed: {e}')
                    print('[ScreenCapture] Chrome/Electron tabs may capture as black')

    def read(self):
        """
        Capture one frame from the selected source.
        Returns (True, frame_bgr) on success, (False, None) on failure.
        Output is always resized to 1280×720 for consistent pipeline input.
        """
        if self._source is None:
            return False, None

        try:
            if self._source['type'] == 'monitor':
                return self._read_monitor()
            elif self._source['type'] == 'window':
                return self._read_window()
            return False, None
        except Exception as e:
            print(f'[ScreenCapture] read() error: {e}')
            return False, None

    def _read_monitor(self):
        """Capture a full monitor."""
        mon = self.sct.monitors[self._source['index']]
        img = self.sct.grab(mon)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (1280, 720))
        return True, frame

    def _read_window(self):
        """Capture a specific application window by hwnd."""
        hwnd = self._source['hwnd']
        rect = win32gui.GetWindowRect(hwnd)
        x, y, x2, y2 = rect
        w = x2 - x
        h = y2 - y
        if w <= 0 or h <= 0:
            return False, None
        mon = {'left': x, 'top': y, 'width': w, 'height': h}
        img = self.sct.grab(mon)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (1280, 720))
        return True, frame

    def close(self):
        try:
            self.sct.close()
        except Exception:
            pass
