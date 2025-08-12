import time
import numpy as np
from pywinauto.application import Application
from window_capture import capture_window

# Launch Notepad as a test window
def main():
    app = Application().start('notepad.exe')
    time.sleep(1)  # Wait for the window to appear
    hwnd = app.top_window().handle
    print(f"Testing capture_window on Notepad hwnd: {hwnd}")
    arr = capture_window(hwnd)
    print(f"Captured array shape: {arr.shape}")
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3 and arr.shape[2] == 3
    print("Test passed!")
    app.kill()

if __name__ == "__main__":
    main()
