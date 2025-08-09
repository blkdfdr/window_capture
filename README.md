# window_capture: a simple GDI+ window capture Python extension

## Install
Add it to your dependencies:
```toml
[project]
dependencies = [
    "window_capture @ https://github.com/blkdfdr/window_capture.git"
]
```

## Usage
```python
import window_capture
from pywinauto import Desktop

# Get a window handle (HWND)
window = Desktop(backend="uia").window(title="Untitled - Notepad")
hwnd = window.handle

# Capture the window as a numpy array (height, width, 3) in BGR order
image = window_capture.capture_window(hwnd)
```

- The function `capture_window(hwnd)` returns a numpy array containing the window's image in BGR format.
- Requires: numpy, pywinauto (for handle discovery), Windows OS.
- Errors are raised as Python exceptions if capture fails.
