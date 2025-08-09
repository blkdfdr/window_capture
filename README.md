# window_capture: a simple GDI+ wrapper

## Install:
Just add it to your dependencies like this:
```toml
[project]
dependencies = [
    "window_capture @ https://github.com/blkdfdr/window_capture.git"
]
```

## Usage:
```python
from window_capture import capture
from pywinauto import Desktop

#Get a window handle
window = Desktop(backend="uia").window(title="Untitled - Notepad")

# Capture the window by handle
image = capture(window.handle) # A np array with the image data