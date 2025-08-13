#include <windows.h>
#include <cstdio>
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
#include <opencv2/core.hpp>
#include "../obs-game-capture-lib/game_capture.h"

#pragma comment(lib, "d3d11.lib")

#include <unordered_set>
#include <unordered_map>
#include <mutex>

// Global set and mutex to track injected HWNDs
static std::unordered_set<HWND> injected_hwnds;
static std::unordered_map<HWND, std::shared_ptr<GameCapture>> hwnd_to_gc;
static std::mutex injected_hwnds_mutex;

extern "C" static PyObject* py_inject_hook(PyObject* self, PyObject* args) {
    unsigned long hwnd_val;
    if (!PyArg_ParseTuple(args, "k", &hwnd_val)) {
        return NULL;
    }
    HWND hwnd = (HWND)hwnd_val;
    DWORD pid = 0;
    // Reserve the hwnd early to avoid race conditions
    {
        std::lock_guard<std::mutex> lock(injected_hwnds_mutex);
        if (injected_hwnds.find(hwnd) != injected_hwnds.end()) {
            Py_RETURN_FALSE; // Already injected or in progress
        }
        injected_hwnds.insert(hwnd); // Reserve
    }
    GetWindowThreadProcessId(hwnd, &pid);
    if (!pid) {
        std::lock_guard<std::mutex> lock(injected_hwnds_mutex);
        injected_hwnds.erase(hwnd);
        PyErr_SetString(PyExc_RuntimeError, "Failed to get process id from hwnd");
        return NULL;
    }
    auto gc_ptr = std::make_shared<GameCapture>(1, 1, 1, 1, "");
    {
        std::lock_guard<std::mutex> lock(injected_hwnds_mutex);
        hwnd_to_gc[hwnd] = gc_ptr;
    }
    HANDLE h = gc_ptr->inject_hook(pid);
    if (h == NULL || h == INVALID_HANDLE_VALUE) {
        std::lock_guard<std::mutex> lock(injected_hwnds_mutex);
        injected_hwnds.erase(hwnd);
        hwnd_to_gc.erase(hwnd);
        PyErr_SetString(PyExc_RuntimeError, "inject_hook failed");
        return NULL;
    }
    {
        std::lock_guard<std::mutex> lock(injected_hwnds_mutex);
        hwnd_to_gc[hwnd] = gc_ptr;
    }
    Py_RETURN_TRUE;
}

extern "C" static PyObject* py_capture_window(PyObject* self, PyObject* args) {
    unsigned long hwnd_val;
    if (!PyArg_ParseTuple(args, "k", &hwnd_val)) {
        return NULL;
    }
    HWND hwnd = (HWND)hwnd_val;

    // Ensure hook is injected before capturing
    PyObject* inject_result = py_inject_hook(nullptr, Py_BuildValue("k", hwnd_val));
    if (inject_result == NULL) {
        return NULL;
    }
    Py_DECREF(inject_result);
    RECT rc;
    if (!GetClientRect(hwnd, &rc)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get window rect");
        return NULL;
    }
    int width = rc.right - rc.left;
    int height = rc.bottom - rc.top;

    try {
        char window_title[256] = {0};
        GetWindowTextA(hwnd, window_title, sizeof(window_title)-1);
        std::string win_name(window_title);
        GameCapture gc(width, height, width, height, win_name);
        cv::Mat frame = gc.get_frame();
        if (!frame.empty()) {
            // Convert to numpy array (BGR order)
            npy_intp dims[3] = {frame.rows, frame.cols, frame.channels()};
            PyObject* arr = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, frame.data);
            if (!arr) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to create numpy array from GameCapture");
                return NULL;
            }
            // Prevent OpenCV from freeing the data
            PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
            frame.release();
            return arr;
        }
    } catch (const std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return NULL;
    }
    PyErr_SetString(PyExc_RuntimeError, "GameCapture failed to capture frame");
    return NULL;
}

static PyMethodDef WindowCaptureMethods[] = {
    {"capture_window", py_capture_window, METH_VARARGS, "Capture a window and return an np array (H, W, 3) in BGR order."},
    {"inject_hook", py_inject_hook, METH_VARARGS, "Inject hook into a window if not already injected."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef window_capture = {
    PyModuleDef_HEAD_INIT,
    "window_capture",
    "",
    -1,
    WindowCaptureMethods
};

extern "C" PyMODINIT_FUNC PyInit_window_capture(void) {
    import_array();
    if (PyArray_API == NULL) {
        PyErr_SetString(PyExc_ImportError, "NumPy C API not initialized");
        return NULL;
    }
    return PyModule_Create(&window_capture);
}
