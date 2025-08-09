#include <windows.h>
#include <gdiplus.h>
#include <cstdio>
#include <Python.h>
#include <numpy/arrayobject.h>

using namespace Gdiplus;

static PyObject* py_capture_window(PyObject* self, PyObject* args) {
    unsigned long hwnd_val;
    if (!PyArg_ParseTuple(args, "k", &hwnd_val)) {
        return NULL;
    }
    HWND hwnd = (HWND)hwnd_val;
    RECT rc;
    if (!GetClientRect(hwnd, &rc)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get window rect");
        return NULL;
    }
    int width = rc.right - rc.left;
    int height = rc.bottom - rc.top;
    HDC hdcWindow = GetDC(hwnd);
    HDC hdcMemDC = CreateCompatibleDC(hdcWindow);
    HBITMAP hbmScreen = CreateCompatibleBitmap(hdcWindow, width, height);
    SelectObject(hdcMemDC, hbmScreen);
    BitBlt(hdcMemDC, 0, 0, width, height, hdcWindow, 0, 0, SRCCOPY);

    // Initialize GDI+
    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    if (GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL) != Ok) {
        DeleteObject(hbmScreen);
        DeleteDC(hdcMemDC);
        ReleaseDC(hwnd, hdcWindow);
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize GDI+");
        return NULL;
    }

    Bitmap* bmp = Bitmap::FromHBITMAP(hbmScreen, NULL);
    if (!bmp) {
        GdiplusShutdown(gdiplusToken);
        DeleteObject(hbmScreen);
        DeleteDC(hdcMemDC);
        ReleaseDC(hwnd, hdcWindow);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create GDI+ Bitmap");
        return NULL;
    }

    BitmapData bmpData;
    Rect rect(0, 0, width, height);
    if (bmp->LockBits(&rect, ImageLockModeRead, PixelFormat24bppRGB, &bmpData) != Ok) {
        delete bmp;
        GdiplusShutdown(gdiplusToken);
        DeleteObject(hbmScreen);
        DeleteDC(hdcMemDC);
        ReleaseDC(hwnd, hdcWindow);
        PyErr_SetString(PyExc_RuntimeError, "Failed to lock bitmap bits");
        return NULL;
    }

    int row_stride = ((width * 3 + 3) & ~3);
    int img_size = row_stride * height;
    unsigned char* data = (unsigned char*)malloc(img_size);
    if (!data) {
        bmp->UnlockBits(&bmpData);
        delete bmp;
        GdiplusShutdown(gdiplusToken);
        DeleteObject(hbmScreen);
        DeleteDC(hdcMemDC);
        ReleaseDC(hwnd, hdcWindow);
        PyErr_NoMemory();
        return NULL;
    }
    // Copy pixel data row by row
    for (int y = 0; y < height; ++y) {
        memcpy(data + y * row_stride, (unsigned char*)bmpData.Scan0 + y * bmpData.Stride, row_stride);
    }
    bmp->UnlockBits(&bmpData);
    delete bmp;
    GdiplusShutdown(gdiplusToken);
    DeleteObject(hbmScreen);
    DeleteDC(hdcMemDC);
    ReleaseDC(hwnd, hdcWindow);

    npy_intp dims[3] = {height, width, 3};
    PyObject* arr = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, data);
    if (!arr) {
        free(data);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create numpy array");
        return NULL;
    }
    PyArray_ENABLEFLAGS((PyArrayObject*)arr, NPY_ARRAY_OWNDATA);
    return arr;
}

static PyMethodDef WindowCaptureMethods[] = {
    {"capture_window", py_capture_window, METH_VARARGS, "Capture a window and return an np array (H, W, 3) in BGR order."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef windowcapturemodule = {
    PyModuleDef_HEAD_INIT,
    "window_capture",
    NULL,
    -1,
    WindowCaptureMethods
};

PyMODINIT_FUNC PyInit_window_capture(void) {
    import_array();
    return PyModule_Create(&windowcapturemodule);
}
