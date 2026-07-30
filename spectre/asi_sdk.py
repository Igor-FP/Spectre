"""ctypes binding for the ZWO ASI camera SDK (ASICamera2.dll).

Mirrors include/ASICamera2.h from the ZWO SDK (tested against SDK 1.21 with an
ASI290MM).  Only the entry points this application needs are bound; adding more
is one line in `_bind`.

All wrappers raise `ASIError` on a non-zero ASI_ERROR_CODE, except where a
non-success code is a normal outcome (`get_video_data` timeout).
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, byref, c_char, c_double, c_float, c_int, c_long, c_ubyte
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Enums (ASICamera2.h)
# ---------------------------------------------------------------------------


class ImgType(IntEnum):
    RAW8 = 0
    RGB24 = 1
    RAW16 = 2
    Y8 = 3
    END = -1


class BayerPattern(IntEnum):
    RG = 0
    BG = 1
    GR = 2
    GB = 3


class ControlType(IntEnum):
    GAIN = 0
    EXPOSURE = 1
    GAMMA = 2
    WB_R = 3
    WB_B = 4
    OFFSET = 5
    BANDWIDTHOVERLOAD = 6
    OVERCLOCK = 7
    TEMPERATURE = 8  # 10 * degrees C
    FLIP = 9
    AUTO_MAX_GAIN = 10
    AUTO_MAX_EXP = 11  # milliseconds
    AUTO_TARGET_BRIGHTNESS = 12
    HARDWARE_BIN = 13
    HIGH_SPEED_MODE = 14
    COOLER_POWER_PERC = 15
    TARGET_TEMP = 16
    COOLER_ON = 17
    MONO_BIN = 18
    FAN_ON = 19
    PATTERN_ADJUST = 20
    ANTI_DEW_HEATER = 21


class ErrorCode(IntEnum):
    SUCCESS = 0
    INVALID_INDEX = 1
    INVALID_ID = 2
    INVALID_CONTROL_TYPE = 3
    CAMERA_CLOSED = 4
    CAMERA_REMOVED = 5
    INVALID_PATH = 6
    INVALID_FILEFORMAT = 7
    INVALID_SIZE = 8
    INVALID_IMGTYPE = 9
    OUTOF_BOUNDARY = 10
    TIMEOUT = 11
    INVALID_SEQUENCE = 12
    BUFFER_TOO_SMALL = 13
    VIDEO_MODE_ACTIVE = 14
    EXPOSURE_IN_PROGRESS = 15
    GENERAL_ERROR = 16
    INVALID_MODE = 17
    END = 18


class ExposureStatus(IntEnum):
    IDLE = 0
    WORKING = 1
    SUCCESS = 2
    FAILED = 3


class CameraMode(IntEnum):
    NORMAL = 0
    TRIG_SOFT_EDGE = 1
    TRIG_RISE_EDGE = 2
    TRIG_FALL_EDGE = 3
    TRIG_SOFT_LEVEL = 4
    TRIG_HIGH_LEVEL = 5
    TRIG_LOW_LEVEL = 6
    END = -1


class FlipStatus(IntEnum):
    NONE = 0
    HORIZ = 1
    VERT = 2
    BOTH = 3


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


class _CameraInfo(ctypes.Structure):
    _fields_ = [
        ("Name", c_char * 64),
        ("CameraID", c_int),
        ("MaxHeight", c_long),
        ("MaxWidth", c_long),
        ("IsColorCam", c_int),
        ("BayerPattern", c_int),
        ("SupportedBins", c_int * 16),
        ("SupportedVideoFormat", c_int * 8),
        ("PixelSize", c_double),
        ("MechanicalShutter", c_int),
        ("ST4Port", c_int),
        ("IsCoolerCam", c_int),
        ("IsUSB3Host", c_int),
        ("IsUSB3Camera", c_int),
        ("ElecPerADU", c_float),
        ("BitDepth", c_int),
        ("IsTriggerCam", c_int),
        ("Unused", c_char * 16),
    ]


class _ControlCaps(ctypes.Structure):
    _fields_ = [
        ("Name", c_char * 64),
        ("Description", c_char * 128),
        ("MaxValue", c_long),
        ("MinValue", c_long),
        ("DefaultValue", c_long),
        ("IsAutoSupported", c_int),
        ("IsWritable", c_int),
        ("ControlType", c_int),
        ("Unused", c_char * 32),
    ]


class _ID(ctypes.Structure):
    _fields_ = [("id", c_ubyte * 8)]


@dataclass(frozen=True)
class CameraInfo:
    """Snapshot of ASI_CAMERA_INFO, decoded into Python types."""

    camera_id: int
    name: str
    max_width: int
    max_height: int
    is_color: bool
    bayer_pattern: int
    supported_bins: tuple
    supported_formats: tuple
    pixel_size_um: float
    mechanical_shutter: bool
    st4_port: bool
    is_cooler_cam: bool
    is_usb3_host: bool
    is_usb3_camera: bool
    elec_per_adu: float
    bit_depth: int
    is_trigger_cam: bool

    @classmethod
    def _from_struct(cls, s: _CameraInfo) -> "CameraInfo":
        return cls(
            camera_id=s.CameraID,
            name=s.Name.decode("utf-8", "replace"),
            max_width=int(s.MaxWidth),
            max_height=int(s.MaxHeight),
            is_color=bool(s.IsColorCam),
            bayer_pattern=int(s.BayerPattern),
            # 0 terminates the bin list, -1 (IMG_END) the format list
            supported_bins=tuple(b for b in s.SupportedBins if b != 0),
            supported_formats=tuple(
                _take_until(s.SupportedVideoFormat, ImgType.END)
            ),
            pixel_size_um=float(s.PixelSize),
            mechanical_shutter=bool(s.MechanicalShutter),
            st4_port=bool(s.ST4Port),
            is_cooler_cam=bool(s.IsCoolerCam),
            is_usb3_host=bool(s.IsUSB3Host),
            is_usb3_camera=bool(s.IsUSB3Camera),
            elec_per_adu=float(s.ElecPerADU),
            bit_depth=int(s.BitDepth),
            is_trigger_cam=bool(s.IsTriggerCam),
        )


def _take_until(values: Sequence[int], terminator: int) -> list:
    """SupportedVideoFormat is terminated by ASI_IMG_END, then zero-padded.

    RAW8 is 0, so a plain `if v` filter would drop it: stop at the terminator
    instead, and ignore trailing zeros beyond it.
    """
    out = []
    for v in values:
        if v == terminator:
            break
        out.append(ImgType(v))
    return out


@dataclass(frozen=True)
class ControlCaps:
    """Snapshot of ASI_CONTROL_CAPS."""

    control_type: int
    name: str
    description: str
    min_value: int
    max_value: int
    default_value: int
    is_auto_supported: bool
    is_writable: bool

    @classmethod
    def _from_struct(cls, s: _ControlCaps) -> "ControlCaps":
        return cls(
            control_type=int(s.ControlType),
            name=s.Name.decode("utf-8", "replace"),
            description=s.Description.decode("utf-8", "replace"),
            min_value=int(s.MinValue),
            max_value=int(s.MaxValue),
            default_value=int(s.DefaultValue),
            is_auto_supported=bool(s.IsAutoSupported),
            is_writable=bool(s.IsWritable),
        )


class ASIError(RuntimeError):
    def __init__(self, code: int, func: str):
        try:
            name = ErrorCode(code).name
        except ValueError:
            name = "UNKNOWN"
        super().__init__(f"{func} failed: {name} ({code})")
        self.code = code
        self.func = func


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_LIB_NAME = "ASICamera2.dll" if sys.platform == "win32" else "libASICamera2.so"

#: Searched in order when `load()` gets no explicit path.
DEFAULT_DLL_PATHS = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib", _LIB_NAME),
    r"C:\src\ZWOCameraSDK\lib\x64\ASICamera2.dll",
    r"C:\Program Files\ASIStudio\ASICamera2.dll",
    r"C:\Program Files (x86)\Common Files\ASCOM\ZWO\ASICamera2.dll",
    _LIB_NAME,  # anywhere on PATH / LD_LIBRARY_PATH
)

_dll: Optional[ctypes.CDLL] = None
_dll_path: Optional[str] = None


def candidate_paths() -> list:
    """DLL paths to try, honouring the SPECTRE_ASI_DLL environment variable."""
    env = os.environ.get("SPECTRE_ASI_DLL")
    return ([env] if env else []) + list(DEFAULT_DLL_PATHS)


def load(path: Optional[str] = None) -> str:
    """Load ASICamera2. Returns the path that was loaded.

    Idempotent: a second call with no path is a no-op.
    """
    global _dll, _dll_path
    if _dll is not None and path is None:
        return _dll_path

    tried = []
    for candidate in [path] if path else candidate_paths():
        try:
            directory = os.path.dirname(os.path.abspath(candidate))
            if sys.platform == "win32" and os.path.isdir(directory):
                # Let the loader find sibling dependencies of the DLL.
                os.add_dll_directory(directory)
            dll = ctypes.CDLL(candidate)
        except OSError as exc:
            tried.append(f"  {candidate}: {exc}")
            continue
        _bind(dll)
        _dll, _dll_path = dll, candidate
        return candidate

    raise OSError(
        "Could not load the ZWO ASI SDK. Set SPECTRE_ASI_DLL to the full path of "
        f"{_LIB_NAME}, or drop it into <project>/lib/.\nTried:\n" + "\n".join(tried)
    )


def is_loaded() -> bool:
    return _dll is not None


def library_path() -> Optional[str]:
    return _dll_path


def _bind(dll: ctypes.CDLL) -> None:
    P_INFO = POINTER(_CameraInfo)
    P_CAPS = POINTER(_ControlCaps)
    P_INT = POINTER(c_int)
    P_LONG = POINTER(c_long)
    P_BYTE = POINTER(c_ubyte)

    signatures = {
        "ASIGetNumOfConnectedCameras": ([], c_int),
        "ASIGetCameraProperty": ([P_INFO, c_int], c_int),
        "ASIGetCameraPropertyByID": ([c_int, P_INFO], c_int),
        "ASIOpenCamera": ([c_int], c_int),
        "ASIInitCamera": ([c_int], c_int),
        "ASICloseCamera": ([c_int], c_int),
        "ASIGetNumOfControls": ([c_int, P_INT], c_int),
        "ASIGetControlCaps": ([c_int, c_int, P_CAPS], c_int),
        "ASIGetControlValue": ([c_int, c_int, P_LONG, P_INT], c_int),
        "ASISetControlValue": ([c_int, c_int, c_long, c_int], c_int),
        "ASISetROIFormat": ([c_int, c_int, c_int, c_int, c_int], c_int),
        "ASIGetROIFormat": ([c_int, P_INT, P_INT, P_INT, P_INT], c_int),
        "ASISetStartPos": ([c_int, c_int, c_int], c_int),
        "ASIGetStartPos": ([c_int, P_INT, P_INT], c_int),
        "ASIGetDroppedFrames": ([c_int, P_INT], c_int),
        "ASIDisableDarkSubtract": ([c_int], c_int),
        "ASIStartVideoCapture": ([c_int], c_int),
        "ASIStopVideoCapture": ([c_int], c_int),
        "ASIGetVideoData": ([c_int, P_BYTE, c_long, c_int], c_int),
        "ASIStartExposure": ([c_int, c_int], c_int),
        "ASIStopExposure": ([c_int], c_int),
        "ASIGetExpStatus": ([c_int, P_INT], c_int),
        "ASIGetDataAfterExp": ([c_int, P_BYTE, c_long], c_int),
        "ASIGetSDKVersion": ([], ctypes.c_char_p),
        "ASIGetSerialNumber": ([c_int, POINTER(_ID)], c_int),
        "ASISetCameraMode": ([c_int, c_int], c_int),
        "ASIGetCameraMode": ([c_int, P_INT], c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        func = getattr(dll, name, None)
        if func is None:  # older SDK: leave it unbound, callers will AttributeError
            continue
        func.argtypes = argtypes
        func.restype = restype


def _lib() -> ctypes.CDLL:
    if _dll is None:
        load()
    return _dll


def _check(code: int, func: str) -> None:
    if code != ErrorCode.SUCCESS:
        raise ASIError(code, func)


def _as_byte_ptr(array: np.ndarray):
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("buffer must be C-contiguous")
    return array.ctypes.data_as(POINTER(c_ubyte))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def get_sdk_version() -> str:
    return _lib().ASIGetSDKVersion().decode("ascii", "replace")


def get_num_connected_cameras() -> int:
    return int(_lib().ASIGetNumOfConnectedCameras())


def get_camera_property(index: int) -> CameraInfo:
    """Camera info by *enumeration index* (valid before opening the camera)."""
    info = _CameraInfo()
    _check(_lib().ASIGetCameraProperty(byref(info), index), "ASIGetCameraProperty")
    return CameraInfo._from_struct(info)


def list_cameras() -> list:
    return [get_camera_property(i) for i in range(get_num_connected_cameras())]


def open_camera(camera_id: int) -> None:
    _check(_lib().ASIOpenCamera(camera_id), "ASIOpenCamera")


def init_camera(camera_id: int) -> None:
    _check(_lib().ASIInitCamera(camera_id), "ASIInitCamera")


def close_camera(camera_id: int) -> None:
    _check(_lib().ASICloseCamera(camera_id), "ASICloseCamera")


def get_control_caps(camera_id: int) -> list:
    count = c_int()
    _check(_lib().ASIGetNumOfControls(camera_id, byref(count)), "ASIGetNumOfControls")
    caps = []
    for i in range(count.value):
        c = _ControlCaps()
        _check(_lib().ASIGetControlCaps(camera_id, i, byref(c)), "ASIGetControlCaps")
        caps.append(ControlCaps._from_struct(c))
    return caps


def get_control_value(camera_id: int, control: int) -> tuple:
    """Returns (value, is_auto)."""
    value, auto = c_long(), c_int()
    _check(
        _lib().ASIGetControlValue(camera_id, int(control), byref(value), byref(auto)),
        "ASIGetControlValue",
    )
    return int(value.value), bool(auto.value)


def set_control_value(camera_id: int, control: int, value: int, auto: bool = False) -> None:
    _check(
        _lib().ASISetControlValue(camera_id, int(control), int(value), 1 if auto else 0),
        "ASISetControlValue",
    )


def set_roi_format(camera_id: int, width: int, height: int, binning: int, img_type: int) -> None:
    _check(
        _lib().ASISetROIFormat(camera_id, width, height, binning, int(img_type)),
        "ASISetROIFormat",
    )


def get_roi_format(camera_id: int) -> tuple:
    """Returns (width, height, binning, img_type)."""
    w, h, b, t = c_int(), c_int(), c_int(), c_int()
    _check(
        _lib().ASIGetROIFormat(camera_id, byref(w), byref(h), byref(b), byref(t)),
        "ASIGetROIFormat",
    )
    return w.value, h.value, b.value, ImgType(t.value)


def set_start_pos(camera_id: int, x: int, y: int) -> None:
    _check(_lib().ASISetStartPos(camera_id, x, y), "ASISetStartPos")


def get_start_pos(camera_id: int) -> tuple:
    x, y = c_int(), c_int()
    _check(_lib().ASIGetStartPos(camera_id, byref(x), byref(y)), "ASIGetStartPos")
    return x.value, y.value


def get_dropped_frames(camera_id: int) -> int:
    n = c_int()
    _check(_lib().ASIGetDroppedFrames(camera_id, byref(n)), "ASIGetDroppedFrames")
    return n.value


def disable_dark_subtract(camera_id: int) -> None:
    _check(_lib().ASIDisableDarkSubtract(camera_id), "ASIDisableDarkSubtract")


def start_video_capture(camera_id: int) -> None:
    _check(_lib().ASIStartVideoCapture(camera_id), "ASIStartVideoCapture")


def stop_video_capture(camera_id: int) -> None:
    _check(_lib().ASIStopVideoCapture(camera_id), "ASIStopVideoCapture")


def get_video_data(camera_id: int, buffer: np.ndarray, timeout_ms: int) -> bool:
    """Fill `buffer` with the next video frame.

    Returns False on ASI_ERROR_TIMEOUT (a normal outcome while waiting), raises
    for every other failure.
    """
    code = _lib().ASIGetVideoData(camera_id, _as_byte_ptr(buffer), buffer.nbytes, int(timeout_ms))
    if code == ErrorCode.TIMEOUT:
        return False
    _check(code, "ASIGetVideoData")
    return True


def start_exposure(camera_id: int, is_dark: bool = False) -> None:
    _check(_lib().ASIStartExposure(camera_id, 1 if is_dark else 0), "ASIStartExposure")


def stop_exposure(camera_id: int) -> None:
    _check(_lib().ASIStopExposure(camera_id), "ASIStopExposure")


def get_exp_status(camera_id: int) -> ExposureStatus:
    status = c_int()
    _check(_lib().ASIGetExpStatus(camera_id, byref(status)), "ASIGetExpStatus")
    return ExposureStatus(status.value)


def get_data_after_exp(camera_id: int, buffer: np.ndarray) -> None:
    _check(
        _lib().ASIGetDataAfterExp(camera_id, _as_byte_ptr(buffer), buffer.nbytes),
        "ASIGetDataAfterExp",
    )


def get_serial_number(camera_id: int) -> Optional[str]:
    """Hex serial number, or None if the camera does not report one."""
    sn = _ID()
    code = _lib().ASIGetSerialNumber(camera_id, byref(sn))
    if code != ErrorCode.SUCCESS:
        return None
    return "".join(f"{b:02x}" for b in sn.id)


def set_camera_mode(camera_id: int, mode: int) -> None:
    _check(_lib().ASISetCameraMode(camera_id, int(mode)), "ASISetCameraMode")


def get_camera_mode(camera_id: int) -> CameraMode:
    mode = c_int()
    _check(_lib().ASIGetCameraMode(camera_id, byref(mode)), "ASIGetCameraMode")
    return CameraMode(mode.value)
