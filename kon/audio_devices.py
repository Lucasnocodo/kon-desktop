"""Select a microphone without activating a phone's Continuity microphone.

PortAudio device indices are session-local. Persist a verified Core Audio UID
when available, or names and host APIs, and refuse missing or ambiguous saved
identities rather than changing inputs.
Discovery reads device metadata only; it never opens audio or changes macOS's
default input. On macOS, read Core Audio transport metadata so a renamed
Continuity microphone cannot bypass the device-name filter.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import sys
import unicodedata

import sounddevice as sd


@dataclass(frozen=True, slots=True)
class MicrophoneDevice:
    index: int
    name: str
    hostapi: str = ""
    device_id: str = ""


@dataclass(frozen=True, slots=True)
class _CoreAudioInput:
    name: str
    device_id: str
    transport: str


class _AudioPropertyAddress(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32),
                ("element", ctypes.c_uint32)]


def _fourcc(value: str) -> int:
    return int.from_bytes(value.encode("ascii"), "big")


class _CoreAudioProperties:
    """Read-only HAL properties using declarations from Apple's SDK headers.

    AudioHardwareBase.h defines names ('lnam'), persistent UIDs ('uid '), input
    streams ('stm#'), transport ('tran'), and Continuity's ccwd/ccwl/legacy ccap.
    AudioHardware.h defines the system object (1) and its device array ('dev#').
    No AudioDeviceStart, IOProc, property-setter, or audio stream API is used.
    https://developer.apple.com/documentation/coreaudio/audioobjectgetpropertydata(_:_:_:_:_:_:)
    """

    def __init__(self):
        audio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        self.cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        address_pointer = ctypes.POINTER(_AudioPropertyAddress)
        size_pointer = ctypes.POINTER(ctypes.c_uint32)
        self._size = audio.AudioObjectGetPropertyDataSize
        self._size.argtypes = [ctypes.c_uint32, address_pointer, ctypes.c_uint32,
                               ctypes.c_void_p, size_pointer]
        self._size.restype = ctypes.c_int32
        self._data = audio.AudioObjectGetPropertyData
        self._data.argtypes = [ctypes.c_uint32, address_pointer, ctypes.c_uint32,
                               ctypes.c_void_p, size_pointer, ctypes.c_void_p]
        self._data.restype = ctypes.c_int32
        self.cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
        self.cf.CFStringGetLength.restype = ctypes.c_long
        self.cf.CFStringGetMaximumSizeForEncoding.argtypes = [ctypes.c_long, ctypes.c_uint32]
        self.cf.CFStringGetMaximumSizeForEncoding.restype = ctypes.c_long
        self.cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_long, ctypes.c_uint32]
        self.cf.CFStringGetCString.restype = ctypes.c_ubyte
        self.cf.CFRelease.argtypes = [ctypes.c_void_p]
        self.cf.CFRelease.restype = None

    def read(self, object_id: int, selector: str, scope: str = "glob") -> bytes:
        address = _AudioPropertyAddress(_fourcc(selector), _fourcc(scope), 0)
        size = ctypes.c_uint32()
        status = self._size(object_id, ctypes.byref(address), 0, None, ctypes.byref(size))
        if status or size.value > 1024 * 1024:
            raise OSError("Core Audio property size unavailable")
        if not size.value:
            return b""
        allocated = size.value
        buffer = ctypes.create_string_buffer(allocated)
        status = self._data(object_id, ctypes.byref(address), 0, None, ctypes.byref(size), buffer)
        if status or size.value > allocated:
            raise OSError("Core Audio property unavailable")
        return buffer.raw[:size.value]

    def text(self, object_id: int, selector: str) -> str:
        raw = self.read(object_id, selector)
        if len(raw) != ctypes.sizeof(ctypes.c_void_p):
            raise OSError("Core Audio string property unavailable")
        string = ctypes.c_void_p.from_buffer_copy(raw).value
        if not string:
            return ""
        try:
            # kCFStringEncodingUTF8 from CFString.h; property CFStrings are
            # caller-owned according to AudioHardwareBase.h and need release.
            encoding = 0x08000100
            length = self.cf.CFStringGetLength(string)
            capacity = self.cf.CFStringGetMaximumSizeForEncoding(length, encoding) + 1
            if not 0 < capacity <= 1024 * 1024:
                raise OSError("Core Audio string size invalid")
            buffer = ctypes.create_string_buffer(capacity)
            if not self.cf.CFStringGetCString(string, buffer, capacity, encoding):
                raise OSError("Core Audio string conversion failed")
            return buffer.value.decode("utf-8")
        finally:
            self.cf.CFRelease(string)


def _coreaudio_inputs() -> tuple[_CoreAudioInput, ...] | None:
    """Return input metadata on macOS; None means a non-macOS platform."""
    if sys.platform != "darwin":
        return None
    properties = _CoreAudioProperties()
    raw_ids = properties.read(1, "dev#")
    if len(raw_ids) % ctypes.sizeof(ctypes.c_uint32):
        raise OSError("Core Audio device array invalid")
    identifiers = (ctypes.c_uint32 * (len(raw_ids) // 4)).from_buffer_copy(raw_ids)
    result = []
    for identifier in identifiers:
        try:
            if not properties.read(identifier, "stm#", "inpt"):
                continue
            raw_transport = properties.read(identifier, "tran")
            if len(raw_transport) != 4:
                continue
            transport = ctypes.c_uint32.from_buffer_copy(raw_transport).value
            result.append(_CoreAudioInput(
                properties.text(identifier, "lnam"), properties.text(identifier, "uid "),
                transport.to_bytes(4, "big").decode("ascii", errors="replace"),
            ))
        except (OSError, UnicodeError):
            # A hot-unplugged or unverifiable device has no safe name mapping.
            continue
    return tuple(result)


_PHONE_NAMES = (
    "iphone", "ipad", "continuity", "接續互通", "接续互通",
    "連係カメラ", "連携カメラ", "연속성 카메라",
)
_LOOPBACK_NAMES = (
    "blackhole", "soundflower", "loopback", "vb-cable", "vb-audio",
    "background music", "backgroundmusic", "groundcontrol", "ground control",
    "zoom audio device", "microsoft teams audio", "aggregate device",
    "multi-output device", "聚集裝置", "聚集设备", "多重輸出裝置", "多输出设备",
)
_BLOCKED_TRANSPORTS = {"ccwd", "ccwl", "ccap", "virt", "grup"}
_NO_MICROPHONE = (
    "找不到可用的非 iPhone 麥克風。請接上 USB 攝影機的麥克風、USB 麥克風或耳麥，"
    "再重新整理麥克風清單；Mac mini 沒有內建麥克風。"
)


def _normalized(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


def _eligible(name: str) -> bool:
    normalized = _normalized(name)
    return bool(normalized.strip()) and not any(
        marker in normalized for marker in (*_PHONE_NAMES, *_LOOPBACK_NAMES)
    )


def list_microphones() -> tuple[MicrophoneDevice, ...]:
    """List input devices, excluding phone/Continuity and known loopback names.

    On macOS, exact unique names connect the PortAudio inventory to Core Audio
    transport and UID metadata. Unknown or ambiguous Core Audio inputs are
    excluded rather than guessed. Other host APIs retain the name filter.
    """
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as exc:
        raise ValueError("無法讀取麥克風清單，請確認音訊裝置已連接後重試。") from exc
    result = []
    for index, info in enumerate(devices):
        name = str(info.get("name", ""))
        if info.get("max_input_channels", 0) <= 0 or not _eligible(name):
            continue
        api_index = info.get("hostapi")
        hostapi = ""
        if isinstance(api_index, int) and 0 <= api_index < len(hostapis):
            hostapi = str(hostapis[api_index].get("name", ""))
        result.append(MicrophoneDevice(index, name, hostapi))
    if not any(_normalized(device.hostapi).replace(" ", "") == "coreaudio" for device in result):
        return tuple(result)
    try:
        native = _coreaudio_inputs()
    except Exception as exc:
        raise ValueError("無法確認 macOS 麥克風來源，請重新連接 USB 麥克風後重新整理清單。") from exc
    if native is None:
        return tuple(result)
    verified = []
    for device in result:
        if _normalized(device.hostapi).replace(" ", "") != "coreaudio":
            verified.append(device)
            continue
        matches = [item for item in native if item.name == device.name]
        # A duplicate name cannot establish which PortAudio index is safe or
        # which persistent UID belongs to it. Do not infer ordering across APIs.
        if (len(matches) != 1 or matches[0].transport in _BLOCKED_TRANSPORTS or
                sum(other.name == device.name and other.hostapi == device.hostapi
                    for other in result) != 1):
            continue
        item = matches[0]
        uid = item.device_id if item.device_id and sum(
            other.device_id == item.device_id for other in native
        ) == 1 else ""
        verified.append(MicrophoneDevice(device.index, device.name, device.hostapi, uid))
    return tuple(verified)


def microphone_setting(device: MicrophoneDevice) -> dict[str, str]:
    """Serialize microphone identity; never persist a PortAudio index."""
    setting = {"name": device.name, "hostapi": device.hostapi}
    if device.device_id:
        setting["device_id"] = device.device_id
    return setting


def _fallback_rank(device: MicrophoneDevice) -> int:
    name = _normalized(device.name)
    if "emeet" in name:
        return 0
    if any(marker in name for marker in (
        "usb", "headset", "耳麥", "耳麦", "headphone", "airpods",
    )):
        return 1
    if any(marker in name for marker in (
        "built-in", "builtin", "macbook", "內建", "内建", "内蔵", "내장",
    )):
        return 2
    return 3


def resolve_microphone(preferred: dict | str | None = None) -> MicrophoneDevice:
    """Resolve a saved identity, or choose a safe input for automatic mode.

    Automatic mode honors the current default only when it is eligible. When
    macOS defaults to an iPhone, prefer the EMEET webcam's microphone, then an
    external USB/headset microphone or built-in microphone. Explicit choices
    never fall back to another device if disconnected or ambiguous.
    """
    if preferred is not None and not isinstance(preferred, (dict, str)):
        raise ValueError("麥克風設定必須使用裝置名稱與音訊介面，不能只儲存裝置編號。")
    devices = list_microphones()
    if preferred is not None:
        if isinstance(preferred, str):
            name, hostapi, device_id = preferred, None, ""
        else:
            name = preferred.get("name", "")
            hostapi = preferred.get("hostapi")
            device_id = preferred.get("device_id", "")
            if (not isinstance(name, str) or not isinstance(device_id, str) or
                    (hostapi is not None and not isinstance(hostapi, str))):
                raise ValueError("儲存的麥克風名稱或裝置識別資訊無效，請重新選擇麥克風。")
        if not name.strip() and not device_id:
            raise ValueError("儲存的麥克風設定缺少裝置識別資訊，請重新選擇麥克風。")
        if name and not _eligible(name):
            raise ValueError("請選擇 USB 攝影機、USB 麥克風或耳麥，避免啟用 iPhone／接續互通麥克風。")
        if device_id:
            matches = [device for device in devices if device.device_id == device_id]
        else:
            matches = [device for device in devices if device.name == name and
                       (hostapi is None or device.hostapi == hostapi)]
        if not matches:
            raise ValueError("找不到已選的麥克風，請重新連接裝置，或重新整理清單並選擇麥克風。")
        if len(matches) != 1:
            raise ValueError("有多個同名麥克風，無法確認已儲存的裝置；請更改裝置名稱後重新選擇。")
        return matches[0]
    if not devices:
        raise ValueError(_NO_MICROPHONE)
    try:
        default_index = sd.default.device[0]
    except Exception:
        default_index = -1
    for device in devices:
        if device.index == default_index:
            return device
    return min(devices, key=_fallback_rank)
