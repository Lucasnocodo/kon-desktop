import ctypes
import unittest
from unittest.mock import patch

from kon.audio_devices import (
    MicrophoneDevice,
    list_microphones,
    microphone_setting,
    resolve_microphone,
    _CoreAudioInput,
    _coreaudio_inputs,
    _fourcc,
)


def input_device(name, *, hostapi=0, channels=1):
    return {"name": name, "hostapi": hostapi, "max_input_channels": channels}


class MicrophoneDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.devices = patch("kon.audio_devices.sd.query_devices", return_value=[]).start()
        self.apis = patch("kon.audio_devices.sd.query_hostapis", return_value=[{"name": "Core Audio"}]).start()
        self.default = patch("kon.audio_devices.sd.default", device=(-1, -1)).start()
        self.stream = patch("kon.audio_devices.sd.InputStream").start()
        self.native = patch("kon.audio_devices._coreaudio_inputs", return_value=None).start()
        self.addCleanup(patch.stopall)

    def test_metadata_enumeration_excludes_output_and_never_opens_audio(self):
        self.devices.return_value = [
            input_device("Speakers", channels=0),
            input_device("EMEET C960"),
            input_device("iPhone Microphone"),
        ]
        self.assertEqual(list_microphones(), (MicrophoneDevice(1, "EMEET C960", "Core Audio"),))
        self.stream.assert_not_called()
        self.assertEqual(self.default.device, (-1, -1))

    def test_phone_and_continuity_names_including_localized_names_are_excluded(self):
        for name in (
            "Lucas 的 iPhone 麥克風", "iPad Microphone", "Continuity Camera Audio",
            "接續互通相機麥克風", "接续互通相机", "連係カメラ", "連携カメラ",
            "연속성 카메라", "ＩＰＨＯＮＥ Microphone",
        ):
            with self.subTest(name=name):
                self.devices.return_value = [input_device(name)]
                self.assertEqual(list_microphones(), ())

    def test_known_virtual_and_aggregate_devices_are_excluded(self):
        self.devices.return_value = [input_device(name) for name in (
            "BlackHole 2ch", "Soundflower (2ch)", "Loopback Audio", "VB-Cable",
            "Background Music", "Aggregate Device", "聚集裝置", "USB Headset",
        )]
        self.assertEqual([device.name for device in list_microphones()], ["USB Headset"])

    def test_unavailable_inventory_reports_actionable_error(self):
        self.devices.side_effect = RuntimeError("PortAudio failed")
        with self.assertRaisesRegex(ValueError, "無法讀取麥克風清單"):
            list_microphones()
        self.stream.assert_not_called()

    def test_safe_os_default_is_honored(self):
        self.devices.return_value = [input_device("EMEET C960"), input_device("USB Headset")]
        self.default.device = (1, -1)
        self.assertEqual(resolve_microphone().name, "USB Headset")

    def test_iphone_default_is_replaced_by_emeet_without_changing_system_default(self):
        self.devices.return_value = [
            input_device("BlackHole 2ch"), input_device("iPhone Microphone"),
            input_device("MacBook Air Microphone"), input_device("EMEET C960"),
        ]
        self.default.device = (1, -1)
        self.assertEqual(resolve_microphone(), MicrophoneDevice(3, "EMEET C960", "Core Audio"))
        self.assertEqual(self.default.device, (1, -1))
        self.stream.assert_not_called()

    def test_virtual_default_is_replaced_by_physical_input(self):
        self.devices.return_value = [input_device("BlackHole 2ch"), input_device("USB Microphone")]
        self.default.device = (0, -1)
        self.assertEqual(resolve_microphone().index, 1)

    def test_automatic_fallback_prefers_usb_then_built_in(self):
        self.devices.return_value = [
            input_device("Unknown input"), input_device("MacBook Air Microphone"),
            input_device("USB Headset"),
        ]
        self.assertEqual(resolve_microphone().index, 2)
        self.devices.return_value.pop()
        self.assertEqual(resolve_microphone().index, 1)

    def test_phone_only_or_no_devices_reports_mac_mini_mic_requirement(self):
        for devices in ([], [input_device("iPhone Microphone")]):
            with self.subTest(devices=devices):
                self.devices.return_value = devices
                with self.assertRaisesRegex(ValueError, "Mac mini 沒有內建麥克風"):
                    resolve_microphone()
        self.stream.assert_not_called()

    def test_saved_selection_tracks_identity_after_index_reordering(self):
        selected = MicrophoneDevice(7, "EMEET C960", "Core Audio")
        setting = microphone_setting(selected)
        self.assertEqual(setting, {"name": "EMEET C960", "hostapi": "Core Audio"})
        self.devices.return_value = [input_device("EMEET C960"), input_device("USB Headset")]
        self.default.device = (1, -1)
        self.assertEqual(resolve_microphone(setting).index, 0)

    def test_disconnected_saved_selection_never_falls_back(self):
        self.devices.return_value = [input_device("USB Headset")]
        self.default.device = (0, -1)
        with self.assertRaisesRegex(ValueError, "找不到已選的麥克風"):
            resolve_microphone({"name": "EMEET C960", "hostapi": "Core Audio", "index": 0})

    def test_same_name_on_different_api_requires_saved_api(self):
        self.apis.return_value = [{"name": "Core Audio"}, {"name": "Other API"}]
        self.devices.return_value = [input_device("USB Mic"), input_device("USB Mic", hostapi=1)]
        with self.assertRaisesRegex(ValueError, "多個同名麥克風"):
            resolve_microphone("USB Mic")
        self.assertEqual(resolve_microphone({"name": "USB Mic", "hostapi": "Other API"}).index, 1)

    def test_identical_name_and_api_never_uses_saved_index_to_disambiguate(self):
        self.devices.return_value = [input_device("USB Mic"), input_device("USB Mic")]
        with self.assertRaisesRegex(ValueError, "多個同名麥克風"):
            resolve_microphone({"name": "USB Mic", "hostapi": "Core Audio", "index": 1})

    def test_explicit_phone_is_rejected_even_when_another_mic_is_available(self):
        self.devices.return_value = [input_device("EMEET C960"), input_device("iPhone Microphone")]
        with self.assertRaisesRegex(ValueError, "避免啟用 iPhone"):
            resolve_microphone("iPhone Microphone")

    def test_invalid_or_index_only_preferences_are_rejected(self):
        for preferred in (0, False, [], "", {}, {"index": 1}, {"name": 2}, {"name": "USB", "hostapi": 4}):
            with self.subTest(preferred=preferred), self.assertRaises(ValueError):
                resolve_microphone(preferred)

    def test_names_are_exact_not_portaudio_substrings(self):
        self.devices.return_value = [input_device("USB Mic"), input_device("USB Mic 2")]
        self.assertEqual(resolve_microphone("USB Mic").index, 0)
        with self.assertRaisesRegex(ValueError, "找不到已選的麥克風"):
            resolve_microphone("USB")

    def test_verified_uid_has_priority_and_does_not_fallback_to_reused_name(self):
        devices = (MicrophoneDevice(2, "USB Mic", "Core Audio", "new-uid"),)
        with patch("kon.audio_devices.list_microphones", return_value=devices):
            with self.assertRaisesRegex(ValueError, "找不到已選的麥克風"):
                resolve_microphone({"device_id": "old-uid", "name": "USB Mic", "hostapi": "Core Audio"})
            self.assertEqual(resolve_microphone({"device_id": "new-uid"}), devices[0])
        self.assertEqual(microphone_setting(devices[0])["device_id"], "new-uid")

    def test_native_transport_excludes_renamed_continuity_virtual_and_aggregate_inputs(self):
        for transport in ("ccwd", "ccwl", "ccap", "virt", "grup"):
            with self.subTest(transport=transport):
                self.devices.return_value = [input_device("Lucas 的麥克風"), input_device("EMEET C960")]
                self.native.return_value = (
                    _CoreAudioInput("Lucas 的麥克風", "phone-uid", transport),
                    _CoreAudioInput("EMEET C960", "usb-uid", "usb "),
                )
                self.default.device = (0, -1)
                self.assertEqual(resolve_microphone(), MicrophoneDevice(1, "EMEET C960", "Core Audio", "usb-uid"))
        self.stream.assert_not_called()

    def test_native_uid_maps_only_exact_unambiguous_names_and_apis(self):
        self.devices.return_value = [input_device("USB Mic"), input_device("USB Mic 2")]
        self.native.return_value = (_CoreAudioInput("USB Mic", "usb-uid", "usb "),)
        self.assertEqual(list_microphones(), (MicrophoneDevice(0, "USB Mic", "Core Audio", "usb-uid"),))
        self.devices.return_value = [input_device("USB Mic"), input_device("USB Mic")]
        self.assertEqual(list_microphones(), ())
        self.devices.return_value = [input_device("USB Mic")]
        self.native.return_value = (
            _CoreAudioInput("USB Mic", "usb-uid", "usb "),
            _CoreAudioInput("USB Mic", "phone-uid", "ccwl"),
        )
        self.assertEqual(list_microphones(), ())

    def test_native_name_mismatch_and_failed_inventory_never_guess(self):
        self.devices.return_value = [input_device("Unidentified input")]
        self.native.return_value = ()
        self.assertEqual(list_microphones(), ())
        self.native.side_effect = OSError("HAL unavailable")
        with self.assertRaisesRegex(ValueError, "無法確認 macOS 麥克風來源"):
            list_microphones()

    def test_non_coreaudio_api_does_not_get_coreaudio_uid_or_native_filter(self):
        self.devices.return_value = [input_device("USB Mic", hostapi=1)]
        self.apis.return_value = [{"name": "Core Audio"}, {"name": "Other API"}]
        self.native.return_value = (_CoreAudioInput("USB Mic", "phone-uid", "ccwl"),)
        self.assertEqual(list_microphones(), (MicrophoneDevice(0, "USB Mic", "Other API"),))
        self.native.assert_not_called()

    def test_native_uid_keeps_selected_device_after_rename_and_reindex(self):
        self.devices.return_value = [input_device("Renamed USB Mic")]
        self.native.return_value = (_CoreAudioInput("Renamed USB Mic", "usb-uid", "usb "),)
        selected = resolve_microphone({"name": "Old USB Mic", "hostapi": "Core Audio", "device_id": "usb-uid"})
        self.assertEqual(selected, MicrophoneDevice(0, "Renamed USB Mic", "Core Audio", "usb-uid"))


class CoreAudioMetadataTests(unittest.TestCase):
    def test_native_reader_uses_readonly_properties_and_input_scope(self):
        def uints(*values):
            return bytes((ctypes.c_uint32 * len(values))(*values))

        def read(identifier, selector, scope="glob"):
            if identifier == 1 and selector == "dev#":
                return uints(10, 11, 12)
            if selector == "stm#" and scope == "inpt":
                return uints(100) if identifier != 11 else b""
            if selector == "tran":
                if identifier == 12:
                    raise OSError("hot unplug")
                return uints(_fourcc("ccwd"))
            raise AssertionError((identifier, selector, scope))

        with patch("kon.audio_devices.sys.platform", "darwin"), patch("kon.audio_devices._CoreAudioProperties") as factory:
            factory.return_value.read.side_effect = read
            factory.return_value.text.side_effect = lambda identifier, selector: {
                (10, "lnam"): "Lucas 的麥克風", (10, "uid "): "phone-uid",
            }[(identifier, selector)]
            self.assertEqual(_coreaudio_inputs(), (_CoreAudioInput("Lucas 的麥克風", "phone-uid", "ccwd"),))
            factory.return_value.read.assert_any_call(10, "stm#", "inpt")
            self.assertEqual(set(call[0][1] for call in factory.return_value.read.call_args_list), {"dev#", "stm#", "tran"})

    def test_non_mac_platform_never_loads_native_frameworks(self):
        with patch("kon.audio_devices.sys.platform", "linux"), patch("kon.audio_devices._CoreAudioProperties") as factory:
            self.assertIsNone(_coreaudio_inputs())
            factory.assert_not_called()
