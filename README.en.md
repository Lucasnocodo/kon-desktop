# KON · Fox summoning for your desktop

[繁體中文](README.md) · English

Aim at a file with a hand gesture, hold to lock it, and make a short sound or say a phrase to summon the fox. KON runs locally on an **Apple Silicon Mac with macOS 14 or later**, using one camera. It includes virtual-file practice, personal gesture and aiming calibration, a fox animation, moving ordinary desktop files to Trash, and restoring the last file.

## Download and open the app

1. Download the Apple Silicon macOS archive from [GitHub Releases](https://github.com/Lucasnocodo/kon-desktop/releases/latest).
2. Unzip it and drag **KON.app** into **Applications**, then open it. Python, model weights, and the local speech recognizer are included; no Python installation or model download is needed.
3. The release uses an ad-hoc signature and is not Apple-notarized. If macOS blocks the downloaded app, open **System Settings → Privacy & Security** and use **Open Anyway** for KON after attempting to open it. Follow the included installation instructions.
4. Choose **English** or **繁體中文** in the control panel's **Language** selector. The interface updates immediately and remembers your choice. Changing the interface language does not change the phrase recognition language or your calibration.

Opening the control panel does not start the camera or microphone. **Preview fox animation** uses a virtual target and plays the local summoning sound; it does not operate on real files. When you explicitly start the sensors, allow KON to access the camera and microphone. The first desktop scan also requests permission to control Finder.

macOS 14 is the minimum for the pinned runtime dependencies, not a claim that every hardware setup has been tested on that version. Actual aiming accuracy, your trigger pronunciation, and the full real-Finder workflow still need checking on your own equipment. Automated checks use synthetic tracking data, offline audio, and simulated Finder operations. Finder Stacks and Stage Manager have not been verified.

## First use

1. In **01 Connect devices**, choose your camera and aiming mode. **Aim through the hand ring** needs your dominant eye and entire hand in view. **Move the hand ring to aim** uses hand movement in the mirrored camera view and does not need your eyes visible. Pause the sensors before switching modes.
2. Press **Start camera and microphone**. Eye-through-ring aiming needs a nine-point screen calibration. In hand movement mode, use the default range or record comfortable top-left and bottom-right positions in step 02.
3. In step 03, choose **Calibrate fox hand ring**. Record the same hand shape at five angles: center, up, down, left, and right. Make two independent recordings per angle, relaxing your hand between them. Each recording needs at least 12 fresh frames spanning one stable second. After all five angles, relax, repeat the gesture, pass the trial, and choose **Save and use**. Instructions appear on screen.
4. Step 04 defaults to **Any loud short sound**. It detects volume and does not require a particular word; other loud sounds can also trigger it. For a specific phrase, choose **Recognize trigger phrase**. Use `kon` or a short phrase of your own, then optionally record three pronunciations, select accepted transcriptions, and pass the test. Recognition uses local CPU Whisper. Pronunciation testing does not move files.
5. In step 05, choose **Practice · Virtual files**. The default gesture extends the index and little fingers while the middle and ring fingers press against the thumb; a saved personal template replaces that shape. Aim at a file's cell, hold until locked, and make a new sound or say the phrase. After a successful animation, keep the gesture and aim at a different file to continue.
6. When ready, choose **Desktop · Move to Trash**. Only visible regular files directly on the primary desktop are eligible. Folders, hidden files, symbolic links, overlapping icons, and ambiguous cells are excluded. The file moves to the system Trash only after the fox swallows it. **Restore last file** restores the last successful operation without overwriting a file at the original path.

The menu bar icon lets you return to the control panel or pause the camera and microphone. Canceling before the swallow prevents the file operation. See [RULES.en.md](RULES.en.md) for dwell scoring, gesture sampling, cancellation, and file revalidation details.

## Local processing and saved data

Camera, gesture, and audio processing stay on the Mac. KON does not save camera images, videos, or microphone recordings. Phrase mode recognizes short audio segments with Whisper and deletes temporary WAV files afterward. While sensors are running, the microphone remains open; summoning triggers are enabled only with valid tracking and a locked target.

The automatic microphone selection excludes identifiable iPhone, iPad, aggregate, and virtual audio inputs. You can choose a specific physical microphone. KON saves the device identity and does not change the macOS default input device.

The app stores settings and numeric calibration data in `~/Library/Application Support/KON Desktop/`. A source checkout uses its own `.kon/` directory, which is ignored by Git. Compatible settings are reused; canceling or failing calibration keeps the previous saved mapping.

## Run or build from source

Source development needs **Python 3.12+, Git, and Xcode Command Line Tools** on an Apple Silicon Mac running macOS 14+. Check `python3 --version`; install Apple's compiler tools with `xcode-select --install` if needed.

```sh
git clone https://github.com/Lucasnocodo/kon-desktop.git
cd kon-desktop
sh scripts/setup.sh
.venv/bin/kon
# Equivalent entry point:
.venv/bin/python -m kon.app
# Choose an interface language explicitly:
.venv/bin/kon --language en
.venv/bin/kon --language zh-TW
```

Setup creates `.venv`, installs [locked dependencies](requirements.lock.txt), installs this project in editable mode, downloads three public models, and builds the pinned CPU whisper.cpp recognizer. Tactyle Touch is installed from a fixed Git commit; see [UPSTREAMS.json](UPSTREAMS.json). Model size and SHA-256 are checked against the committed [models.expected.json](models.expected.json), including files already present. An interrupted download can be retried; remove a corrupt model before retrying. A different or modified whisper.cpp checkout is rejected rather than overwritten.

Keep the complete source checkout: its images, models, and recognizer are loaded alongside the source. A Python-only wheel is not the standalone distribution. Dependency ranges and the development extra are in `pyproject.toml`; `requirements.txt` installs the development extra, while setup uses the fixed lock file.

```sh
# Run automated checks without real camera or microphone capture
.venv/bin/python -m pytest -q
.venv/bin/python -m kon.speech --self-test
# Render the control panel without sensors
QT_QPA_PLATFORM=offscreen .venv/bin/python -m kon.app --language en --screenshot output/kon-ui-en.png
# Render the actual Qt animation with a virtual file
.venv/bin/python scripts/preview-fox-animation.py
.venv/bin/python scripts/preview-fox-animation.py --smoke-only
# Rebuild the retained generated sprites
.venv/bin/python scripts/prepare-fox-sprites.py
.venv/bin/python scripts/prepare-smoke-sprites.py
# Build the standalone application
.venv/bin/python -m pip install -r requirements-build.txt
.venv/bin/python scripts/build-macos.py
.venv/bin/python scripts/verify-macos.py
.venv/bin/python scripts/package-macos.py
```

Generated previews go into ignored `output/`. The speech self-test uses the pinned whisper.cpp public audio sample and does not open the microphone. The app build is produced at `dist/macos-arm64/KON.app`; the build includes its runtime dependencies, verified models, and third-party notices. The macOS CI checks the public source boundary, tests, offline recognition, and sensor-free rendering. These checks do not establish real-world aiming accuracy or personal speech recognition rates.

## Code, dependencies, and media rights

Original KON code, scripts, and documentation use the [MIT License](LICENSE), copyright `2026 KON contributors`. Third-party components retain their own terms; see [NOTICE](NOTICE), [UPSTREAMS.json](UPSTREAMS.json), and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/). The app also includes component notices under `KON.app/Contents/Resources/ThirdPartyNotices/`. Model sources and upstream terms are linked from [MediaPipe Hand](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), [MediaPipe Face](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker), and the [Whisper model repository](https://huggingface.co/ggerganov/whisper.cpp).

**The fox images, smoke images, and summoning sound are excluded from the code's MIT license.** This release does not establish an open media license or assert rights in the reference character. Known origins and rights boundaries are recorded in [assets/NOTICE.md](assets/NOTICE.md) and [assets/provenance.json](assets/provenance.json). You can replace these files with media you have permission to use.
