# KON for macOS — Getting started

Requires an Apple Silicon Mac and macOS 14 or later. Python, models and the
CPU speech recognizer are included; no Python, Xcode or installation script is
needed. The oldest supported macOS version is a binary compatibility target;
read `Release.json` for the actual tested macOS version and validation scope.

1. Download the ZIP and `SHA256SUMS` from the same official GitHub release:
   https://github.com/Lucasnocodo/kon-desktop/releases
2. In Terminal, change to the download folder and run `shasum -a 256 -c SHA256SUMS`.
   The ZIP must report `OK`. A checksum checks the downloaded bytes, not the
   publisher's identity; use the official release page.
3. Extract the ZIP and drag `KON.app` to Applications. Open it there.
4. This release is ad-hoc signed and has **not** been notarized by Apple. If macOS
   says it cannot verify the developer, first verify the official source and
   checksum. After attempting to open KON, follow macOS System Settings → Privacy
   & Security → Open Anyway for this app. Apple's official instructions:
   https://support.apple.com/102445
   If macOS identifies malware or reports damage, stop and report the exact error.
5. The language selector offers English and 繁體中文; the initial choice follows
   your saved setting or system language. Opening the control room does not start
   cameras or microphone. Try the fox animation preview first.
6. Choose hand-circle cursor or eye-through-hand aiming, then explicitly start
   sensors. Grant Camera and Microphone permission to KON when prompted. Complete
   the aim/gesture calibration shown on screen. A new short sound triggers the
   fox only after a target has been locked; phrase recognition is an optional
   local CPU mode.
7. Desktop mode can move selected ordinary files from the primary desktop to
   Trash. Finder automation permission is requested when needed. Use virtual
   practice before real files; **Restore last file** restores the most recent operation without
   overwriting an existing file. This app uses one camera.

Settings and personal numeric calibration data are stored outside the app in
`~/Library/Application Support/KON Desktop/`. Camera frames and microphone
recordings are not retained. Quit KON before replacing the app with an update.
The archive includes validation reports and third-party licenses. Automated
checks use synthetic/blank images and a public audio fixture, without sensors
or real Finder file operations; they do not establish personal aiming accuracy.

KON code is MIT licensed. Bundled components retain their licenses; fox artwork
and sound are excluded from the code license. See `ThirdPartyNotices/`.
