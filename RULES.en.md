# KON operating rules

[繁體中文](RULES.md) · English

These rules describe the single-camera app's default behavior. For installation and first use, see [README.en.md](README.en.md).

## Aiming, locking, and summoning

- The primary screen is divided into **30 columns × 12 rows** for selection. An icon belongs to the cell containing its center; the entire cell can select it. KON does not rearrange Finder icons or show the complete grid.
- The crosshair moves smoothly with tracking before locking. Rendering interpolates measured positions at about 60 Hz, without inventing tracking frames or contributing dwell time. Locking fixes the crosshair at the saved file center.
- Each cell must contain at most one selectable file. Multiple files in a cell or overlapping icons are ambiguous; move them apart and read the desktop again.
- Dwell scores use the latest **3 seconds** of fresh tracking timestamps. Leaving and returning still accumulates within that window. The unique leading file locks once it reaches **1.5 weighted seconds**; ties or a leading empty cell keep waiting.
- Stable aim requires raw hand XY speed no greater than **0.04 frame fractions per second** and the measured crosshair staying in one cell for **0.3 seconds**. Subsequent dwell earns **2×** weight; the first 0.3 seconds are not retroactively doubled. Typical stable aiming reaches the threshold in about **0.9 real seconds**. Empty cells always earn normal weight.
- Before locking, an **8 px boundary margin** tolerates small movements near a cell edge. It does not allow ambiguous, invalid, or off-screen positions.
- After locking, aim may move within the locked cell and **two surrounding cells**, including diagonals. Moving farther unlocks and clears the old dwell score. A confirmed open hand or hand leaving the frame for about **0.25 seconds** also unlocks.
- Invalid tracking immediately pauses scoring, voice triggers, and actions. An interruption of up to **0.5 seconds** preserves recent dwell, but neither the gap nor the first recovered frame earns time. Longer interruptions, icon changes, screen size changes, or cancellation clear unfinished dwell.
- A near-threshold hand-shape mismatch may reuse an actual recent match for up to **200 ms**, with the same hand, tracking source, and template. Failed frames do not extend that grace. Clear mismatches, invalid landmarks, a different hand, or interrupted tracking cancel it. Voice triggers still require the current fresh frame to pass; the **0.25-second** release timer starts with the actual mismatch.
- Tracking or depth loss preserves an existing lock but pauses actions and audio triggers. After recovery, only a new sound can trigger it. Old sounds or delayed recognition cannot transfer to another file.
- Each lock can trigger once. After confirmed success and the completed animation, you may keep the gesture and lock a different file. Files already handled during the continuous gesture are not selected again. Cancellation, failure, stale callbacks, aim movement, and tracking interruption do not reset an action that has already fired.
- After each successful real-file operation, Finder positions are read again before selection continues. With no remaining files, KON returns to the control panel.

## Lock settings

Step 05 exposes seven saved settings. Changing one cancels the current lock, clears dwell and pending sounds, and keeps the sensors running.

| Setting | Default | Meaning |
| --- | --- | --- |
| Dwell window | 3 seconds | Count only recent dwell within this interval. |
| Lock threshold | 1.5 weighted seconds | The unique leading file must reach this score. |
| Time to stable aim | 0.3 seconds | Required steady time in one cell before increasing the weight. |
| Stable-aim weight | 2× | Multiplier for subsequent stable dwell. |
| Hand motion threshold | 0.04 frame fractions / second | Faster movement uses normal weight and resets the stable period. |
| Unlock distance | 2 surrounding cells | Moving beyond this range unlocks the file. |
| Tracking interruption grace | 0.5 seconds | Preserve scores during a shorter interruption, without scoring the gap. |

The lock threshold is constrained by the dwell window and stable-aim weight so that it remains reachable. Settings are saved under `selection_settings` in `preferences.json`.

## Single-camera modes and calibration

- **Aim through the hand ring** uses the dominant eye and hand ring to estimate an aiming ray, then applies a personal nine-point mapping. This is not eye-gaze detection. Keep the camera and posture consistent; smoothing cannot remove a fixed aiming bias.
- **Move the hand ring to aim** uses hand tracking only. It maps the ring center in the mirrored camera image to the screen, with an optional two-corner comfortable movement range. It does not need eye tracking or a nine-point eye calibration.
- Pause the sensors before switching aiming modes or camera. Compatible saved calibration is reused automatically. Hand movement range, eye aiming, gesture shape, and phrase pronunciation are saved separately.
- An explicit Record action accepts only newer source frames. Old, repeated, out-of-order, and stale frames do not count. Nine-point position calibration does not require the summoning gesture to match, but it does require valid eye and hand ring positions.
- Each aiming point needs at least **6 valid frames spanning 1 stable second**, with up to **6 seconds** to collect them. Depth, outlier, and drift checks remain active. A failed sample does not overwrite the saved model. You may retest one point while keeping the other eight.
- Personal hand-shape calibration records **center, up, down, left, and right** angles using the same hand and tracking source. Keep the finger shape fixed and change only palm orientation. Each angle needs two independent recordings, each with at least **12 fresh frames spanning 1 stable second**; relax between recordings. The second is compared only with that angle's first recording.
- After five angles, relax and repeat the gesture, pass the stable trial, then explicitly choose **Save and use**. Canceled, failed, or unsaved attempts keep the previous gesture and aiming data.
- Older one-angle or incomplete sets may be reused. **Add another angle** adds an independently confirmed angle, up to five. Cancellation keeps the existing set.
- Each angle keeps a separate numeric template; matching any angle is sufficient. Recognition and second-recording confirmation use **1.35×** the saved tolerance. Adding an angle does not average templates, widen their ranges, or teach an unstable gesture.
- Saved data contains numeric parameters, not camera images. A saved mapping does not prove real-world accuracy. The optional fresh-frame accuracy check uses points separate from the nine calibration points; its result does not erase the applied mapping.

## Microphones, language, and triggers

- Opening the control panel and listing devices do not record audio. Camera tracking and pronunciation calibration share the selected microphone. Pause both and wait for the microphone to stop before changing it.
- Automatic selection avoids identifiable iPhone, iPad, Continuity, aggregate, and virtual inputs. Explicit selection saves device identity rather than a temporary index. A missing or ambiguous saved device produces an error instead of silently switching inputs. KON does not modify the macOS default microphone.
- **Any loud short sound** detects volume, not a particular syllable. Claps or other loud sounds can trigger it. Each activation first requires at least **160 ms of quiet**, then a fresh loud segment of about **60 ms**; a sustained sound does not trigger again in the next round.
- **Recognize trigger phrase** uses local CPU Whisper and accepts complete short phrases. Before personal pronunciation calibration, the default accepts standalone `kon` or `con`. After calibration, only the configured phrase and selected complete transcriptions are accepted. Longer sentences require an explicit full-phrase configuration; arbitrary fuzzy matching is not used.
- Pronunciation calibration collects one segment only after an explicit recording or test request. It stops after completion, cancellation, or timeout and deletes temporary WAV files. Three transcriptions, selected spellings, and the recognition language are saved; recordings are not retained. This does not retrain the speech model.
- Changing a phrase, its accepted spellings, or its recognition language requires another test. Changing the interface language between English and Traditional Chinese does not change recognition language or user-entered text.
- Calibration and pronunciation tests never trigger a real file operation. Animation previews use a virtual file and a local non-speech summoning sound. Canceling or closing stops the sound.

## Real files and saved data

- Desktop mode accepts visible regular files directly on the primary desktop. It excludes folders, hidden files, symbolic links, off-screen files, overlapping icons, and ambiguous cells. Finder permission errors are reported rather than treated as an empty desktop.
- Before committing an action, KON rechecks the file's identity, icon position, non-overlap, and uniqueness in the selected cell. Changed identity or position cancels the action. The file moves to the system Trash only after the animation confirms the swallow.
- **Restore last file** restores the most recent successful operation. It checks the trashed file's identity and never overwrites an existing original path. If the item or destination has changed, restore it manually from Trash.
- The standalone app keeps settings under `~/Library/Application Support/KON Desktop/`; a source checkout uses `.kon/`. Aiming mappings, gesture templates, hand movement ranges, microphone identity, trigger settings, and interface language are stored as numeric or text settings, without camera images or microphone recordings.
- First use should start with virtual-file practice. Personal speech recognition, real desktop operation, Finder Stacks, Stage Manager, and actual aiming accuracy require validation on your own setup.
