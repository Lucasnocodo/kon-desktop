# KON · 靈狐桌面召喚

繁體中文 · [English](README.en.md)

Apple Silicon macOS 上的 Python 原型：用手圈瞄準檔案，維持手勢完成鎖定，再發出短音或咒語，召喚狐狸吞下目標。支援虛擬檔案演練、桌面一般檔案移至垃圾桶與復原、個人手勢／瞄準校正，以及減少動態效果。

提供內建繁體中文／English 切換的獨立 macOS App，以及可重建的原始碼。只使用一台鏡頭，支援眼睛穿過手圈與手圈移動兩種瞄準方式。個人觸發音與真實 Finder 的完整端到端操作仍需在自己的設備確認；自動化測試使用合成追蹤、離線音訊與模擬 Finder。

## 下載 macOS App（不需 Python）

從 [GitHub Releases](https://github.com/Lucasnocodo/kon-desktop/releases/latest) 下載 `KON-0.2.0-macOS-arm64.zip`，解壓縮後將 `KON.app` 拖到「應用程式」。支援 Apple Silicon；系統最低版本以發行包內的驗證報告為準。App 內含 Python、模型及 CPU Whisper，不需另裝 Python、Xcode 或下載模型。

此版採 ad-hoc 簽章，未使用 Developer ID 簽署，亦未經 Apple 公證。第一次開啟若被 macOS 阻擋，確認下載來源與 `SHA256SUMS` 後，依「系統設定 → 隱私權與安全性」顯示的「仍要打開」流程開啟。右上角 Language 可立即切換 English／繁體中文並保存選擇。英文介面與辨識咒語使用的語言可分別設定。

## 從原始碼安裝與啟動

需要 **Apple Silicon Mac（macOS 14+）、Python 3.12+、Git 與 Xcode Command Line Tools**。macOS 14 是固定依賴的最低需求，並不表示已完成該系統版本的實機操作驗證。安裝前確認 `python3 --version` 至少為 3.12；未安裝 Apple 編譯工具時先執行 `xcode-select --install`。首次安裝需連線下載 Python 套件、模型與固定版本的 whisper.cpp。

```sh
git clone https://github.com/Lucasnocodo/kon-desktop.git
cd kon-desktop
sh scripts/setup.sh
.venv/bin/kon
# 同一入口：.venv/bin/python -m kon.app
```

也可雙擊 [啟動 KON.command](啟動%20KON.command)。開啟控制台不會啟動鏡頭或麥克風；按「預覽狐狸吞噬動畫」只播放虛擬目標與召喚音效，不操作真實檔案。首次明確啟動感測器時，依 macOS 提示授予啟動程式的鏡頭與麥克風權限；首次讀取桌面時再授予 Finder 自動化權限。

`setup.sh` 建立獨立 `.venv`，依 `requirements.lock.txt` 安裝固定版本，以 editable 方式安裝本專案，下載三個模型並編譯 CPU Whisper CLI。Tactyle Touch 由固定 commit 的 Git dependency 安裝；whisper.cpp commit 見 [UPSTREAMS.json](UPSTREAMS.json)。模型不隨原始碼發布；下載與既有檔案每次都會核對 [models.expected.json](models.expected.json) 中提交的大小與 SHA-256，不符便停止。網路中斷可重跑 setup；若既有模型損壞，先移除錯誤檔再重試。既有 whisper.cpp 版本或追蹤內容有變更時 setup 會停止，不會覆寫它。

原始碼執行時由 checkout 載入資源，請保留完整目錄；若只想使用，請下載上方獨立 App。單獨的 Python wheel 不包含完整資源。

## 第一次操作

1. 在第 01 步選瞄準方式與鏡頭。「眼睛穿過手圈瞄準」需讓慣用眼、整隻手入鏡；「手圈移動控制準星」只需手部，依鏡像畫面映射位置。暫停感測器後才能切換。
2. 明確按「啟動鏡頭與麥克風」。眼睛模式先做九點瞄準校正；手圈移動模式可直接用預設範圍，或在第 02 步記錄舒適的左上、右下兩角。
3. 在第 03 步校正靈狐手圈：依中間、上、下、左、右五個角度各記錄兩次，兩次間先放鬆手。每次需至少 12 個新影格與 1 秒穩定資料；完成後放鬆、重新比出手圈並通過試做，最後按「儲存並使用」。校正以中性畫面指引進行。
4. 第 04 步預設「大聲短音即可」，不用辨識成特定文字；其他大聲音也可能觸發。選「辨識咒語文字」時，可使用 `kon` 或自訂口令，並透過「校正我的咒語發音」錄三次、勾選寫法及通過試唸。文字辨識在本機使用 CPU Whisper，校正發音不會移動檔案。
5. 第 05 步先選「演練 · 虛擬檔案」。預設手形是食指、小指伸出，中指、無名指壓住拇指；已儲存個人模板時使用你的手形。將準星保持在目標格，鎖定後再發出一個新短音／口令。成功後可保持手勢瞄準下一個不同檔案。
6. 熟悉後選「桌面 · 移到垃圾桶」。只處理主螢幕桌面第一層的可見一般檔案，忽略資料夾、隱藏檔與符號連結；同格多檔或圖示重疊會停止選取。吞下目標後才移至系統垃圾桶，可用「復原上一個檔案」還原，同名檔案不覆寫。

選單列 KON 圖示可返回控制台或暫停鏡頭／麥克風；動畫吞下目標前取消便不移檔。鎖定計算、手勢取樣與檔案重驗的完整規則見 [RULES.md](RULES.md)。

## 本機資料

App 的設定與個人數值存於 `~/Library/Application Support/KON Desktop/`；原始碼版存於專案 `.kon/`，不納入 Git。影像、手勢與音訊在本機處理；不保存鏡頭畫面、影片或麥克風錄音。Whisper 只在咒語文字模式處理短片段，暫存 WAV 完成後刪除；一般追蹤期間麥克風持續運作，只有有效手勢及已鎖定目標才啟用召喚觸發。預設自動麥克風選擇會排除可辨識的 iPhone／iPad 與虛擬音訊裝置；也可明確選擇自己的實體輸入。

## 開發與驗證

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m kon.speech --self-test
# 不開啟感測器的控制台截圖
QT_QPA_PLATFORM=offscreen .venv/bin/python -m kon.app --screenshot output/kon-ui.png
# 實際 Qt 畫面逐幀預覽；不操作桌面檔案
.venv/bin/python scripts/preview-fox-animation.py
.venv/bin/python scripts/preview-fox-animation.py --smoke-only
# 從保留的生成來源重建透明素材
.venv/bin/python scripts/prepare-fox-sprites.py
.venv/bin/python scripts/prepare-smoke-sprites.py
```

預覽輸出在忽略的 `output/`。`kon.speech --self-test` 使用 whisper.cpp 的公開音訊範例，不開麥克風；需先完成 setup。macOS CI 執行測試及公開內容檢查。測試通過不代表使用者發音辨識率或 Finder「堆疊」／Stage Manager 已通過實機驗證。

依賴範圍及開發工具定義於 `pyproject.toml`；若要自行更新套件，可在獨立環境執行 `python -m pip install -r requirements.txt`。正式重現本版請使用 setup 的固定版本。重建 Whisper 可執行 `sh scripts/build-whisper.sh`，`KON_BUILD_JOBS` 可設定正整數的編譯平行數。

## 重建獨立 App

```sh
.venv/bin/python -m pip install -r requirements-build.txt
.venv/bin/python scripts/build-macos.py
.venv/bin/python scripts/verify-macos.py
.venv/bin/python scripts/package-macos.py
```

打包只選取執行所需的狐狸影像、非語音音效、三個模型、離線辨識器、翻譯與授權聲明；個人資料及開發輸出不會納入。封裝檢查不開鏡頭或麥克風、不操作真實桌面檔案。

## 授權與素材

KON 自有程式碼、腳本與文件使用 [MIT License](LICENSE)，署名 `2026 KON contributors`。第三方元件保留各自授權，詳見 [NOTICE](NOTICE)、[UPSTREAMS.json](UPSTREAMS.json) 與 [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/)。模型另行下載並適用上游條款，來源見 [MediaPipe Hand](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)、[MediaPipe Face](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) 及 [Whisper 模型](https://huggingface.co/ggerganov/whisper.cpp)。

狐狸、煙霧與召喚音效**不在 MIT 授權範圍**；本次沒有替素材或參考角色建立開放素材授權。生成方式、來源與權利範圍見 [assets/NOTICE.md](assets/NOTICE.md) 與 [assets/provenance.json](assets/provenance.json)。
