# 素材來源與權利範圍

本目錄的 PNG 與 WAV 是 KON 示範使用的素材，**不適用根目錄的 MIT 程式碼授權**。本次發布沒有替影像、音效或其中可辨識的角色外觀另行建立開放素材授權，也不主張原參考圖或角色的權利。重用或再散布素材時，須自行確認相關權利；程式碼可依 MIT 使用，亦可替換成自己有權使用的素材。

| 素材 | 已知來源與製作方式 |
| --- | --- |
| `fox-heads.png` | 內建 imagegen 依使用者提供的白毛、多眼、橘紅螺旋眼狐狸參考生成，兩格透明開閉嘴圖。原參考圖未納入本儲存庫。 |
| `fox-dash/` | 內建 imagegen 一次生成 12 格動作圖，以原狐狸外觀及先前生成圖為參考；來源圖與提示保存在 `source/`，由 `scripts/prepare-fox-sprites.py` 切圖。 |
| `fox-smoke/` | 內建 imagegen 一次生成 12 格透明煙流，以狐狸配色及先前效果為參考；來源圖與提示保存在 `source/`，由 `scripts/prepare-smoke-sprites.py` 等格切圖。 |
| `sfx/yaohu-short.wav` | 原專案本機合成的 1.34 秒非語音音效；技術資料與雜湊在 `sfx/manifest.json`。原合成工程未納入本版。 |

公開的生成提示是製作來源紀錄；提示中提到的早期輸入或外部參考不代表那些圖檔一併公開。機器可讀的來源、檔案範圍與 SHA-256 見 [provenance.json](provenance.json)。
