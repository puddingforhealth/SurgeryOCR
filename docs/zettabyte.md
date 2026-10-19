# Zettabyte 傳輸與雲端分析

本分支提供可獨立安裝的 `zb` CLI。影片、資料集、模型權重、設定檔和分析結果都能傳輸；資料不必進 Git。
Python 3.10+、AzCopy v10 為必要條件。純傳輸工具不依賴 PyTorch、OpenCV 或 SurgeryOCR。

## 安裝

在本機及 Zettabyte IDE 各自拉取分支：

```bash
git fetch origin
git switch --track origin/zettabyte
# 若本機已有 zettabyte 分支，改用 git switch zettabyte，再 git pull --ff-only
```

從 [Microsoft 官方頁面](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy) 安裝 AzCopy v10，並確認 `azcopy --version` 可執行。macOS 也可使用 `brew install azcopy`。

在專案根目錄，以目前 Python 環境安裝 CLI（建議使用虛擬環境）：

```bash
python3 -m pip install --no-deps -e .
zb --help
```

Windows 若命令為 `python`，請將文件中的 `python3` 換成 `python`。
不安裝套件也可在專案根目錄執行 `python3 -m zettabyte_transfer`，功能與 `zb` 相同。
這個套件安裝步驟只安裝傳輸工具；雲端分析仍須原專案的 Python/GPU 相依環境與樣板資料。

## 憑證

本機、IDE 都要各自設定。從**原始信件**複製完整的容器 SAS URL，包含 `https://...?...&sig=...`；不要使用聊天中截斷或混雜的內容。

```bash
zb configure
zb doctor
zb list
```

`configure` 隱藏輸入，原子寫入 `~/.config/zettabyte-transfer/config.json`；Unix 檔案權限為 0600。
Windows 的檔案存取權依使用者目錄的 ACL 管理。SAS 不儲存在 repo，CLI 輸出會遮蔽 SAS URL／簽章。
AzCopy 子程序仍須透過命令列參數取得 SAS，主機上的程序檢視工具可能看得到；請在可信任的本機／IDE 環境執行。

自動化可從 secret manager 注入環境變數 `ZETTABYTE_SAS_URL`，優先於設定檔。不要把值寫進腳本、命令歷史或提交到 Git。
多個租戶／專案可使用獨立設定檔：

```bash
zb --config ~/.config/zettabyte-transfer/other.json configure
zb --config ~/.config/zettabyte-transfer/other.json list
```

`doctor` 只檢查 URL 結構與 AzCopy 是否存在；`list` 才會連線驗證權限。過期／被撤銷的 SAS 請向租戶管理員取得新值後重新 configure。

## 傳輸規則

依 Zettabyte onboarding 信件：

```text
本機 --upload--> Azure 容器根目錄 --Zettabyte 自動同步--> 租戶共享 PVC
IDE  --publish--> Azure outbound/ --download--> 本機
```

- `upload` 可指定根目錄下的子路徑，會拒絕 `outbound/`，避免資料無法進 PVC。
- `publish` 自動加上 `outbound/`；參數不要再加此前綴。從本機執行也能上傳到 outbound，但不會進 PVC。
- 同步到 `zb-sharedata-pvc` 的實際掛載路徑、延遲及搬移行為由平台決定。Azure 上傳成功不等於 PVC 已可讀；請等狀態信並在 IDE 核對檔案大小／雜湊，再開始分析。
- 同一租戶共用容器，使用專案名稱和唯一批次 ID 區隔。不要在檔案尚未同步完成時開始分析。
- CLI 不會自動啟動 GPU 工作；分析指令須在有 PVC 與 GPU 的 IDE／工作環境執行。

## 上傳影片、資料與模型

```bash
# 單一檔案：遠端參數為完整檔名
zb upload ./data/case001.mp4 SurgeryOCR/jobs/job001/case001.mp4
zb upload ./models/model.pth SurgeryOCR/models/v1/model.pth

# 資料夾：把「內容」放到指定目錄，不再多套一層來源資料夾名稱
zb upload ./dataset SurgeryOCR/datasets/v1
zb upload ./models SurgeryOCR/models/v1

# 預覽，不發出傳輸請求
zb --dry-run upload ./models SurgeryOCR/models/v2
zb list SurgeryOCR
```

路徑含空白請加引號。支援 Unicode 檔名。傳輸不跟隨符號連結；來源頂層若為符號連結會拒絕。
預設不覆寫同名檔案；需要更新時明確加 `--overwrite`。建議以版本／job ID 建立新目錄，避免混用不同版本。
上傳會計算 Content-MD5；下載如有 Content-MD5 則檢查一致性，缺少雜湊的第三方檔案仍可下載。
覆寫關閉時，同名檔案會被 AzCopy 跳過，成功退出不表示它已被更新。

## 在雲端跑 SurgeryOCR 並發布

先把相同版本的專案部署到 IDE，確認設定、樣板快取及模型齊全。範例路徑 `/mnt/share` **必須換成實際 PVC 掛載路徑**。

```bash
# 在專案根目錄，使用能跑原專案的 Python 環境
python3 -c "import torch; print(torch.cuda.is_available())"

python3 zettabyte_analyze.py \
  --video /mnt/share/SurgeryOCR/jobs/job001 \
  --job-id job001 \
  --output-root /mnt/share/SurgeryOCR/results \
  --mode ram \
  --publish
```

`--video` 可為單一影片或包含 MP4 的目錄（不遞迴）。影片 stem 必須唯一。
`ram` 只保存數值／階段等結果；`frame` 加上完整影格，`detail` 再加 ROI 小圖，檔案量較大。
`--publish` 會在全部影片成功、且存在非零影格階段結果與 OCR JSONL 後，發布到 `outbound/results/job001/`。

輸出包含 `job.json`、`analysis.log` 與各影片的結果目錄。`job.json` 記錄 running／failed／complete，代表分析狀態，**不代表發布狀態**；發布失敗以 AzCopy 非零退出碼表示。
同一 job ID 的本地輸出目錄已存在時會拒絕執行，避免發布舊資料。重跑分析請換新 job ID。
完整 log 寫到輸出目錄的 `analysis.log`，長工作可用另一個終端追蹤。不要同時跑多個分析程序，原 pipeline 會更新共用的機型設定。

不帶 `--publish` 可只分析。完成後也能手動發布，或發布訓練模型：

```bash
zb publish /mnt/share/SurgeryOCR/results/job001 results/job001
zb publish /mnt/share/training/checkpoints models/experiment001
```

## 本機下載與 GUI 檢查

```bash
# 目錄內容直接下載到指定目錄
zb download results/job001 ./data/cloud-results/job001
zb download models/experiment001 ./models/experiment001

# 單一檔案：指定完整的本機檔名
zb download results/job001/job.json ./job001.json

# 進階：讀 Azure 容器根目錄（不加 outbound/）
zb download SurgeryOCR/models/v1 ./models/v1 --container
```

保留本機原影片後，在原專案 GUI 開啟影片即可。GUI 會搜尋 `data/` 下的影片分析目錄；若分析結果存到其他位置，可透過 `ANALYSIS_DATA_ROOT` 指向下載的 job 目錄。
若不同 job 存在相同影片名稱，建議明確設定 `ANALYSIS_DATA_ROOT`，避免 GUI 選到舊結果。
下載整個 job 時，`job.json` 可能先於其他檔案抵達；請等下載指令成功結束後再檢查 GUI。

## 失敗、續傳與診斷

CLI 會保留 AzCopy 的退出碼；工具設定／參數錯誤為 2，分析接線失敗為 1，中斷為 130。
保留 AzCopy 印出的 Job ID 與同一機器的 job plan／log（預設在 `~/.azcopy`）：

```bash
zb status 12345678-1234-1234-1234-123456789abc
zb resume 12345678-1234-1234-1234-123456789abc
```

續傳使用目前設定的 SAS，適用本工具同一租戶的上／下載工作；更換機器或刪除 job plan 後無法續傳。
續傳無法追回已被平台搬走的輸入檔；必要時重新上傳到新 job 路徑。
可用 `ZETTABYTE_AZCOPY` 或全域 `--azcopy /path/to/azcopy` 指定執行檔。
全域選項放在子指令之前；`--overwrite`、`--container` 放在其子指令之後。

## 未來抽成小 repo

通用部分只有 `zettabyte_transfer/` 與根目錄的 `pyproject.toml`，可直接搬到新 repo。
`pip install .` 只打包通用部分並提供 `zb`，不會打包手術資料、模型、SurgeryOCR 或其依賴。
`zettabyte_analyze.py` 是本專案接線，留在 SurgeryOCR，透過通用 CLI API 發布結果。

## 驗證範圍

```bash
python3 -m unittest discover -s tests -p 'check_*.py' -v
```

離線測試使用模擬 AzCopy 子程序驗證資料／模型逐位元往返、目錄映射、覆寫、錯誤碼、憑證遮蔽、設定與分析發布閘門。
另外已使用官方 AzCopy 10.32.8 驗證所用參數相容性，並驗證獨立 wheel 安裝與 `zb` 入口。
這些測試不取代租戶的 Azure 授權、PVC 自動同步或實際 GPU 推論驗收。
第一次使用請先以上傳小型非敏感檔案、在 IDE 發布、下載後比對雜湊的方式確認通路，再處理正式資料。

參數依據：[AzCopy copy](https://learn.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-copy)、[AzCopy jobs resume](https://learn.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-jobs-resume)。
