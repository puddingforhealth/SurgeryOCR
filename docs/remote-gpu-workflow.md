# 從本機派送 SurgeryOCR 到遠端 GPU

`rcs` 是通道，分析程式、Python 環境與 GPU 都在遠端。一般任務拆成：

| 部分 | 準備方式 | 更新頻率 |
|---|---|---|
| 程式、設定、ROI 樣板、git 已追蹤的權重 | 固定 git commit，或 `git archive` 打包 | 程式／模型改版時 |
| Linux Python／CUDA／PyTorch／EasyOCR 環境 | 使用平台提供的 GPU 環境，補齊專案依賴並驗證 CUDA | 建立或重建環境時 |
| git 未追蹤的權重、EasyOCR model cache | 另外用 `rcs push` 傳送到程式使用的位置 | 模型改版或容器遺失資料時 |
| 輸入影片 | 本頁的批次上傳腳本 | 每次任務 |
| 輸出結果與 log | 獨立輸出目錄，完成後下載並核對 | 每次任務 |

不用每次把 code、環境與整套 dataset 重新打包。Windows 的 `release_env` 不能直接複製成
Linux 的 Python 環境。遠端 GPU 型號正常也不代表 PyTorch／EasyOCR 已安裝或 CUDA 可用。
本頁的上傳腳本只處理影片傳輸，**不會部署環境或啟動分析**。從 Drive 直接下載、RAM 分析與結果回收請見[Drive 批次分析](drive-batch.md)。

## 已分析影片的批次上傳

需求：本機 Python 3.10+、已安裝 `rcs >= 0.3.0`，並設定可用的 Web IDE target。
遠端需 Python 3 標準函式庫，用於磁碟容量與 SHA-256 檢查；傳輸本身走 `rcs push`。
不需要啟用 SurgeryOCR 的 GPU 虛擬環境。

以下在本機 SurgeryOCR repo 根目錄執行。這台電腦已把先前準備的 19 支影片清單放在
`build/rcs-video-upload/manifest.json`，可以直接執行 `plan`／`upload`：

```powershell
# 只看順序、容量與目的地，不連遠端
python -X utf8 tools/upload_analyzed_videos.py plan --smallest-first

# 逐支上傳，保留 data/ 下的子目錄；預設至少留 2 GiB 空間
python -X utf8 tools/upload_analyzed_videos.py upload --smallest-first

# 同一命令可重新執行：核對已完成檔案後跳過，接著處理剩下的
python -X utf8 tools/upload_analyzed_videos.py upload --smallest-first

# 只新增上傳一支，或指定另一個有足夠容量的掛載
python -X utf8 tools/upload_analyzed_videos.py upload --smallest-first --limit 1
python -X utf8 tools/upload_analyzed_videos.py upload --target another-ide --remote-dir /data/videos --state build/rcs-video-upload/another-state.json

python tools/upload_analyzed_videos.py -h
```

預設 target 為 `zettabyte`、目的地為 `/root/SurgeryOCR/data`。例如本機
`data/subfolder/video.MP4` 會放到遠端 `/root/SurgeryOCR/data/subfolder/video.MP4`。
程式開始時固定該 target 的 URL，執行途中另開終端切換預設 target 不會改變本批目的地。

每支影片的流程：

1. 重新算本機 SHA-256，確認與輸入清單一致。
2. 查遠端檔案。大小及 SHA-256 都吻合才跳過；不同內容視為衝突並停止，絕不自動覆蓋。
3. 查目的地檔案系統可用空間。需要「影片大小 + 預留空間」；建立父目錄後再確認一次。
4. 執行 `rcs push`，沿用分塊傳輸、進度、暫存檔與 SHA-256 讀回驗證。
5. 在遠端另算一次 SHA-256，確認等於清單，再原子更新本機進度紀錄。

`build/rcs-video-upload/state.json` 是進度紀錄，包含 `verified`、`uploading`、
`blocked_space`、`failed` 或 `interrupted`。清單是輸入快照，狀態檔才是上傳進度。
狀態依清單、遠端 endpoint 和目的地綁定；換任一項請使用新的 `--state`。
重新執行即使看到 `verified` 也會重新核對遠端，若檔案已刪除則會重新上傳。
同一狀態檔有程序鎖，避免兩個批次同時更新。

**整支檔案完成後可接續批次，但單支檔案中途不支援斷點續傳。**
斷線或 Ctrl+C 後，下次會先檢查正式檔案是否已完成；尚未完成的影片從頭傳。
`rcs` 若無法清除 `.part` 會列出位置，請確認沒有傳輸正在進行再處理。腳本不自動刪除
影片、結果或遺留暫存檔，也不會為了塞下更多影片自行降低預留容量。

上傳 N GB 的 `rcs` 讀回驗證還會下載 N GB；批次的額外 hash 在遠端本機執行，不再傳回影片。
容量檢查無法鎖住其他程序的磁碟使用量，若期間磁碟滿或傳輸失敗會停止。
`--reserve-gib` 預設 2，只是最小緩衝，不保證足夠安裝 PyTorch 或產生分析輸出。

退出碼：`0` 全部完成或達到 `--limit`、`1` 檢查／傳輸失敗、`3` 空間不足、`130` 使用者中斷。
`--limit` 限制本次新增上傳數量，已存在且吻合的檔案不計入。

2026-10-20 的這個 Zettabyte 容器另有 `/tmp` 磁碟掛載，`df` 顯示約 13 TiB 可用，
與根目錄及共享 PVC 是不同容量資訊。這不代表平台承諾的個別配額或持久儲存；釋放資源後是否
保存尚未確認。若選擇用它存放本次分析的暫存輸入，可改用：

```powershell
python -X utf8 tools/upload_analyzed_videos.py upload --smallest-first --remote-dir /tmp/SurgeryOCR-data --state build/rcs-video-upload/tmp-state.json
```

換目的地不會自動搬動已傳到舊位置的檔案；它會核對新的目的地，再上傳缺少的檔案。
本機原影片始終保留。容器可見掛載可用 `rcs 'df -h -x tmpfs -x devtmpfs'` 查詢，
個別有配額的掛載也應以 `rcs df -h /實際目的地` 檢查。

### 在其他 checkout 或更新影片清單

```powershell
# 初次產生預設清單：只選有 stage_analysis.json + OCR 結果的影片，計算 SHA-256
python -X utf8 tools/upload_analyzed_videos.py prepare

# 日後新增分析過的影片，建立另一份快照與狀態
python -X utf8 tools/upload_analyzed_videos.py prepare --manifest build/next-upload/manifest.json
python -X utf8 tools/upload_analyzed_videos.py upload --manifest build/next-upload/manifest.json --state build/next-upload/state.json
```

`prepare` 會搜尋 `data/` 的 MP4（含大寫副檔名），分析證據來自影片旁的同名目錄、
`data/<stem>/`，或 `runs/baseline|candidate/<stem>/`。有重複 stem 時，只有影片旁的
分析結果可確定歸屬，不會冒用另一支影片的 golden 結果。它不覆寫既有清單。
清單、狀態與影片名稱留在 gitignored `build/`，不提交 GitHub。

## 程式／模型準備與單支任務

先確認遠端程式與本機要跑的是同一版本。可使用 git clone/fetch 部署指定 commit；
若遠端不方便登入 GitHub，可在本機打包**已提交**的程式：

```powershell
git rev-parse HEAD
git archive --format=tar --output=build/rcs-video-upload/pipeline.tar HEAD
rcs push build/rcs-video-upload/pipeline.tar /root/pipeline.tar
```

`git archive` 不含尚未提交的修改、被忽略的影片、Windows venv 或使用者目錄的 EasyOCR cache。
在遠端將封裝解到新的版本目錄，再設定 Linux 環境；不要直接覆蓋正在跑的 checkout。
大型未追蹤權重與 `~/.EasyOCR/model` 另外部署。先前本機的 `pipeline.zip` 是另一份有固定
版本的封裝，不應只因檔名相同就視為最新程式。

`.venv` 由遠端執行一次 `bash scripts/setup.sh` 建立（可重複執行，`git pull` 後再跑即可更新依賴），
它會裝好 CUDA PyTorch、EasyOCR、OpenCV 與 `requirements.txt` 其餘依賴，並在 GPU 可見但 torch
用不到時直接報錯。根目錄空間不足時用 `VENV_DIR=/大容量路徑/venv bash scripts/setup.sh`，
`.venv` 會成為指向該處的 symlink。視訊經 `rcs` 傳送時可加 `SKIP_GCLOUD=1`。

假設遠端 `/root/SurgeryOCR` 已部署正確版本並跑過 setup，單支任務可直接從本機送出：

```powershell
rcs --cwd /root/SurgeryOCR .venv/bin/python -c 'import torch, easyocr, cv2; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
rcs --cwd /root/SurgeryOCR .venv/bin/python run_pipeline.py --video data/subfolder/video.MP4 --mode ram --output-dir runs/cloud
```

這些是環境完成後的用法，不代表目前遠端環境已建好。使用 `--mode ram` 可避免逐幀落盤；
仍須為模型、結果、暫存檔預留空間。較長任務可由遠端 `tmux` 或平台工作管理機制持續執行，
因為 `rcs` 前景命令沒有斷線後恢復工作狀態的功能。

`run_pipeline.py --video <目錄>` 會遞迴掃描子目錄中的 MP4（不分大小寫），所以
`--video data` 即可整批分析；不同子目錄若有同名影片會在開始前報錯，因為結果會寫進同一個
`<output-dir>/<檔名>/`。任務成功也應核對預期結果檔，
不能僅以 GPU 可見或 shell 已退出判定分析完成。

容器容量放不下全量影片時，完整流程應是：**上傳一支 → 分析 → 下載並驗證結果 → 清除該次
遠端輸入／暫存 → 下一支**。本頁上傳腳本只負責第一段；不會在結果尚未收回時刪除影片。需要自動執行完整循環時，使用 [Drive 批次分析](drive-batch.md) 的 worker 與 collector。

## 排隊、通知與計費

可在本機做可用性監測：低頻率嘗試 `rcs nvidia-smi`，成功後通知或開始已準備好的任務。
這只能證明當下 IDE／終端及 GPU 已可用，不能推算平台排隊位置、分配時間或計費狀態。
若每次配置得到新 job URL，需要更新 target；同一 URL 才可直接沿用。

較省 GPU 等待時間的方式，是先準備本機的任務清單、固定程式版本、模型與依賴封裝，
資源可用後立即部署／派工。能否在未配置 GPU 時先寫入持久磁碟、停止後是否保存資料、
以及如何正式釋放 GPU，要由平台機制確認，不能從 Web IDE 連線能力推論。
關閉 IDE 分頁、退出 shell 或在容器執行 `shutdown` 都不能當作已完成平台停止計費的證據。
此上傳腳本不操作平台的開機、排隊或釋放資源。

## 驗證

離線行為測試（不連遠端，不需 GPU）：

```powershell
python -X utf8 -m unittest discover -s tests -p 'check_*.py' -v
```

涵蓋逐支順序、已完成檔案核對與跳過、空間保留、內容衝突、來源改變、失敗與中斷紀錄、
endpoint 綁定、程序鎖、清單發現規則、CLI 分派與實際 helper 的 hash／容量檢查。
既有主流程另使用 `tools/golden.py check --set full --rtol 0` 驗證四支影片。

2026-10-20 實測紀錄：

- 23 項離線測試全部通過；Python 語法檢查通過。
- 完整 golden 四支影片全部 `IDENTICAL`，`rtol=0`，總分析時間約 954.8 秒。
  Windows 測試父程序設定 `PYTHONUTF8=1`，確保分析子程序也能輸出 Unicode。
- 實際批次從 19 支清單中上傳並核對 4 支，合計 3,033,834,631 bytes；包含完整超過 1 GB 的影片。
- 第五支開始前，目的地剩 2.08 GiB，需要 3.33 GiB（含 2 GiB 預留），因此以退出碼 3 停止，
  尚餘 15 支。沒有自動降低預留量或覆寫遠端檔案。
- 遠端 `/root/SurgeryOCR` 當時為 `2160328`，預設 `python3` 缺 torch／easyocr／cv2／h5py，
  專案沒有 `.venv`。本次只上傳影片，沒有在遠端啟動分析或釋放平台 GPU 資源。

本機完整紀錄位於 `build/rcs-video-upload/upload.log` 與 `state.json`，不納入 git。
