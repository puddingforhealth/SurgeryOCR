# Google Drive → 雲端 RAM 分析 → 本機結果

`tools/drive_batch.py` 的 Linux worker 使用獨立的 SurgeryOCR 程式快照與既有 GPU Python 環境，
以一個 rclone 下載器和一個 RAM 分析程序重疊執行。本機 collector 持續取回完成結果。
這是 SurgeryOCR 的分析流程工具，通用連線和傳檔由獨立 `rcs` 負責。

## 目前批次

私人清單、連線設定與操作紀錄保存在被 git 忽略的 `build/drive-batch/`，不提交帳號或影片資料。
本批次已於 2026-10-23 切換到 `D:/SurgeryOCR/runs/士林/<Drive 原始子資料夾>/<影片 stem>/`，包含完整 RAM 分析數據、
`pipeline.log`、來源對應 `_source.json`、逐檔校驗清單 `_files.json`、取回確認 `_received.json`。
不保存逐幀影像或 ROI 截圖。不同月份的同名影片不會共用輸出目錄。

影片也從 Drive 逐部下載到同一個月份目錄：

```text
D:/SurgeryOCR/runs/士林/202507/
  clip.MP4
  clip/
    stage_analysis.json
    region1_ocr_testing.jsonl
    ...
```

本機影片下載與遠端分析是兩個獨立工作；分析完成數不代表本機已有同樣數量的影片。
本機下載器預設一次一部、8 MiB/s、保留 20 GiB，使用既有 `gdrive` 登入。
下載中的檔案使用 `.part`（rclone 另有 `.partial` 暫存），大小與 MD5 符合批次清單後才發布為 MP4。
最多重試三輪，錯誤保留在狀態及 `_download_logs/`；重啟會驗證並略過已完成影片。
既有但內容不同的正式影片不會被覆蓋。來源 Drive 不會被修改。

GUI 載入此處的 MP4 後會找旁邊同名資料夾；如果先前手動指定了另一個分析根目錄，
可用「分析資料夾…」選影片所在月份，明確使用這一份結果。沒有本機影片時仍可閱讀 JSON，
但 GUI 的影片播放與逐幀對照需要影片。

遷移時逐檔校驗了 35 部、457 個檔案。原 `runs/drive/士林/` 保留為切換當時的備份，
不再更新；後續結果和影片以 D 槽為準。私人設定仍在 `build/drive-batch/`，封裝暫存改到
`D:/SurgeryOCR/.transfer-cache/士林/`。這是本批設定，不會改變其他批次的輸出預設值。

```powershell
# 查看最後一次回收到本機的進度（不是即時遠端查詢）
python -X utf8 tools/drive_batch.py status build/drive-batch/collect.json

# 本機 collector 中斷後接回；有程序鎖避免同時啟動兩個
python -X utf8 tools/drive_batch.py collect build/drive-batch/collect.json

# 查看本機影片下載進度（與雲端分析進度分開）
python -X utf8 tools/drive_local.py status build/drive-batch/local-download.json

# 本機影片下載器中斷後接續；既有影片先校驗，不重複下載
python -X utf8 tools/drive_local.py download build/drive-batch/local-download.json
```

本機接收器會在背景運作。關閉 terminal 不影響它；電腦關機、睡眠或斷網時無法回收。
雲端完成但尚未收到回收確認的影片也計入暫存上限，因此不會持續下載到磁碟滿。
重新啟動 collector 後繼續回收；來源 Drive 檔案不會被改動或刪除。

## 即時進度與 tmux

本次已建立獨立的 `drive-progress` tmux 看板，每 2 秒更新。從本機進入：

```powershell
rcs shell zettabyte
```

進入遠端後執行：

```bash
tmux attach -t drive-progress
```

看板顯示已回收到本機的部數、待處理／下載／分析／失敗數量、當前影片、
分析 log 的幀數與預估剩餘時間、下載 log，以及暫存磁碟剩餘容量。
下載 log 每 30 秒更新；`Last state change` 指狀態轉換時間，不是程序存活保證。
分析本身是獨立背景程序，看板只讀取檔案，不控制分析程序。
按 **Ctrl+B，放開後按 D** 離開看板，工作持續；Ctrl+C 只停止看板，不停止分析。

看板關閉後可在遠端重新建立：

```bash
tmux new-session -s drive-progress 'python3 /tmp/surgeryocr-drive-20261022/progress_view.py watch /tmp/surgeryocr-drive-20261022'
```

也能不用 tmux，直接在遠端執行同一個 `watch` 指令；加 `--once` 僅印一次。
`progress_view.py` 是部署時另存的工具副本，更新看板不會替換正在工作的 worker。

## 保護與恢復

- 一次下載一部；最多三部尚未回收的成功／進行中影片，包含正在分析的一部。本機下載保持單一串流及 8 MiB/s 限速。
- 每部下載前檢查可用容量，預留至少 20 GiB；容量不足等待，不自行降低預留值。
- 影片大小和 MD5 必須符合準備清單；下載失敗最多三輪，每輪由 rclone 處理有限次重試。
- 只分析下載驗證成功的影片。程序 exit 0 還不夠：必須有 stage、OCR、run_stats 且 frame counts 一致。
- 本機驗證封裝 SHA-256、每個輸出檔 SHA-256 與檔案清單，才發布結果並發送 ack。
- 遠端只有收到 checksum 相符的 ack 才移除成功影片的暫存、分析資料與封裝，保留狀態紀錄。
- 下載失敗的副本、分析失敗的輸入不持續堆積；原始影片仍在 Drive，失敗 log／狀態保留供重試。
- 重跑 worker 會略過 ready／collected，重新處理其他項目；不要重新部署或覆蓋正在執行的 manifest。
- `/tmp` 不是持久備份；容器被移除後需重新部署，不能在回收完畢前釋放資源。
- 雲端遇到反覆 EOF 時，本批次的 `bin/rclone` 包裝器改呼叫 `range_download.py`：以兩個 32 MiB 範圍請求下載同一部影片，各自限速 16 MiB/s；已完成分段保存 SHA-256，下次只補抓缺少或損壞的分段。合併後仍以清單的整片 MD5 校驗，才交給分析。尚未完成的單一分段需重抓，不宣稱任意位元組位置續傳。

2026-10-23 曾觀察到 Drive `unexpected EOF` 及 `rateLimitExceeded`。下載中／重試中時 GPU
可以閒置；tmux 看板仍存在並不表示正在分析。另有其他 GPU 程序時，GPU 總使用率也不能當作
本批分析進度。原生下載器保留為 `bin/rclone-real`，分段快取放在批次的 `range-parts/`。
完成組裝後移除該片分段；中斷的已驗證分段保留供重試，仍受可用容量檢查限制。

`status` 顯示的是帶時間戳的本機快照。`download_failed`、`analysis_failed` 或 `producer_error`
都不代表完成；完整成功需全部項目為 `collected` 且批次為 `complete`。

## 夜間監看與整輪重試

`tools/drive_watch.py` 每 30 秒更新心跳，利用既有 worker 的檔案鎖判斷工作是否還在。
它不終止正在下載或分析的程序；程序退出而清單仍有未完成項目時才重新啟動。
重啟會略過已回收結果，下載器則續用已驗證分段。連續重啟以 5、10、20、30 分鐘退避，
之後上限維持 30 分鐘；有新的完成項目會重置退避。整輪失敗不會被當成全部完成。

```powershell
python tools/drive_watch.py download build/drive-batch/local-download.json
python tools/drive_watch.py collect build/drive-batch/collect.json
```

雲端使用 `python3 /tmp/surgeryocr-drive-20261022/drive_watch.py worker /tmp/surgeryocr-drive-20261022`。
這次部署的三個監看均為獨立背景程序；關閉聊天不影響它們。電腦睡眠、重開機、容器被移除，
則不保證監看存活；本機下載與回收需要電腦保持開機及連線。沒有建立登入自啟動或外部通知。

本機輸出根目錄的 `_watch_download.json`、`_watch_collect.json`，以及遠端批次根目錄的
`_watch_worker.json` 記錄最新心跳、完成數、失敗 ID 與下次重試時間。
對應 `.log` 保存監看程序所啟動的 worker 日誌。建立同名 `.stop` 檔可停止監看；
它不會殺掉已在執行的工作，重新啟用監看前需移除 `.stop`。
同一角色的監看本身也持有鎖，避免重複啟動。

2026-10-23 實機確認三個監看均辨識原有工作鎖、心跳持續更新，沒有啟動第二份分析。
回歸測試涵蓋下載失敗後退避重啟、活動 worker 不被重啟、已回收項目不重跑、
未列入狀態的影片仍算待處理，以及重新開始批次前清除舊 `finished_at`／錯誤狀態，
避免回收程序把新一輪誤認為已結束。監看無法解除 Drive 配額限制，也不會強制殺掉卡住的活程序。

同晚實機亦發現一部影片整片 MD5 正確，但 `run_stats.frames=48253`、
`stage_analysis.total_frames=48258`。此項被維持為 `analysis_failed`，不回收為成功結果，
交由後續輪次重試。來源容器宣告幀數與解碼實得幀數不同，尚不能只憑下載成功就排除
來源或解碼問題；不應為了完成批次而放寬幀數一致性檢查。

## 驗證

```powershell
$env:PYTHONUTF8 = '1'
python -X utf8 -m unittest discover -s tests -p 'check_*.py' -v
python tools/golden.py check --set full --rtol 0
```

測試覆蓋目錄映射、名稱衝突、路徑安全、內容校驗、空間等待、失敗重試、下載／分析重疊、
重啟略過，以及回收確認後清理。實際 Drive 速度、GPU 資源競爭與服務中斷仍需實機觀察。

2026-10-22 驗證：60 項離線測試通過；完整 golden 集合 4 部影片使用 `--rtol 0`
比對均與基準一致。遠端實際完成 RAM 分析、封裝、本機逐檔校驗及確認後清理；
首部 30,833 幀、11 個輸出檔驗證成功。tmux 看板已在實際下載／分析並行時確認顯示幀數。
這是流程驗證，不表示整批 97 部已完成；完整狀態以本機回收紀錄為準。

本機影片下載測試另外覆蓋限速／單一傳輸、先校驗再發布、相鄰結果保持原樣、
重啟略過、既有不同影片拒絕覆寫、校驗失敗重試、空間不足等待，以及 rclone 實際
`.part.<hash>.partial` 檔名的進度讀取。

2026-10-23 實機配對驗證：D 槽首部影片完成大小／MD5 校驗，能解碼 1920×1080 首幀；
影片與分析均為 30,833 幀。GUI 使用的 `resolve_video_analysis_dir` 解析到影片旁邊的
D 槽分析資料夾。此檢查驗證實際檔案與 GUI 路徑邏輯，未操作 GUI 視窗逐項檢查。

新增本機與分段傳輸後，75 項離線測試通過；完整 golden 4 部結果亦全部與基準一致。
加入夜間監看及批次重新開始的狀態修正後，離線測試增至 86 項且全部通過；
監看的 10 項測試亦在實際 Linux 容器通過。本輪完整 golden 4 部重新執行，
全部以 `--rtol 0` 與基準相同（總測試執行時間 1045.1 秒）。
分段測試另驗證 EOF 後只補抓失敗段、保存段損壞重抓、短回應不發布、整片 MD5 不符不發布，
以及不覆寫既有不同影片。雲端實際恢復進度須另查看當前紀錄，測試通過不代表網路配額已解除。

## GUI 機型與幀位置核對

### 本機舊資料集中到 D 槽

2026-10-23 專案原有 `data`、`runs` 的實體資料移至
`D:/SurgeryOCR/runs/本機專案/data` 與 `D:/SurgeryOCR/runs/本機專案/runs`。
專案內同名路徑改為 junction，原指令及樣板路徑仍可使用；不要把這兩個 junction
當成第二份備份。版本控制的樣板仍可透過原路徑讀取及比對。

原有 20 部影片共約 32.94 GiB；含舊分析與截圖快取的完整搬移約 41.36 GiB，
共 485,601 個檔案。切換前比對全部檔案清單／大小／修改時間，以及 1,006 個影片、
分析文字資料與必要樣板的 SHA-256。切換後 20 部影片均可透過原路徑解碼首幀，
94 項離線測試通過。搬移紀錄在 D 槽 `本機專案/_storage_migration.json`。

原有 17 部已有相鄰分析資料夾，另將兩部 0816 影片已驗證的 candidate 結果
複製到影片旁（保留 runs 中的歷史基準與版本），共 19 部可自動配對。
`2025-08-16_133324.MP4` 沒有找到既有分析結果，未憑空建立或標記完成。
比較其他分析版本時，用 GUI「分析資料夾…」選 D 槽 `本機專案/runs` 中對應版本。

### 機型與時間軸

雲端分析會把偵測結果寫進雲端的 `config/rois.json`；只有結果封裝回到本機時，
本機設定不一定有同一影片的機型對照。GUI 現在先讀結果 `_source.json` 的 `machine_id`
（若有），再讀既有 `pipeline.log` 的偵測／已知／fallback 機型，最後才使用本機對照及預設。
機型與來源會顯示在視窗標題。載入不會修改本機對照表，OCR ROI 與 header 同步選用該機型，
機型 3 的階段 ROI 與 pattern 名稱亦依專屬設定載入。設定證據衝突時不畫推測的 OCR 框。

2026-10-23 實機核對 95 包回收日誌及雲端對照，全為機型 2，沒有缺漏；雲端 ROI 的
canonical JSON SHA-256 與來源 commit `272ebedab4d93b657306b53a57b43a5a16c72b11` 相同。
另核對本機 95 部原影片，全部為 1920×1080，影片宣告幀數均與其回收的
`run_stats.frames` 一致，沒有開啟失敗。
曾觀察到 GUI region1 套用機型 1 的 Y=325..397，實際分析用機型 2 的 Y=225..297，
因此框到 ASP 而不是 IOP；此證據確認顯示端錯用座標，不代表逐筆 OCR 的準確率經人工確認。

指定幀號的 GUI 操作直接讀取該幀，避免 `int((frame / fps) * fps)` 的浮點截斷。
對實際案例的第 6964 幀，順序解碼與依幀號跳轉的像素完全相同，PTS 都是 232133.3 ms；
平均 FPS 換算則為 232125.746 ms。這個單一案例無法證明其他影片的平均 FPS 定位可靠；2026-10-24 已將 GUI 秒數、跳轉及時間軸統一改用實際 PTS，詳見[影格與時間同步](viewer-timing.md)。
分析主流程順序解碼每一幀；宣告總幀數用於進度與結果完整性檢查，不決定解碼迴圈次數。
這次修正的 94 項離線測試通過；完整 golden 4 部以 `--rtol 0` 比對相同，
總執行時間 830.2 秒。新增測試涵蓋雲端機型覆蓋本機對照、ROI/header 同步、
切片不沿用前片機型、衝突拒絕，以及小數 FPS 下指定幀號不經時間往返轉換。

### 影片區域自動縮放

GUI 會依實際可用空間等比例顯示完整影格，允許放大，寬或高其中一邊貼齊；OCR 與手術階段 ROI 使用同一組縮放比例。暫停時調整視窗只重畫快取的原始影格，不讀取下一幀、不 seek，也不改變選取的幀。右側標註面板先取得所需寬度，剩餘空間再給影片。

2026-10-23 驗證：102 項離線測試通過。實際 Tk 視窗測試涵蓋 1350×800、1850×1000、1000×800，影片分別顯示為 1013×569、1447×814、663×372；三種尺寸均完整容納影片並貼齊一邊，標註面板保持 320 像素，暫停幀 6964 不變且未觸發影片讀取或定位。此檢查使用合成影格；ROI 縮放另以像素斷言驗證。尺寸紀錄位於本機 `build/gui-fit-live-check.json`。

本次完整 golden 回歸 4 部影片均以 `--rtol 0` 比對完全一致，執行時間 812.6 秒。

## 效能量測紀錄

每部回收的 `run_stats.json` 持續保存幀數、內部分析秒數、分段計時及 OCR 工作量。
2026-10-23 01:49 的 40 部快照已保存到
`D:/SurgeryOCR/runs/士林/_performance/20261023-014929/measurement.json`，包含原始檔 SHA-256、
逐片資料與環境版本；同一截點已整理進使用者指定的附件三。整體 FPS 使用總幀數／總分析秒數，
不平均各片 FPS；下載、排隊、封裝與回傳不計入分析秒數。共用 GPU 的瞬間使用率不能推回
逐片平均使用率或峰值顯存。文件為固定截點，後續每片數據繼續留在各自 `run_stats.json`。
