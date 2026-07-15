# OxideSeeker — 以图搜文件系统架构设计（v2）

## 项目概述

**OxideSeeker** 是一个运行在 Windows 上的 Rust 后端服务，支持局域网用户通过浏览器上传截图、快速匹配服务器上的 PDF/AI 设计文件。

- **核心算法**：DINOv2 视觉向量（主，整页 + 重叠 tile 双粒度） + OCR 文本通道（FTS5） + pHash 内存表，**三信号融合排序**
- **典型查询**：**局部截图为主**（用户截取页面的一部分），整页截图为辅
- **运行环境**：纯 CPU，16核+，128GB 内存，无 GPU
- **文件规模**：几十万个 PDF / Adobe Illustrator (AI) 文件
- **访问方式**：局域网浏览器，Web UI

> v1（CLIP ViT-B/32 + 不重叠 grid tile + SQLite pHash 全表扫描）在 30 万文件
> 规模下检索质量与延迟均不可用，v2 为完全重构，数据库/索引与 v1 不兼容。
> 重构决策与阶段性成果见 `REFACTOR_PLAN.md`，OCR 通道方案见 `OCR_PLAN.md`。

---

## 系统架构总览

```mermaid
graph TD
    A[用户浏览器] -->|上传截图 前端预压缩 2048| B[Web 服务层 Axum]
    B -->|尺寸校验 60MP + 降采样 1600| C[搜索引擎]
    C -->|DINOv2 编码| C1[视觉向量]
    C -->|PP-OCR 识别 与编码并行| C2[查询文本]
    C1 -->|HNSW 取500 ef=256| E[HNSW 向量索引 hnsw_rs]
    C2 -->|FTS5 MATCH bm25| FT[page_ocr_fts]
    C -->|u64 XOR+popcount rayon| D[pHash 内存表]
    D & E & FT -->|按 page_id 聚合| F[三信号融合 ranker]
    F -->|返回文件信息+缩略图| A

    G[文件索引器 子进程隔离] -->|扫描文件系统| H[PDF/AI 文件]
    H -->|渲染 640px| I[图像渲染 pdfium]
    I -->|整页 + 9个50%重叠tile| J[DINOv2 推理 + pHash]
    J -->|向量 BLOB 落库| S[(SQLite regions 表)]
    S -->|region_id 回填| E
    S -->|region_id 回填| D
    S -.图缺失/损坏时.->|流式重灌 ~20min 无需推理| E

    O[OCR 回填 独立子进程] -->|渲染 1280px + PP-OCR| FT
    K[文件监控 notify] -->|增量更新| G
```

> 所有 FFI 密集环节（pdfium 渲染、DINOv2/OCR 推理）都跑在崩溃隔离子进程中，
> 子进程内置请求看门狗防挂死；父进程只做 DB 与内存索引维护。详见「§9 崩溃
> 隔离与看门狗」。

---

## 核心技术选型

### 1. 图像向量化 — DINOv2 ViT-S/14（ONNX）

| 项目 | 选型 |
|------|------|
| 模型 | `dinov2_vits14`（Meta，自监督，Apache-2.0） |
| 推理引擎 | `ort` 2.0.0-rc.12（锁定精确版本） |
| 向量维度 | 384 维 float32，L2 归一化（DistDot 跳过重复归一化） |
| 输入 | 224×224（patch 14 → 16×16 patches），**letterbox** 保长宽比，ImageNet mean/std 归一化 |
| 模型文件 | FP32 约 85 MB；INT8 量化约 25 MB（CPU 2-3× 加速） |
| 导出 | `python scripts/export_dinov2.py`（含 ONNX/Torch 输出一致性自检） |

**为什么从 CLIP 换成 DINOv2？**

任务是**实例检索**（找"同一张图"），不是语义检索。CLIP 的向量空间按语义组
织——30 万库中同版式设计稿全部挤在 0.85-0.95 相似度区间，库越大区分度坍塌
越严重。DINOv2 的自监督特征在 instance-level 检索基准上显著优于 CLIP，对局
部-整体匹配（局部截图召回整页）泛化更好，且 FLOPs 仅为 CLIP ViT-B/32 的
~1.5×。升级路径：`dinov2_vitb14`（768 维，更准，~3× 慢），代码零改动（维度
从输出自动探测），仅需全量重建。

**引擎与模型解耦**：`VisionEmbedder` 接受任意输入为 `pixel_values
[N,3,224,224]`、输出 `[N,D]` 的 ONNX 编码器；嵌入维度 D 运行时探测。

#### 推理并发模型

| 类型 | 用途 | 数量 | 加锁 |
|------|------|------|------|
| `VisionEmbedder`（工厂） | 持模型路径，`new_session(intra_threads)` 按需生成 | 单例 `Arc` | 无 |
| `VisionSession`（推理句柄） | 拥有一个 `ort::Session` | 索引器每 worker 各一份（intra=1）；查询路径单例 `Mutex`（intra=全核） | 仅查询路径 |

### 2. 双粒度索引 — 整页 + 50% 重叠 tile（局部截图召回的核心）

每页产生 **10 个 region**：整页 1 个 + 重叠 tile 9 个。

- tile 尺寸 = 页面的 ½（每边），步长 = ¼ → 每边 3 个位置，3×3 = 9 个，相邻重叠 50%
- **覆盖保证**：任何小于页面 ¼ 的目标区域必被某个 tile 完整包含；¼-½ 的大
  部分落入某 tile；更大的由整页向量覆盖
- v1 的不重叠 3×3 grid 有边界效应：目标跨 tile 边界时两边都只含一半，全都
  不匹配——这是局部截图召回差的主要原因之一
- `indexer.tiles_enabled = false` 可退化为每页 1 向量

### 3. 向量存储与索引 — SQLite BLOB（真相源） + hnsw_rs（可重建缓存）

**向量以 f32-LE BLOB 存进 `regions.vector`，HNSW 图只是缓存**：

- `regions.id` 即 HNSW 向量 id——DB 行与 ANN 条目永不漂移
- 图文件缺失/损坏/参数调整/墓碑压缩 → 启动时从 DB 流式重灌
  （600 万向量 ~15-25 分钟），**永不需要重跑模型推理**（重跑是天级成本）
- 写入顺序：DB 事务先提交 → 内存结构（HNSW + pHash 表）后插入；
  中间崩溃只留下可重建的缺口，不会有悬空引用

HNSW 参数：`M=16`, `ef_construction=200`, **`ef_search=256`**（v1 的 64 在
数百万向量下召回不足），查询 over-fetch `max(top_k×25, 500)`（10 region/页
聚合后仍有充足页级候选）。

### 4. pHash — 内存表 + rayon 并行扫描

- 每个 region（含 tile）一个 64-bit pHash（DoubleGradient 8×8），
  SQLite 存 `INTEGER`（u64 位转换 i64），**u64 全程无 hex 字符串**
- 启动时全量加载进 `PhashStore`（600 万条 ≈ 96 MB），查询时 rayon
  XOR+popcount，单次扫描 <10ms
- v1 每次查询 `SELECT id, phash FROM pages` 全表拉取 + 逐行 String 分配，
  30 万文件时**这一步就是秒级延迟的主源**
- 重叠 tile 的独立 pHash 让"局部截图对准某个 tile 位置"也能享受近重复强信
  号——v1 的整页 hash 对局部查询完全失效
- 候选截断 2000，防止低熵页面（色卡/留白）泛滥

### 5. OCR 文本通道 — PP-OCR + FTS5（同版式区分的关键）

评测标定发现命中目标的视觉相似度分布（p05=0.76）与噪声 top1 分布
（p50=0.83）**严重重叠**——视觉信号无法区分「同版式不同文字」的设计稿
（同一模板换个产品名/型号）。文字是唯一正交的强区分信号。

**引擎**：PP-OCRv3 det + rec，**纯 Rust 推理**（复用 `ort`，零新部署依赖）：

| 阶段 | 实现 | 说明 |
|------|------|------|
| det（检测） | DBNet：概率图 → 二值化 → 3×3 膨胀 → 连通域 → 轴对齐框 + DB unclip 扩张 | 印前文字以水平为主，轴对齐省掉旋转矩形/透视变换整套依赖 |
| rec（识别） | CRNN/SVTR：48px 高文字条 → softmax → 贪心 CTC 解码 | 6625 类词表内嵌在 ONNX metadata，部署仍是两个文件 |

阅读顺序排序用**两趟分组**（先按中点全序排序，再切行带内按 x 排）——
早期用单比较器"中点相近算同行"违反传递性，在文字密集页触发 Rust 排序的
非全序 panic（线上故障，已修）。

**全文索引**：SQLite FTS5 + **trigram 分词**：

- 中文无需分词器；子串匹配（截图只含 "UK1457" 也能命中 "VUK1457-115T"）；对 OCR 错字有韧性（其余 trigram 仍匹配）
- `page_ocr` 存文本，FTS5 external-content 表 + 三触发器保持同步
- 查询侧 OCR 文本按空白切段（≥3 字符），最长的 16 段 OR 组合 MATCH，取 top 500 页 + bm25，min-max 归一到 `[0,1]`

**回填模型**：OCR 不在索引管线内做，而由独立后台循环认领缺 `page_ocr` 行
的页（`pages.id` 游标增量扫描），渲染 1280px（640px 对小字号识别不可用）→
识别 → 入库。文件变更时 CASCADE 清行、自动重做。回填走**独立子进程**（同
崩溃隔离协议 + 看门狗），一页挂死不阻塞其余。

### 6. 融合排序 — 三信号

```
score = 0.70 × max(region 视觉 sim)          -- 主信号，按 page 取 region max
      + 0.10 × (1 − min(hamming)/64)          -- 近重复辨识，按 page 取 region min
      + 0.20 × text_score                     -- OCR/FTS bm25 归一化，同版式区分
```

- 三权重可从 `config.toml`（`weight_vector/phash/text`）调整，**无需重编译**
- **文本是独立召回源**：仅文本命中的页（视觉未召回）也进排序，`sim` 取 0 —
  强文本匹配（text_score≈1）得 0.2 分足以进 top-K，这是同版式文件靠文字翻盘
  的机制
- v1 的 first-page / tile-hit bonus 已删除（重叠 tile 下 tile 命中不再携带额外
  信息，且从未被评测验证）
- `similarity_threshold = 0.30` 仅为宽松噪声下限；实际排序由融合分决定；
  **所有权重/阈值调整必须过 `--evaluate` 评测**

**评测实证**（seed=42，纯视觉 → +OCR）：R@1 21.8%→30.4%（+39%）、
MRR 0.323→0.404、medium 桶 R@5 +12.4pp。OCR 主要作用是把已在候选里的正确答
案顶到前排（R@1/MRR 涨幅 >> R@10）。

### 7. PDF 渲染 — pdfium-render

| 项目 | 选型 |
|------|------|
| 库 | `pdfium-render` 0.8（`thread_safe` feature） |
| 索引分辨率 | 长边 **640 px**（tile = ½ 页 ≈ 320px → 缩到 224 仍有细节余量；pHash 更稳定） |
| OCR 分辨率 | 长边 **1280 px**（小字号识别需要） |
| 隔离 | 全部渲染跑在子进程；FFI 崩溃只死子进程，父进程重启之 + crash_attempts 毒丸黑名单 |

**AI 文件**：内嵌 PDF 内容，pdfium 直接解析。
**拼版过滤**：XMP `egExtFL:files` 原字节预扫描，PDFium 打开前零成本跳过。

### 8. 崩溃隔离与看门狗

FFI 代码有两种失败模式，都会拖垮进程内调用方，故全部推到子进程：

1. **崩溃**（结构化异常，如 pdfium `0xE0000008`）：Rust `catch_unwind` 抓不
   到 SEH。子进程死 → 父进程观察到管道断裂 → 记失败/空结果、重启子进程。
2. **挂死**（FFI 调用既不返回也不崩，线上在 OCR 分辨率的病态页面上出现过）：
   子进程内置**看门狗线程**，每个请求前 arm 一个 deadline（索引 600s / OCR
   120s），超时则 `exit(3)`——父进程同样当管道断裂处理。

`crash_attempts` 计数在每次尝试前持久化（WAL），崩溃后仍在，达阈值自动加入
毒丸黑名单。这套隔离同时覆盖索引 worker 与 OCR 回填 worker。

### 9. NaN 向量三道防护

INT8 量化模型在病态页面上偶发输出 NaN 向量，NaN 进入 HNSW 后毒化后续所有
距离排序（违反全序 → Rust 排序 panic，线上故障）。三道防线：

1. **源头**：`l2_normalize` 检测非有限值 → 替换为零向量（与一切距离为 1，
   永不参与排名）
2. **排序**：所有 f32 排序用 `total_cmp`（NaN 混入也不 panic）
3. **重建**：HNSW 从库重建时过滤历史毒向量（删 dump 重启即清理）

### 10. Web / 存储 / 监控

- Axum：multipart 上传、base64 剪贴板、WebSocket 进度、静态资源编译内嵌
- **大图处理**：前端 canvas 预压缩到长边 2048；服务端上传上限 64MB，读图头
  校验 ≤60MP（超限返回明确 400 而非 opaque 500），解码后统一降到长边 1600
  再进管线（避免上亿像素原图在编码/OCR/pHash 里多次克隆）
- SQLite（sqlx）：WAL + synchronous=NORMAL + 外键
- 无损 WebP 缩略图（256px，仅页级一张）
- notify watcher + 定时重扫 + content_sha1 短路（mtime 变了但字节没变不重索引）
- HNSW `save()` 加互斥锁：周期重扫与 watcher 删除并发保存会争 tmp 文件互相
  报错（线上 868 次），串行化解决

### 11. 评测框架 — `--evaluate`（合成查询集）

```bash
oxide_seeker.exe --evaluate --config config.toml \
    --samples 300 --queries-per-page 3 --seed 42 \
    --label my-run --out eval.json
```

- 从已索引页**确定性采样**（拉全量 → seeded RNG 洗牌 → 截断；同 seed 严格
  复现同一查询集，早期用 SQL `ORDER BY RANDOM()` 无法复现，已修）
- pdfium 以 900px（≠索引侧 640px，避免像素同源）重渲染 → 随机裁剪
  （面积 8%-90%，模拟局部截图为主的分布）+ 缩放 + 亮度扰动 + JPEG 往返
- 指标：Recall@1/@5/@10、MRR、**分阶段延迟 P50/P95**（encode/ocr/ann/phash/
  rank，定位瓶颈）、按裁剪面积分桶、**阈值标定统计**（命中/噪声相似度分布）
- 输出 console 报告 + JSON（`--label` 标记，跨改动 diff）
- 与服务实例可并存（无单实例锁），只读搜索栈；OCR 通道按配置自动启用

---

## 数据库 Schema（migrations/）

`001_initial.sql`（视觉三表）：

```sql
files   (id, path UNIQUE, filename, file_type, file_size, modified_at,
         page_count, is_excluded, indexed_at, created_at,
         crash_attempts, exclusion_reason, content_sha1)

pages   (id, file_id→files, page_num, width_px, height_px, thumb_path,
         UNIQUE(file_id, page_num))          -- 每页一行

regions (id,               -- == HNSW 向量 id
         page_id→pages, kind 'full'|'tile', idx 0-8,
         bbox_x/y/w/h,     -- 归一化坐标,整页=(0,0,1,1),预留圈选高亮
         phash INTEGER,    -- u64 位转换 i64
         vector BLOB,      -- f32-LE,维度由字节长度推断
         UNIQUE(page_id, kind, idx))

index_tasks (id, file_id, status, error_msg, attempts, ...)
```

`002_ocr.sql`（OCR 文本通道，纯增量，可加到现有库）：

```sql
page_ocr (page_id→pages PK, text, ocr_at)              -- 每页 OCR 文本
page_ocr_fts USING fts5(text, content='page_ocr',      -- 全文索引
         content_rowid='page_id', tokenize='trigram')
+ 3 触发器（INSERT/DELETE/UPDATE 同步 content → fts）
```

> v1 数据库无迁移路径（嵌入模型变了，向量必须全部重算）：删除 data 目录重建。
> OCR 表则是纯增量——已建好的视觉库直接加表，后台回填即可。

---

## 搜索流程

1. 上传图 → 前端预压缩长边 2048 → 服务端校验 ≤60MP、降采样长边 1600
2. **并行**：DINOv2 编码（~600ms，延迟大头） ‖ PP-OCR 识别（~300ms，被编码遮蔽）
3. 查询图 → pHash u64
4. HNSW 取 500（ef=256），按 region_id 取最小距离合并（~13ms）
5. pHash 内存表 rayon 扫描，阈值 ≤12，截断 2000（~37ms）
6. 查询文本 → FTS5 MATCH，取 top 500 页 + bm25 归一化
7. `get_region_hits` 一次 join 解析 region → page，按 page_id 聚合三信号
8. 融合排序 → Top-K → 批量取 pages/files 行组装（缩略图、元数据）

**实测延迟 P50 ≈ 746ms**（INT8 模型 + OCR，编码占 ~600ms）。查询侧 OCR 崩溃
/panic 降级为纯视觉搜索，不会让请求失败。

## 索引流程

1. 扫描 + 过滤（mtime → content_sha1 短路 → crash 黑名单）
2. 子进程：pdfium 渲染 640px → 切 1+9 region → 批量推理（chunk 32）→ 每 region pHash
3. 父进程：单事务 INSERT pages+regions（含向量 BLOB）→ 拿回 region_ids
4. HNSW `add_batch` + PhashStore `add_batch`（失败可由重建自愈）
5. 每 500 文件 checkpoint 保存 HNSW dump
6. **OCR 回填**（独立后台循环）：认领缺 `page_ocr` 的页 → 子进程渲染 1280px +
   识别 → 入库；扫空后给孤儿页（已索引但后被排除的文件页）写空占位行使
   `ocr_pending` 收敛到 0

**索引速度估算**（16 核，INT8，tiles on）：向量索引 ≈ 5-9 页/s/worker，
30 万文件 ≈ 7-15 小时；OCR 回填 8-17 小时后台独立进行，不阻塞搜索。一次性
成本，向量落库后永不重复。

---

## 项目模块结构

```
src/
├── main.rs                    # 入口：--worker-mode / --evaluate / 服务启动
├── evaluate.rs                # 合成查询评测框架（指标 + 分阶段延迟 + 标定）
├── config.rs                  # 配置（tiles_enabled / 融合权重 / OCR 路径等）
├── worker_proc.rs             # 子进程：渲染 + tile 切割 + 推理 + pHash + OCR，含看门狗
│
├── embedder/
│   ├── vision.rs              # VisionEmbedder/VisionSession（模型无关 ONNX 编码器 + NaN 防护）
│   ├── image_prep.rs          # letterbox + ImageNet 归一化 → NCHW 224
│   └── phash.rs               # u64 pHash + XOR/popcount 汉明距离
│
├── ocr/
│   ├── mod.rs                 # OcrEngine（det+rec 编排 + 置信度过滤）
│   ├── det.rs                 # DBNet 检测：概率图 → 连通域 → 轴对齐框 + 两趟阅读序
│   └── rec.rs                 # CRNN 识别：文字条预处理 + 贪心 CTC 解码（词表内嵌）
│
├── indexer/
│   ├── mod.rs                 # start_full_index（扫描/过滤/清理编排）
│   ├── scanner.rs / watcher.rs / filter.rs / pdf_processor.rs
│   ├── subprocess.rs          # 父子进程协议（WorkerRequest: Index | OcrPage）
│   ├── worker_pool.rs         # DB 先行提交 → region_ids 回填双内存索引
│   └── ocr_backfill.rs        # 后台 OCR 回填循环（子进程隔离 + 游标增量）
│
├── search/
│   ├── mod.rs                 # SearchEngine（并行编码+OCR）+ rebuild_index_if_needed + 大图处理
│   ├── vector_index.rs        # hnsw_rs 封装（id=regions.id, ef=256, 墓碑, save 互斥）
│   ├── phash_store.rs         # 内存 pHash 表（rayon 扫描, 增量维护）
│   └── ranker.rs              # page_id 聚合 + 三信号融合（total_cmp 排序）
│
├── storage/
│   ├── database.rs            # 五表 CRUD + 向量 BLOB 编解码 + FTS 检索 + 流式重建
│   └── thumbnail.rs
│
└── web/                       # Axum 路由/handlers/WS（含大图预压缩前端）
```

---

## 数据规模核算（30 万文件，约 28 万页 × 10 region ≈ 280 万向量）

| 项 | 规模 | 占用 |
|----|------|------|
| HNSW 常驻 | 280 万 × 384d f32 | ~4.3GB RAM + 图 ~1GB |
| pHash 内存表 | 280 万 × 16B | ~45MB RAM |
| SQLite（含向量 BLOB + OCR 文本 + FTS） | | ~6GB 磁盘 |
| 合计常驻 | | **~6GB RAM** / 128GB，余量充足 |

> 规模随每文件页数线性变化；上表按实测约 0.8 页/文件估算。128GB 内存下即使
> 升 ViT-B/14（768d，向量翻倍）或加 tile 金字塔 L2 层也毫无压力。

---

## 部署说明

### 所需文件
1. `oxide_seeker.exe` + `oxide_seeker_service.exe`（可选服务包装）
2. `config.toml`
3. `dinov2_vits14_int8.onnx`（视觉，由 `scripts/export_dinov2.py` 导出）
4. `ppocr_det.onnx` + `ppocr_rec.onnx`（OCR，从 RapidOCR 拷贝；缺失则纯视觉降级）
5. `onnxruntime.dll`、`pdfium.dll`

> ⚠️ `onnxruntime.dll` 必须在 exe 旁边——否则 Windows 可能加载 System32 里的
> 旧版导致进程在 ort 版本握手时挂死。

### 从 v1 升级
1. **先用旧版 exe 跑基线评测留档**（旧库上 `--evaluate --label clip-baseline`）
2. 停服务，**删除整个 data 目录**（v1 库不兼容）
3. 替换 exe 与模型文件，更新 config.toml（`grid_size`→`tiles_enabled`、`model_path`、OCR 路径、融合权重）
4. 启动，全量索引自动开始（后台数小时，INT8 减半以上）
5. 索引完成后 OCR 回填自动跑（后台独立，看 `/api/index/status` 的 `ocr_pending`）
6. `ocr_pending` 归零后 `--evaluate --label final` 跑分对比基线

---

## 后续路线（按评测结果决定）

已完成的四阶段（评测框架 → DINOv2+schema → 搜索管线 → OCR 通道）见
`REFACTOR_PLAN.md` / `OCR_PLAN.md`。剩余候选：

| 候选 | 触发条件 | 说明 |
|------|----------|------|
| 真实查询埋点回归 | 上线积累样本后 | 记录用户点击的正确结果，替换合成评测集做黄金回归；也是判断 large 桶回落是否真实的唯一可靠方式 |
| 密集 pHash 网格精排 | 局部小截图（<页面 1/10）召回不足 | 每页预存多尺度滑窗 pHash，对 ANN top-200 精排，成本≈0 |
| AKAZE+RANSAC 几何验证 | 需要"精确模式" | top-30 关键点校验，+300-500ms，near-dup 指哪打哪 |
| DINOv2 ViT-B/14 | ViT-S 区分度不足 | 768 维，~3× 索引时间，零代码改动（维度自动探测） |
| tile 金字塔 L2 | 极小目标召回不足 | +25 tile/页（尺寸⅓步长⅙），内存可承受 |
