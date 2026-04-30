'use strict';

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const dropContent = document.getElementById('dropContent');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const clearBtn = document.getElementById('clearBtn');
const searchBtn = document.getElementById('searchBtn');
const topKSelect = document.getElementById('topK');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const resultsMeta = document.getElementById('resultsMeta');
const loadingOverlay = document.getElementById('loadingOverlay');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const cardTpl = document.getElementById('resultCardTpl');

const settingsBtn = document.getElementById('settingsBtn');
const settingsModal = document.getElementById('settingsModal');
const closeSettings = document.getElementById('closeSettings');
const dirList = document.getElementById('dirList');
const addDirBtn = document.getElementById('addDirBtn');
const saveSettings = document.getElementById('saveSettings');
const saveStatus = document.getElementById('saveStatus');

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null; // File | null
let wsRetryTimer = null;
let indexFinished = false; // Guard to prevent progress bar re-showing after finish
let currentScanDirs = [];

// ── Utilities ─────────────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (!bytes) return '–';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function formatSimilarity(v) {
  return (v * 100).toFixed(1) + '%';
}

// ── Image selection ───────────────────────────────────────────────────────────
function setImage(file) {
  if (!file || !file.type.startsWith('image/')) {
    alert('请选择有效的图片文件（PNG、JPEG 或 WebP）');
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.classList.remove('hidden');
  dropContent.classList.add('hidden');
  clearBtn.classList.remove('hidden');
  dropZone.classList.add('has-image');
  searchBtn.disabled = false;
}

function clearImage() {
  selectedFile = null;
  previewImg.src = '';
  previewImg.classList.add('hidden');
  dropContent.classList.remove('hidden');
  clearBtn.classList.add('hidden');
  dropZone.classList.remove('has-image');
  searchBtn.disabled = true;
  resultsSection.hidden = true;
}

clearBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearImage();
});

// ── Drag & Drop ─��─────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer?.files?.[0];
  if (file) setImage(file);
});
dropZone.addEventListener('click', (e) => {
  if (e.target === clearBtn) return;
  // 避免点击“选择图片文件”label时触发两次 fileInput.click()
  if (e.target.closest('.upload-btn')) return;
  if (!selectedFile) fileInput.click();
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) {
    setImage(fileInput.files[0]);
  }
  // 关键：无论选择还是取消，都清空 value，避免后续事件状态残留
  fileInput.value = '';
});

// ── Clipboard paste ────────────────────────────────────────────────��──────────
document.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) {
        setImage(file);
        break;
      }
    }
  }
});

// ── Search ──────────────────────��─────────────────────────────────────────────
searchBtn.addEventListener('click', runSearch);

async function runSearch() {
  if (!selectedFile) return;
  const topK = parseInt(topKSelect.value, 10);

  // Show inline loading indicator and disable button — page remains interactive
  loadingOverlay.hidden = false;
  searchBtn.disabled = true;
  resultsSection.hidden = true;

  try {
    const form = new FormData();
    form.append('image', selectedFile);
    form.append('top_k', String(topK));

    const resp = await fetch('/api/search', {method: 'POST', body: form});
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({error: resp.statusText}));
      throw new Error(err.error || resp.statusText);
    }
    const data = await resp.json();
    renderResults(data);
  } catch (err) {
    alert('搜索失败：' + err.message);
  } finally {
    loadingOverlay.hidden = true;
    searchBtn.disabled = false;
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  resultsGrid.innerHTML = '';
  const {results, search_time_ms} = data;

  resultsMeta.textContent =
    `找到 ${results.length} 个结果，耗时 ${search_time_ms} ms`;

  if (results.length === 0) {
    resultsGrid.innerHTML =
      '<p style="color:var(--text-muted);grid-column:1/-1;text-align:center;padding:40px">未找到相似文件</p>';
  } else {
    for (const r of results) {
      resultsGrid.appendChild(buildCard(r));
    }
  }

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({behavior: 'smooth', block: 'start'});
}

function buildCard(r) {
  const node = cardTpl.content.cloneNode(true);
  const card = node.querySelector('.result-card');

  const thumb = card.querySelector('.card-thumb');
  if (r.thumbnail_url) {
    thumb.src = r.thumbnail_url;
    thumb.alt = r.filename;
  } else {
    thumb.src = '';
    thumb.style.display = 'none';
    card.querySelector('.card-thumb-wrap').style.background = '#2a2e45';
  }

  card.querySelector('.card-page-badge').textContent =
    r.page_count > 1 ? `第 ${r.page_num} 页` : '';

  const filenameEl = card.querySelector('.card-filename');
  filenameEl.textContent = r.filename;
  filenameEl.title = '点击复制文件名';
  filenameEl.addEventListener('click', () => {
    navigator.clipboard.writeText(r.filename).then(() => {
      filenameEl.textContent = '✓ 已复制';
      setTimeout(() => {
        filenameEl.textContent = r.filename;
      }, 1500);
    });
  });

  const pathEl = card.querySelector('.card-path');
  pathEl.textContent = r.file_path;
  pathEl.title = '点击复制路径';
  pathEl.addEventListener('click', () => {
    navigator.clipboard.writeText(r.file_path).then(() => {
      pathEl.textContent = '✓ 已复制';
      setTimeout(() => {
        pathEl.textContent = r.file_path;
      }, 1500);
    });
  });

  card.querySelector('.card-similarity').textContent =
    '相似度 ' + formatSimilarity(r.similarity);
  card.querySelector('.card-size').textContent =
    formatBytes(r.file_size);
  card.querySelector('.card-pages').textContent =
    r.page_count > 1 ? r.page_count + ' 页' : '';

  return node;
}

// ── WebSocket: index progress ─────────────────────────────────────────────────
function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/progress`);

  ws.addEventListener('message', (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      handleProgressMsg(msg);
    } catch { /* ignore */
    }
  });

  ws.addEventListener('close', () => {
    wsRetryTimer = setTimeout(connectWs, 3000);
    setStatus('yellow', '连接已断开，重连中…');
  });

  ws.addEventListener('error', () => {
    ws.close();
  });
}

function handleProgressMsg(msg) {
  if (msg.type === 'status') {
    const {status, total_files, indexed_files, excluded_files, failed_files, progress_percent} = msg;
    
    // Logic: Treat >= 99.5% or status "ready" as finished. 
    // This provides a smooth exit even if there are tiny database/atomic mismatches.
    const effectivePct = typeof progress_percent === 'number' ? progress_percent : 0;
    const isEffectivelyDone = status === 'ready' || effectivePct >= 99.5;

    if (isEffectivelyDone) {
      indexFinished = true;
      setStatus('ready', `索引完成，共 ${total_files} 个文件`);
      return;
    }

    if (status === 'indexing') {
      // If we previously finished, don't show indexing again unless it's a significant new task
      if (indexFinished && effectivePct < 90) {
          indexFinished = false;
      }

      if (!indexFinished) {
        const done = (indexed_files || 0) + (excluded_files || 0) + (failed_files || 0);
        setStatus('yellow', `索引中… ${done}/${total_files} (${effectivePct.toFixed(1)}%)`);
      }
    } else if (status === 'error') {
      setStatus('error', '索引错误');
    }
  }
}

function setStatus(cls, text) {
  statusDot.className = 'status-dot' + (cls === 'ready' ? ' ready' : cls === 'error' ? ' error' : '');
  statusText.textContent = text;
}

// ── Poll index status (fallback if WS not available) ─────────────────────────
async function pollStatus() {
  try {
    const resp = await fetch('/api/index/status');
    if (!resp.ok) return;
    const data = await resp.json();
    handleProgressMsg({type: 'status', ...data});
  } catch { /* ignore */
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
connectWs();
pollStatus();
setInterval(pollStatus, 5000);

// ── Settings Logic ───────────────────────────────────────────────────────────
async function initSettings() {
  try {
    console.log('Fetching config...');
    const resp = await fetch('/api/config');
    if (!resp.ok) {
        console.error('Config fetch failed:', resp.status);
        return;
    }
    const data = await resp.json();
    console.log('Config received:', data);
    if (data.is_local) {
      console.log('Local access detected, showing settings button');
      settingsBtn.hidden = false;
      currentScanDirs = data.scan_dirs || [];
      renderDirList();
      if (currentScanDirs.length === 0) {
        console.log('No scan directories configured, showing settings modal automatically');
        settingsModal.hidden = false;
      } else {
        settingsModal.hidden = true;
      }
    } else {
      console.warn('Non-local access detected (is_local: false), settings button hidden. IP might not be in loopback range.');
    }
  } catch (e) { console.error('Failed to fetch config', e); }
}

function renderDirList(dirs) {
  const list = dirs || currentScanDirs;
  dirList.innerHTML = '';
  list.forEach((dir, index) => {
    const item = document.createElement('div');
    item.className = 'dir-item';
    item.innerHTML = `
      <input type="text" class="dir-input" value="${dir}" data-index="${index}" placeholder="例如: C:\\Users\\Documents" />
      <button class="remove-btn" data-index="${index}" title="移除">✕</button>
    `;
    dirList.appendChild(item);
  });
}

function getEditingDirs() {
  const inputs = dirList.querySelectorAll('.dir-input');
  return Array.from(inputs).map(i => i.value.trim());
}

function hasUnsavedChanges() {
  const editing = getEditingDirs().filter(d => d !== '');
  const saved = currentScanDirs.filter(d => d !== '');
  if (editing.length !== saved.length) return true;
  return editing.some((d, i) => d !== saved[i]);
}

settingsBtn.addEventListener('click', () => {
  renderDirList();
  settingsModal.hidden = false;
  saveStatus.textContent = '';
});

closeSettings.addEventListener('click', () => {
  if (hasUnsavedChanges()) {
    if (!confirm('有未保存的更改，确定放弃吗？')) return;
    renderDirList();
  }
  settingsModal.hidden = true;
  saveStatus.textContent = '';
});

addDirBtn.addEventListener('click', () => {
  const editing = getEditingDirs();
  editing.push('');
  renderDirList(editing);
});

dirList.addEventListener('click', (e) => {
  if (e.target.classList.contains('remove-btn')) {
    const index = parseInt(e.target.dataset.index, 10);
    const editing = getEditingDirs();
    editing.splice(index, 1);
    renderDirList(editing);
  }
});

saveSettings.addEventListener('click', async () => {
  const inputs = dirList.querySelectorAll('.dir-input');
  const dirs = Array.from(inputs).map(i => i.value.trim()).filter(d => d !== '');

  saveSettings.disabled = true;
  saveStatus.textContent = '保存中...';
  saveStatus.style.color = 'var(--text-muted)';

  try {
    const resp = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scan_dirs: dirs })
    });
    if (!resp.ok) throw new Error(resp.statusText);

    saveStatus.textContent = '保存成功！正在启动索引...';
    saveStatus.style.color = 'var(--green)';

    // Update current state so it doesn't pop up again on refresh (if data was re-fetched)
    // and so that re-opening the modal shows the correct dirs.
    currentScanDirs = dirs;

    // Close modal after a short delay
    setTimeout(() => {
        settingsModal.hidden = true;
    }, 1500);

    // Trigger immediate status poll to show "indexing"
    setTimeout(pollStatus, 500);

  } catch (e) {
    saveStatus.textContent = '保存失败: ' + e.message;
    saveStatus.style.color = 'var(--red)';
  } finally {
    saveSettings.disabled = false;
  }
});

initSettings();