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
const settingsTitle = document.querySelector('#settingsModal .modal-header h3');
const settingsHint = document.querySelector('#settingsModal .modal-body .hint');

const licenseBtn = document.getElementById('licenseBtn');
const licenseModal = document.getElementById('licenseModal');
const closeLicense = document.getElementById('closeLicense');
const licenseStatusText = document.getElementById('licenseStatusText');
const licenseMessage = document.getElementById('licenseMessage');
const licenseInput = document.getElementById('licenseInput');
const licenseSaveBtn = document.getElementById('licenseSaveBtn');
const machineIdText = document.getElementById('machineIdText');
const copyMachineBtn = document.getElementById('copyMachineBtn');
const licenseHeaderHint = document.getElementById('licenseHeaderHint');
const scrollToTopBtn = document.getElementById('scrollToTopBtn');

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null; // File | null
let wsRetryTimer = null;
let indexFinished = false; // Guard to prevent progress bar re-showing after finish
let currentScanDirs = [];
let licenseAllowsSearch = false;
let isLocal = false;
let pathMappings = {};

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

// 兼容非安全上下文（HTTP 局域网）：navigator.clipboard 仅在 https/localhost 可用
function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  return new Promise((resolve, reject) => {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    ta.style.top = '0';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, ta.value.length);
    try {
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      if (ok) resolve();
      else reject(new Error('execCommand copy 返回 false'));
    } catch (e) {
      document.body.removeChild(ta);
      reject(e);
    }
  });
}

function loadPathMappings() {
  try {
    const stored = localStorage.getItem('pathMappings');
    if (stored) {
      pathMappings = JSON.parse(stored);
    }
  } catch (e) {
    console.error('Failed to load path mappings from localStorage', e);
    pathMappings = {};
  }
}

function getMappedPath(originalPath) {
  if (isLocal || Object.keys(pathMappings).length === 0) {
    return originalPath;
  }
  for (const serverPath of Object.keys(pathMappings)) {
    if (originalPath.startsWith(serverPath)) {
      const mappedDrive = pathMappings[serverPath];
      // 确保原始路径和映射驱动器的斜杠一致
      const serverPathWithSlash = serverPath.endsWith('\\') || serverPath.endsWith('/') ? serverPath : serverPath + '\\';
      const mappedDriveWithSlash = mappedDrive.endsWith('\\') || mappedDrive.endsWith('/') ? mappedDrive : mappedDrive + '\\';
      
      // 替换时，我们假设服务器和客户端都使用 `\` 作为分隔符，或者都是 `/`
      // Rust 服务端总是返回 `\` 分隔的路径
      return mappedDriveWithSlash + originalPath.substring(serverPathWithSlash.length);
    }
  }
  return originalPath;
}


// ── Image selection ───────────────────────────────────────────────────────────

// 上传前在浏览器端把过大的图片缩到长边 UPLOAD_MAX_EDGE。多数手机照片/高分屏
// 截图都远超搜索所需的分辨率（后端 letterbox 到 224、OCR 检测上限 1280），
// 客户端预缩既省带宽，也避免触碰后端上传上限导致的失败。缩放失败时回退到
// 原文件，由后端兜底处理。
const UPLOAD_MAX_EDGE = 2048;

async function downscaleForUpload(file) {
  try {
    const bitmap = await createImageBitmap(file);
    const longEdge = Math.max(bitmap.width, bitmap.height);
    if (longEdge <= UPLOAD_MAX_EDGE) {
      bitmap.close?.();
      return file; // 已足够小，原样上传
    }
    const scale = UPLOAD_MAX_EDGE / longEdge;
    const w = Math.round(bitmap.width * scale);
    const h = Math.round(bitmap.height * scale);
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(bitmap, 0, 0, w, h);
    bitmap.close?.();
    const blob = await new Promise((res) =>
      canvas.toBlob(res, 'image/jpeg', 0.9)
    );
    if (!blob) return file;
    return new File([blob], 'query.jpg', { type: 'image/jpeg' });
  } catch (e) {
    console.warn('客户端压缩失败，改为上传原图', e);
    return file;
  }
}

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
 searchBtn.disabled = !licenseAllowsSearch;
 window.scrollTo({ top: 0, behavior: 'smooth' });
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

// ── Drag & Drop ───────────────────────────────────────────────────────────────
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

// ── Clipboard paste ──────────────────────────────────────────────────────────
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

// ── Search ───────────────────────────────────────────────────────────────────
searchBtn.addEventListener('click', runSearch);

async function runSearch() {
 if (!selectedFile) return;
 if (!licenseAllowsSearch) {
   alert('当前许可状态不允许搜索，请先输入有效许可。');
   return;
 }
 const topK = parseInt(topKSelect.value, 10);

  // Show inline loading indicator and disable button — page remains interactive
  loadingOverlay.hidden = false;
  searchBtn.disabled = true;
  resultsSection.hidden = true;

  try {
    const form = new FormData();
    form.append('image', await downscaleForUpload(selectedFile));
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
    copyText(r.filename).then(() => {
      filenameEl.textContent = '✓ 已复制';
      setTimeout(() => {
        filenameEl.textContent = r.filename;
      }, 1500);
    }).catch((e) => {
      alert('复制失败：' + e.message);
    });
  });

  const pathEl = card.querySelector('.card-path');
  const mappedPath = getMappedPath(r.file_path);
  pathEl.textContent = mappedPath;
  pathEl.title = '点击复制路径';
  pathEl.addEventListener('click', () => {
    copyText(mappedPath).then(() => {
      pathEl.textContent = '✓ 已复制';
      setTimeout(() => {
        pathEl.textContent = mappedPath;
      }, 1500);
    }).catch((e) => {
      alert('复制失败：' + e.message);
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
    
    isLocal = data.is_local;
    currentScanDirs = data.scan_dirs || [];
    settingsBtn.hidden = false; // Always show settings button
    loadPathMappings();

    if (isLocal) {
      settingsTitle.textContent = '设置扫描目录';
      settingsHint.textContent = '这些目录将被扫描以查找 PDF 和 AI 文件。目前仅支持本地服务器路径。';
      addDirBtn.hidden = false;
    } else {
      settingsTitle.textContent = '设置路径映射';
      settingsHint.textContent = '将服务器扫描目录映射到您电脑上的盘符，方便快速访问文件。';
      addDirBtn.hidden = true;
    }

    renderDirList();

    if (isLocal && currentScanDirs.length === 0) {
      console.log('No scan directories configured, showing settings modal automatically');
      settingsModal.hidden = false;
    } else {
      settingsModal.hidden = true;
    }
  } catch (e) { console.error('Failed to fetch config', e); }
}

function renderDirList(dirs) {
  const list = dirs || currentScanDirs;
  dirList.innerHTML = '';

  if (isLocal) {
    list.forEach((dir, index) => {
      const item = document.createElement('div');
      item.className = 'dir-item';
      item.innerHTML = `
        <input type="text" class="dir-input" value="${dir}" data-index="${index}" placeholder="例如: C:\\Users\\Documents" />
        <button class="remove-btn" data-index="${index}" title="移除">✕</button>
      `;
      dirList.appendChild(item);
    });
  } else {
    // LAN user: show mapping UI
    list.forEach((dir) => {
      if (!dir) return;
      const item = document.createElement('div');
      item.className = 'dir-item-map';
      const mappedValue = pathMappings[dir] || '';
      item.innerHTML = `
        <div class="server-path-wrap">
          <label>服务器路径</label>
          <span class="server-path">${dir}</span>
        </div>
        <div class="arrow">→</div>
        <div class="local-path-wrap">
          <label>本地映射路径</label>
          <input type="text" class="dir-input-map" value="${mappedValue}" data-server-path="${dir}" placeholder="例如: Z:\\" />
        </div>
      `;
      dirList.appendChild(item);
    });
  }
}

function getEditingDirs() {
  if (isLocal) {
    const inputs = dirList.querySelectorAll('.dir-input');
    return Array.from(inputs).map(i => i.value.trim());
  }
  // For LAN users, this function returns the current mappings from the UI
  const newMappings = {};
  const inputs = dirList.querySelectorAll('.dir-input-map');
  inputs.forEach(input => {
    const serverPath = input.dataset.serverPath;
    const localPath = input.value.trim();
    if (serverPath && localPath) {
      newMappings[serverPath] = localPath;
    }
  });
  return newMappings;
}

function hasUnsavedChanges() {
  if (isLocal) {
    const editing = getEditingDirs().filter(d => d !== '');
    const saved = currentScanDirs.filter(d => d !== '');
    if (editing.length !== saved.length) return true;
    return editing.some((d, i) => d !== saved[i]);
  } else {
    const editingMappings = getEditingDirs();
    // Simple check: compare stringified versions
    return JSON.stringify(editingMappings) !== JSON.stringify(pathMappings);
  }
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
  if (!isLocal) return;
  const editing = getEditingDirs();
  editing.push('');
  renderDirList(editing);
});

dirList.addEventListener('click', (e) => {
  if (isLocal && e.target.classList.contains('remove-btn')) {
    const index = parseInt(e.target.dataset.index, 10);
    const editing = getEditingDirs();
    editing.splice(index, 1);
    renderDirList(editing);
  }
});

saveSettings.addEventListener('click', async () => {
  saveSettings.disabled = true;
  saveStatus.textContent = '保存中...';
  saveStatus.style.color = 'var(--text-muted)';

  if (isLocal) {
    // Local user: save scan directories to server
    const inputs = dirList.querySelectorAll('.dir-input');
    const dirs = Array.from(inputs).map(i => i.value.trim()).filter(d => d !== '');
    try {
      const resp = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_dirs: dirs })
      });
      if (!resp.ok) throw new Error(resp.statusText);

      saveStatus.textContent = '保存成功！正在启动索引...';
      saveStatus.style.color = 'var(--green)';
      currentScanDirs = dirs;

      setTimeout(() => { settingsModal.hidden = true; }, 1500);
      setTimeout(pollStatus, 500);
    } catch (e) {
      saveStatus.textContent = '保存失败: ' + e.message;
      saveStatus.style.color = 'var(--red)';
    } finally {
      saveSettings.disabled = false;
    }
  } else {
    // LAN user: save path mappings to localStorage
    try {
      const newMappings = {};
      const inputs = dirList.querySelectorAll('.dir-input-map');
      inputs.forEach(input => {
        const serverPath = input.dataset.serverPath;
        let localPath = input.value.trim();
        if (serverPath && localPath) {
          // 自动处理末尾的斜杠
          const serverEndsWithSlash = serverPath.endsWith('\\') || serverPath.endsWith('/');
          const localEndsWithSlash = localPath.endsWith('\\') || localPath.endsWith('/');
          if (serverEndsWithSlash && !localEndsWithSlash) {
            localPath += '\\';
          } else if (!serverEndsWithSlash && localEndsWithSlash) {
            localPath = localPath.slice(0, -1);
          }
          newMappings[serverPath] = localPath;
        }
      });
      
      localStorage.setItem('pathMappings', JSON.stringify(newMappings));
      pathMappings = newMappings;

      saveStatus.textContent = '映射已保存！';
      saveStatus.style.color = 'var(--green)';
      
      setTimeout(() => { settingsModal.hidden = true; }, 1500);
    } catch (e) {
      saveStatus.textContent = '保存失败: ' + e.message;
      saveStatus.style.color = 'var(--red)';
    } finally {
      saveSettings.disabled = false;
    }
  }
});

function formatLicenseStatusLabel(status, expiresAt) {
  if (status === 'trial') return '试用';
  if (status === 'permanent') return '永久';
  if (status === 'valid') {
    if (!expiresAt) return '有效';
    const d = new Date(expiresAt);
    if (Number.isNaN(d.getTime())) return '有效';
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return `有效期至 ${yyyy}-${mm}-${dd}`;
  }
  return '失效';
}

function applyLicenseStatus(data) {
  const status = data.status || 'invalid';
  const label = formatLicenseStatusLabel(status, data.expires_at);
  licenseStatusText.textContent = label;
  licenseStatusText.classList.remove('trial', 'valid', 'expired');
  if (status === 'trial') {
    licenseStatusText.classList.add('trial');
  } else if (status === 'valid' || status === 'permanent') {
    licenseStatusText.classList.add('valid');
  } else {
    licenseStatusText.classList.add('expired');
  }

  if (status === 'trial') {
    licenseHeaderHint.textContent = '试用中，请尽快激活许可';
    licenseHeaderHint.classList.remove('hidden', 'expired');
    licenseHeaderHint.classList.add('trial');
  } else if (status === 'expired' || status === 'invalid') {
    licenseHeaderHint.textContent = '试用已结束，请购买许可';
    licenseHeaderHint.classList.remove('hidden', 'trial');
    licenseHeaderHint.classList.add('expired');
  } else {
    licenseHeaderHint.textContent = '';
    licenseHeaderHint.classList.add('hidden');
    licenseHeaderHint.classList.remove('trial', 'expired');
  }

  licenseMessage.textContent = data.message || '';
  machineIdText.textContent = data.machine_id || 'UNKNOWN-MACHINE-ID';
  licenseAllowsSearch = !!data.search_allowed;
  searchBtn.disabled = !(selectedFile && licenseAllowsSearch);
}

async function refreshLicenseStatus() {
 try {
   const resp = await fetch('/api/license');
   if (!resp.ok) throw new Error(resp.statusText);
   const data = await resp.json();
   applyLicenseStatus(data);
 } catch (e) {
   licenseStatusText.textContent = '失效';
   licenseStatusText.classList.remove('trial', 'valid');
   licenseStatusText.classList.add('expired');
   licenseHeaderHint.textContent = '许可状态异常，请检查后激活';
   licenseHeaderHint.classList.remove('hidden', 'trial');
   licenseHeaderHint.classList.add('expired');
   licenseMessage.textContent = '许可状态获取失败: ' + e.message;
   licenseAllowsSearch = false;
   searchBtn.disabled = true;
 }
}

copyMachineBtn.addEventListener('click', async () => {
 const text = machineIdText.textContent || '';
 if (!text) return;
 try {
   await copyText(text);
   const old = copyMachineBtn.textContent;
   copyMachineBtn.textContent = '已复制';
   setTimeout(() => {
     copyMachineBtn.textContent = old || '复制';
   }, 1200);
 } catch (e) {
   alert('复制失败：' + e.message);
 }
});

licenseBtn.addEventListener('click', () => {
  licenseModal.hidden = false;
});

closeLicense.addEventListener('click', () => {
  licenseModal.hidden = true;
});

licenseSaveBtn.addEventListener('click', async () => {
 const key = (licenseInput.value || '').trim();
 licenseSaveBtn.disabled = true;
 try {
   const resp = await fetch('/api/license', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ license_key: key }),
   });
   if (!resp.ok) {
     const err = await resp.json().catch(() => ({ error: resp.statusText }));
     throw new Error(err.error || resp.statusText);
   }
   const data = await resp.json();
   applyLicenseStatus(data);
 } catch (e) {
   alert('许可保存失败：' + e.message);
 } finally {
   licenseSaveBtn.disabled = false;
 }
});

// ── Scroll to top button logic ───────────────────────────────────────────────
window.addEventListener('scroll', () => {
  if (window.scrollY > 200) {
    scrollToTopBtn.hidden = false;
    scrollToTopBtn.classList.remove('hidden');
  } else {
    scrollToTopBtn.classList.add('hidden');
    // Use a timeout to allow the fade-out animation to complete before setting hidden
    setTimeout(() => {
        if (window.scrollY <= 200) { // Re-check in case user scrolled back down
            scrollToTopBtn.hidden = true;
        }
    }, 200);
  }
});

scrollToTopBtn.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

initSettings();
refreshLicenseStatus();