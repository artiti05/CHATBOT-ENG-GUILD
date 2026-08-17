from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

ADMIN_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>لوحة إدارة قاعدة المعرفة</title>
<style>
  :root { --bg:#f4f6f8; --card:#fff; --accent:#0f6e4f; --danger:#b3261e; --muted:#667085; }
  * { box-sizing: border-box; font-family: 'Segoe UI', Tahoma, sans-serif; }
  body { margin:0; background:var(--bg); color:#1a1f27; }
  .wrap { max-width: 880px; margin: 24px auto; padding: 0 16px; }
  h1 { font-size: 1.3rem; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: .85rem; margin-bottom: 18px; }
  .card { background:var(--card); border-radius:12px; padding:18px; margin-bottom:16px;
          box-shadow:0 1px 3px rgba(16,24,40,.08); }
  label { display:block; font-size:.8rem; color:var(--muted); margin-bottom:6px; }
  input[type=password], input[type=file] { width:100%; padding:9px 10px; border:1px solid #d0d5dd;
          border-radius:8px; font-size:.9rem; }
  button { border:0; border-radius:8px; padding:9px 16px; font-size:.85rem; cursor:pointer;
           background:var(--accent); color:#fff; }
  button.secondary { background:#eef2f6; color:#1a1f27; }
  button.danger { background:var(--danger); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  th, td { text-align:right; padding:8px 6px; border-bottom:1px solid #eef0f3; }
  th { color:var(--muted); font-weight:600; }
  .row-actions { white-space:nowrap; }
  #status { font-size:.82rem; margin-top:10px; min-height:1.2em; }
  #status.ok { color:var(--accent); } #status.err { color:var(--danger); }
  .flex { display:flex; gap:10px; align-items:end; flex-wrap:wrap; }
  .flex > div { flex:1; min-width:220px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>لوحة إدارة قاعدة المعرفة</h1>
  <div class="sub">إضافة وحذف وإعادة معالجة وثائق نظام الأسئلة والأجوبة</div>

  <div class="card">
    <div class="flex">
      <div>
        <label for="adminKey">مفتاح الإدارة (X-API-Key)</label>
        <input type="password" id="adminKey" placeholder="أدخل مفتاح ADMIN_API_KEY">
      </div>
      <button class="secondary" onclick="saveKey()">حفظ المفتاح</button>
      <button onclick="loadDocs()">تحديث القائمة</button>
    </div>
  </div>

  <div class="card">
    <label>رفع وثيقة جديدة (PDF / TXT / MD)</label>
    <div class="flex">
      <div><input type="file" id="fileInput" accept=".pdf,.txt,.md"></div>
      <button id="uploadBtn" onclick="uploadFile()">رفع ومعالجة</button>
    </div>
  </div>

  <div class="card">
    <table>
      <thead><tr><th>الوثيقة</th><th>الحالة</th><th>الأجزاء</th><th class="row-actions">إجراءات</th></tr></thead>
      <tbody id="docsBody"><tr><td colspan="4">اضغط "تحديث القائمة" للبدء</td></tr></tbody>
    </table>
    <div id="status"></div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const keyInput = $('adminKey');
keyInput.value = localStorage.getItem('admin_api_key') || '';

function saveKey() {
  localStorage.setItem('admin_api_key', keyInput.value.trim());
  setStatus('تم حفظ المفتاح في هذا المتصفح.', 'ok');
}
function hdrs(extra) {
  return Object.assign({ 'X-API-Key': keyInput.value.trim() }, extra || {});
}
function setStatus(msg, cls) {
  const s = $('status'); s.textContent = msg; s.className = cls || '';
}
async function guard(res) {
  if (res.status === 401 || res.status === 403) {
    setStatus('مفتاح الإدارة غير صحيح أو مفقود.', 'err');
    throw new Error('auth');
  }
  if (!res.ok) {
    let d = ''; try { d = (await res.json()).detail || ''; } catch (e) {}
    setStatus('خطأ: ' + (d || res.status), 'err');
    throw new Error('http ' + res.status);
  }
  return res.json();
}

async function loadDocs() {
  setStatus('جاري التحميل...');
  try {
    const data = await guard(await fetch('/api/admin/files', { headers: hdrs() }));
    const body = $('docsBody');
    body.innerHTML = '';
    if (!data.documents || data.documents.length === 0) {
      body.innerHTML = '<tr><td colspan="4">لا توجد وثائق في قاعدة المعرفة.</td></tr>';
    } else {
      for (const d of data.documents) {
        const tr = document.createElement('tr');
        const name = d.source_id || d.filename || '—';
        tr.innerHTML =
          '<td>' + escapeHtml(name) + '</td>' +
          '<td>' + escapeHtml(String(d.status ?? '—')) + '</td>' +
          '<td>' + escapeHtml(String(d.chunk_count ?? d.chunks ?? '—')) + '</td>' +
          '<td class="row-actions">' +
            '<button class="secondary" onclick="reingestOne(\\'' + encodeURIComponent(name) + '\\')">إعادة معالجة</button> ' +
            '<button class="danger" onclick="deleteDoc(\\'' + encodeURIComponent(name) + '\\')">حذف</button>' +
          '</td>';
        body.appendChild(tr);
      }
    }
    setStatus('عدد الوثائق: ' + (data.count ?? 0), 'ok');
  } catch (e) { /* status already set */ }
}

async function uploadFile() {
  const f = $('fileInput').files[0];
  if (!f) { setStatus('اختر ملفاً أولاً.', 'err'); return; }
  $('uploadBtn').disabled = true;
  setStatus('جاري الرفع... ستبدأ المعالجة في الخلفية بعد الرفع.');
  try {
    const fd = new FormData();
    fd.append('file', f);
    const data = await guard(await fetch('/api/admin/files/upload', {
      method: 'POST', headers: hdrs(), body: fd
    }));
    setStatus(data.message || 'بدأت المعالجة في الخلفية. حدّث القائمة بعد قليل.', 'ok');
    $('fileInput').value = '';
  } catch (e) { /* status already set */ }
  $('uploadBtn').disabled = false;
}

async function deleteDoc(encName) {
  const name = decodeURIComponent(encName);
  if (!confirm('حذف "' + name + '" من قاعدة المعرفة؟')) return;
  try {
    const data = await guard(await fetch('/api/admin/files/' + encName, {
      method: 'DELETE', headers: hdrs()
    }));
    setStatus(data.message || 'تم الحذف.', 'ok');
    loadDocs();
  } catch (e) { /* status already set */ }
}

async function reingestOne(encName) {
  const name = decodeURIComponent(encName);
  try {
    const data = await guard(await fetch('/api/admin/files/re-ingest', {
      method: 'POST',
      headers: hdrs({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ files: [name], reingest_all: false })
    }));
    setStatus('أُعيدت جدولة المعالجة (' + (data.files_queued ?? 0) + ' ملف).', 'ok');
  } catch (e) { /* status already set */ }
}

function escapeHtml(t) {
  const d = document.createElement('div'); d.textContent = t; return d.innerHTML;
}
</script>
</body>
</html>"""


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_panel():
    return HTMLResponse(content=ADMIN_HTML)
