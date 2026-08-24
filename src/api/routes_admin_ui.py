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
  #status, #ticketStatus { font-size:.82rem; margin-top:10px; min-height:1.2em; }
  #status.ok, #ticketStatus.ok { color:var(--accent); } #status.err, #ticketStatus.err { color:var(--danger); }
  .flex { display:flex; gap:10px; align-items:end; flex-wrap:wrap; }
  .flex > div { flex:1; min-width:220px; }
  .tabs { display:flex; gap:8px; margin-bottom:16px; }
  .tab-btn { background:#eef2f6; color:#1a1f27; border-radius:10px 10px 0 0; }
  .tab-btn.active { background:var(--accent); color:#fff; }
  select.status-select { padding:6px 8px; border-radius:6px; border:1px solid #d0d5dd; font-size:.82rem; }
  .badge { display:inline-block; padding:2px 8px; border-radius:20px; font-size:.75rem; font-weight:600; }
  .badge.open { background:#fde68a; color:#7c4a03; }
  .badge.in_progress { background:#bfdbfe; color:#1e3a8a; }
  .badge.resolved { background:#bbf7d0; color:#14532d; }
  .badge.closed { background:#e2e8f0; color:#475569; }
</style>
</head>
<body>
<div class="wrap">
  <h1>لوحة إدارة قاعدة المعرفة</h1>
  <div class="sub">إضافة وحذف وإعادة معالجة وثائق نظام الأسئلة والأجوبة، ومتابعة تذاكر الدعم</div>

  <div class="card">
    <div class="flex">
      <div>
        <label for="adminKey">مفتاح الإدارة (X-API-Key)</label>
        <input type="password" id="adminKey" placeholder="أدخل مفتاح ADMIN_API_KEY">
      </div>
      <button class="secondary" onclick="saveKey()">حفظ المفتاح</button>
    </div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" id="tabDocsBtn" onclick="switchTab('docs')">الوثائق</button>
    <button class="tab-btn" id="tabTicketsBtn" onclick="switchTab('tickets')">التذاكر</button>
  </div>

  <div id="docsView">
    <div class="card">
      <label>رفع وثيقة جديدة (PDF / TXT / MD)</label>
      <div class="flex">
        <div><input type="file" id="fileInput" accept=".pdf,.txt,.md"></div>
        <button id="uploadBtn" onclick="uploadFile()">رفع ومعالجة</button>
      </div>
    </div>

    <div class="card">
      <div class="flex">
        <div></div>
        <button onclick="loadDocs()">تحديث القائمة</button>
      </div>
      <table>
        <thead><tr><th>الوثيقة</th><th>الحالة</th><th>الأجزاء</th><th class="row-actions">إجراءات</th></tr></thead>
        <tbody id="docsBody"><tr><td colspan="4">اضغط "تحديث القائمة" للبدء</td></tr></tbody>
      </table>
      <div id="status"></div>
    </div>
  </div>

  <div id="ticketsView" style="display:none">
    <div class="card">
      <div class="flex">
        <div>
          <label for="ticketStatusFilter">تصفية حسب الحالة</label>
          <select id="ticketStatusFilter" class="status-select" onchange="loadTickets()">
            <option value="">الكل</option>
            <option value="open">مفتوحة</option>
            <option value="in_progress">قيد المعالجة</option>
            <option value="resolved">تم الحل</option>
            <option value="closed">مغلقة</option>
          </select>
        </div>
        <button onclick="loadTickets()">تحديث القائمة</button>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th><th>التاريخ</th><th>السبب</th><th>الاسم</th><th>الهاتف</th>
            <th>الرقم النقابي</th><th>تفاصيل الطلب</th><th>الحالة</th><th class="row-actions">إجراء</th>
          </tr>
        </thead>
        <tbody id="ticketsBody"><tr><td colspan="9">اضغط "تحديث القائمة" للبدء</td></tr></tbody>
      </table>
      <div id="ticketStatus"></div>
    </div>
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

function switchTab(name) {
  const isDocs = name === 'docs';
  $('docsView').style.display = isDocs ? '' : 'none';
  $('ticketsView').style.display = isDocs ? 'none' : '';
  $('tabDocsBtn').classList.toggle('active', isDocs);
  $('tabTicketsBtn').classList.toggle('active', !isDocs);
  if (!isDocs) loadTickets();
}

function setTicketStatus(msg, cls) {
  const s = $('ticketStatus'); s.textContent = msg; s.className = cls || '';
}

const REASON_LABELS = { user_intent: 'طلب مستخدم', low_confidence: 'ثقة منخفضة' };
const STATUS_LABELS = { open: 'مفتوحة', in_progress: 'قيد المعالجة', resolved: 'تم الحل', closed: 'مغلقة' };

async function loadTickets() {
  setTicketStatus('جاري التحميل...');
  try {
    const filter = $('ticketStatusFilter').value;
    const url = '/api/admin/tickets' + (filter ? ('?status=' + encodeURIComponent(filter)) : '');
    const data = await guard(await fetch(url, { headers: hdrs() }));
    const body = $('ticketsBody');
    body.innerHTML = '';
    if (!data.tickets || data.tickets.length === 0) {
      body.innerHTML = '<tr><td colspan="9">لا توجد تذاكر حالياً.</td></tr>';
    } else {
      for (const t of data.tickets) {
        const tr = document.createElement('tr');
        const statusOptions = Object.keys(STATUS_LABELS).map(function (s) {
          return '<option value="' + s + '"' + (s === t.status ? ' selected' : '') + '>' + STATUS_LABELS[s] + '</option>';
        }).join('');
        tr.innerHTML =
          '<td>' + t.id + '</td>' +
          '<td>' + escapeHtml(String(t.created_at || '—')) + '</td>' +
          '<td>' + escapeHtml(REASON_LABELS[t.reason] || t.reason) + '</td>' +
          '<td>' + escapeHtml(t.name || '—') + '</td>' +
          '<td>' + escapeHtml(t.phone || '—') + '</td>' +
          '<td>' + escapeHtml(t.engineer_number || '—') + '</td>' +
          '<td>' + escapeHtml((t.issue_text || '') + (t.raw_contact_text ? (' | ' + t.raw_contact_text) : '')) + '</td>' +
          '<td><span class="badge ' + t.status + '">' + (STATUS_LABELS[t.status] || t.status) + '</span></td>' +
          '<td class="row-actions">' +
            '<select class="status-select" onchange="updateTicketStatus(' + t.id + ', this.value)">' + statusOptions + '</select>' +
          '</td>';
        body.appendChild(tr);
      }
    }
    setTicketStatus('عدد التذاكر: ' + (data.count ?? 0), 'ok');
  } catch (e) { /* status already set */ }
}

async function updateTicketStatus(ticketId, newStatus) {
  try {
    const data = await guard(await fetch('/api/admin/tickets/' + ticketId + '/status', {
      method: 'POST',
      headers: hdrs({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ status: newStatus })
    }));
    setTicketStatus(data.message || 'تم تحديث الحالة.', 'ok');
    loadTickets();
  } catch (e) { /* status already set */ }
}
</script>
</body>
</html>"""


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_panel():
    return HTMLResponse(content=ADMIN_HTML)
