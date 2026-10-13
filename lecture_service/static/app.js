"use strict";

const $ = (id) => document.getElementById(id);
const displayText = (value) => String(value ?? "").replace(/[\u2012-\u2015\u2212\u2e3a\u2e3b]/g, "-");

function withoutFinalPeriod(text) {
  const ending = text.match(/\.(["'»”’)\]]*(?:\*\*|__|\*|_)?[ \t\r\n]*)$/u);
  if (!ending) return text;
  const prefix = text.slice(0, ending.index);
  // Keep ellipses, standalone list numbers and URL tokens intact.
  if (prefix.endsWith(".") || /^\s*\d+$/.test(prefix) || /(?:https?:\/\/|www\.)\S*$/i.test(prefix)) return text;
  return prefix + ending[1];
}

function displayParagraphs(value) {
  return displayText(value).split(/(\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*)/).map((part, index) => index % 2 ? part : withoutFinalPeriod(part)).join("");
}
const statusNames = {queued: "В очереди", processing: "Обрабатывается", completed: "Готово", failed: "Ошибка"};
const stageNames = {queued: "Ожидает обработки", preparing: "Готовим запись", downloading: "Скачиваем аудиодорожку", loading: "Загружаем запись", normalizing: "Подготавливаем аудио", preprocessing: "Подготавливаем аудио", preparing_audio: "Подготавливаем аудио", converting: "Подготавливаем аудио", transcribing: "Распознаём речь", loading_whisper: "Загружаем модель распознавания", loading_qwen: "Загружаем языковую модель", loading_summarizer: "Загружаем модель конспектирования", postprocessing: "Оформляем расшифровку", cleaning: "Очищаем расшифровку", summarizing: "Собираем конспект", saving: "Сохраняем результат", completed: "Готово", failed: "Обработка прервана"};
let selectedFile = null;
let previewUrl = null;
let activeId = null;
let activeResult = null;
let currentTab = "summary";
let pollTimer = null;
let toastTimer = null;
let jobs = [];
let uploading = false;
let sourceMode = "file";
let dialogJob = null;
let dialogAction = null;
let dialogBusy = false;

async function request(url, options) {
  const response = await fetch(url, options);
  if (response.status === 204 && response.ok) return null;
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail ?? body?.error;
    const message = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((item) => item.msg || JSON.stringify(item)).join("; ") : `Сервис вернул ошибку ${response.status}.`;
    throw new Error(message);
  }
  return body;
}

function notice(message = "") {
  $("notice").textContent = displayParagraphs(message);
  $("notice").hidden = !message;
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = displayParagraphs(message);
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3000);
}

function selectFile(file) {
  if (!file || uploading) return;
  selectedFile = file;
  $("file-label").textContent = displayText(file.name);
  $("file-hint").textContent = `${(file.size / 1024 / 1024).toLocaleString("ru-RU", {maximumFractionDigits: 1})} МБ · нажмите, чтобы выбрать другой файл`;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $("audio-preview").src = previewUrl;
  $("audio-preview").hidden = false;
  notice();
}

function switchSource(mode) {
  if (uploading) return;
  sourceMode = mode;
  for (const button of document.querySelectorAll("[data-source]")) {
    const selected = button.dataset.source === mode;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    $(`${button.dataset.source}-source`).hidden = !selected;
  }
  $("audio-file").required = mode === "file";
  $("audio-file").disabled = mode !== "file";
  $("youtube-url").required = mode === "youtube";
  $("youtube-url").disabled = mode !== "youtube";
  $("youtube-url").setCustomValidity("");
  if (mode !== "file") $("audio-preview").pause();
  $("upload-note").textContent = mode === "youtube" ? "Аудиодорожка скачивается на сервере. Ход загрузки и обработки появится ниже" : "Время обработки зависит от длины записи и мощности устройства. Прогресс появится ниже";
  notice();
}

function validYouTubeUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && !url.username && !url.password && (!url.port || url.port === "443") && ["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"].includes(url.hostname);
  } catch { return false; }
}

function prettyDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "" : date.toLocaleDateString("ru-RU", {day: "numeric", month: "short"});
}

function duration(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  const h = Math.floor(seconds / 3600);
  return `${h ? `${h}:` : ""}${String(Math.floor(seconds / 60) % 60).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function renderHistory() {
  $("history").replaceChildren();
  $("history-empty").hidden = jobs.length > 0;
  for (const job of jobs) {
    const li = document.createElement("li");
    li.className = "history-item";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-open${job.id === activeId ? " active" : ""}`;
    button.dataset.jobId = job.id;
    if (job.id === activeId) button.setAttribute("aria-current", "true");
    const name = document.createElement("span");
    name.className = "history-name";
    name.textContent = displayText(job.filename);
    button.title = displayText(job.filename);
    const detail = document.createElement("span");
    detail.className = "history-detail";
    const dot = document.createElement("span");
    dot.className = `history-dot ${job.status}`;
    dot.setAttribute("aria-hidden", "true");
    detail.append(dot, document.createTextNode(displayText(`${statusNames[job.status] || job.status} · ${prettyDate(job.created_at)}`)));
    button.append(name, detail);
    button.addEventListener("click", () => openJob(job.id));
    li.append(button);
    if (["completed", "failed"].includes(job.status)) {
      const actions = document.createElement("div");
      actions.className = "history-actions";
      for (const [action, label] of [["rename", "Переименовать"], ["delete", "Удалить"]]) {
        const control = document.createElement("button");
        control.type = "button";
        control.className = `history-action history-${action}`;
        control.dataset.historyAction = action;
        control.dataset.jobId = job.id;
        control.textContent = label;
        control.setAttribute("aria-label", displayText(`${label}: ${job.filename}`));
        control.addEventListener("click", () => showLectureDialog(job, action));
        actions.append(control);
      }
      li.append(actions);
    }
    $("history").append(li);
  }
}

function showLectureDialog(job, action) {
  if (dialogBusy || !["completed", "failed"].includes(job.status)) return;
  dialogJob = job;
  dialogAction = action;
  const deleting = action === "delete";
  $("lecture-dialog-title").textContent = deleting ? "Удалить лекцию?" : "Переименовать лекцию";
  $("lecture-dialog-description").textContent = displayParagraphs(deleting ? `Лекция «${job.filename}», её расшифровка, конспект и аудиозапись будут удалены. Отменить удаление нельзя.` : "Новое название появится в истории и результатах. Аудиозапись сохранится.");
  $("rename-field").hidden = deleting;
  $("lecture-name").disabled = deleting;
  $("lecture-name").required = !deleting;
  $("lecture-name").value = job.filename;
  $("lecture-name").setCustomValidity("");
  $("lecture-dialog-error").textContent = "";
  $("lecture-dialog-error").hidden = true;
  $("lecture-dialog-submit").textContent = deleting ? "Удалить лекцию" : "Сохранить";
  $("lecture-dialog-submit").classList.toggle("danger-button", deleting);
  $("lecture-dialog").showModal();
  if (deleting) $("lecture-dialog-cancel").focus();
  else { $("lecture-name").focus(); $("lecture-name").select(); }
}

function closeLectureDialog() {
  if (dialogBusy) return;
  const previousJob = dialogJob;
  const previousAction = dialogAction;
  $("lecture-dialog").close();
  dialogJob = null;
  dialogAction = null;
  const control = [...document.querySelectorAll("[data-history-action]")].find((button) => button.dataset.jobId === previousJob?.id && button.dataset.historyAction === previousAction);
  (control || document.querySelector(".history-open") || $("refresh-history")).focus();
}

$("lecture-dialog-cancel").addEventListener("click", closeLectureDialog);
$("lecture-dialog").addEventListener("cancel", (event) => { event.preventDefault(); closeLectureDialog(); });
$("lecture-name").addEventListener("input", () => $("lecture-name").setCustomValidity(""));
$("lecture-dialog-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (dialogBusy || !dialogJob) return;
  const jobId = dialogJob.id;
  const deleting = dialogAction === "delete";
  const filename = $("lecture-name").value.trim();
  if (!deleting && (!filename || [...filename].length > 255 || /[\/\\\p{Cc}\p{Cs}\p{Zl}\p{Zp}]/u.test(filename) || /^[a-z]:/i.test(filename) || [".", ".."].includes(filename))) {
    $("lecture-name").setCustomValidity("Введите название от 1 до 255 символов без слешей и управляющих символов");
    $("lecture-name").reportValidity();
    return;
  }
  dialogBusy = true;
  $("lecture-dialog-submit").disabled = true;
  $("lecture-dialog-cancel").disabled = true;
  $("lecture-name").disabled = true;
  $("lecture-dialog-error").hidden = true;
  try {
    const updated = await request(`/api/lectures/${encodeURIComponent(jobId)}`, deleting ? {method: "DELETE"} : {method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({filename})});
    if (deleting) {
      jobs = jobs.filter((job) => job.id !== jobId);
      if (activeId === jobId) {
        clearTimeout(pollTimer);
        activeId = null;
        activeResult = null;
        $("job-panel").hidden = true;
        $("result-content").hidden = true;
        try { localStorage.removeItem("lectorium.activeId"); } catch { /* Optional preference. */ }
        if (jobs.length) openJob(jobs[0].id);
      }
      renderHistory();
    } else {
      jobs = jobs.map((job) => job.id === jobId ? updated : job);
      if (activeId === jobId) renderJob(updated);
      renderHistory();
    }
    dialogBusy = false;
    closeLectureDialog();
    toast(deleting ? "Лекция удалена" : "Название сохранено");
  } catch (error) {
    $("lecture-dialog-error").textContent = displayParagraphs(`${deleting ? "Не удалось удалить лекцию" : "Не удалось изменить название"}: ${error.message}`);
    $("lecture-dialog-error").hidden = false;
  } finally {
    dialogBusy = false;
    $("lecture-dialog-submit").disabled = false;
    $("lecture-dialog-cancel").disabled = false;
    $("lecture-name").disabled = deleting;
  }
});

async function loadHistory() {
  const data = await request("/api/lectures");
  jobs = Array.isArray(data) ? data : data.items || [];
  renderHistory();
}

function renderJob(job) {
  $("job-panel").hidden = false;
  $("job-title").textContent = displayText(job.filename);
  $("job-meta").textContent = [prettyDate(job.created_at), job.duration_seconds ? `${duration(job.duration_seconds)} записи` : "", job.language === "ru" ? "Русский" : job.language === "en" ? "Английский" : "Автоопределение языка"].filter(Boolean).join(" · ");
  $("status-badge").textContent = displayText(statusNames[job.status] || job.status);
  $("status-badge").className = `status-badge ${job.status}`;
  $("progress-panel").hidden = !["queued", "processing"].includes(job.status);
  const progress = Math.max(0, Math.min(1, Number(job.progress) || 0));
  $("job-progress").value = progress;
  $("progress-label").textContent = `${Math.round(progress * 100)}%`;
  $("stage-label").textContent = displayText(stageNames[job.stage] || job.stage || statusNames[job.status]);
  $("job-error").hidden = job.status !== "failed";
  $("job-error").textContent = job.status === "failed" ? displayParagraphs(`Не удалось обработать запись. ${job.error || "Подробности доступны в журнале сервиса."}`) : "";
  $("result-content").hidden = job.status !== "completed" || !activeResult;
}

// Build DOM nodes from a small Markdown subset. Model output never becomes HTML.
function inlineText(node, text) {
  const pattern = /(\*\*([^*]+)\*\*|`([^`]+)`)/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    node.append(document.createTextNode(text.slice(last, match.index)));
    const element = document.createElement(match[2] ? "strong" : "code");
    element.textContent = match[2] || match[3];
    node.append(element);
    last = match.index + match[0].length;
  }
  node.append(document.createTextNode(text.slice(last)));
}

function renderMarkdown(container, markdown) {
  container.replaceChildren();
  let list = null;
  let paragraph = null;
  let paragraphLines = [];
  let code = null;
  function finishParagraph() {
    if (!paragraph) return;
    inlineText(paragraph, withoutFinalPeriod(paragraphLines.join("\n")));
    paragraph = null;
    paragraphLines = [];
  }
  for (const line of displayText(markdown || "Конспект отсутствует.").split(/\r?\n/)) {
    if (line.trim().startsWith("```")) {
      finishParagraph();
      if (code) { code = null; } else {
        const pre = document.createElement("pre");
        code = document.createElement("code");
        pre.append(code);
        container.append(pre);
      }
      list = null; continue;
    }
    if (code) { code.append(document.createTextNode(`${line}\n`)); continue; }
    if (!line.trim()) { finishParagraph(); list = null; continue; }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    const bullet = line.match(/^\s*([-*+]\s+|\d+[.)]\s+)(.+)$/);
    if (heading) {
      finishParagraph();
      const h = document.createElement(`h${heading[1].length}`);
      inlineText(h, withoutFinalPeriod(heading[2])); container.append(h); list = null;
    } else if (bullet) {
      finishParagraph();
      const type = /^\d/.test(bullet[1]) ? "ol" : "ul";
      if (!list || list.tagName.toLowerCase() !== type) { list = document.createElement(type); container.append(list); }
      const li = document.createElement("li"); inlineText(li, withoutFinalPeriod(bullet[2])); list.append(li);
    } else {
      list = null;
      if (!paragraph) { paragraph = document.createElement("p"); container.append(paragraph); }
      paragraphLines.push(line);
    }
  }
  finishParagraph();
}

function renderResult(result) {
  const cleanupWarnings = Array.isArray(result.metadata?.cleanup_warnings) ? result.metadata.cleanup_warnings : [];
  const retainedCount = cleanupWarnings.length;
  const fragmentForm = retainedCount % 10 === 1 && retainedCount % 100 !== 11 ? "фрагменте" : "фрагментах";
  $("cleanup-note").hidden = retainedCount === 0;
  $("cleanup-note").textContent = retainedCount ? `В ${retainedCount} ${fragmentForm} сохранён исходный текст, поскольку правки модели не прошли проверку` : "";
  renderMarkdown($("pane-summary"), result.summary);
  $("pane-transcript").textContent = displayParagraphs(result.transcript || "Расшифровка отсутствует.");
  $("pane-raw").textContent = displayParagraphs(result.raw_transcript || result.transcript || "Исходный текст отсутствует.");
  $("pane-segments").replaceChildren();
  const segments = Array.isArray(result.segments) ? result.segments : [];
  if (!segments.length) $("pane-segments").textContent = "Временные метки для этой записи отсутствуют";
  for (const segment of segments) {
    const row = document.createElement("div"); row.className = "segment";
    const time = document.createElement("time");
    const start = segment.start ?? segment.timestamp?.[0];
    const end = segment.end ?? segment.timestamp?.[1];
    time.textContent = `${duration(start)} - ${duration(end ?? start)}`;
    const text = document.createElement("p"); text.textContent = displayParagraphs(segment.text || "");
    row.append(time, text); $("pane-segments").append(row);
  }
  for (const format of ["md", "txt", "json"]) $( `download-${format}`).href = `/api/lectures/${encodeURIComponent(activeId)}/download?format=${format}`;
  $("result-content").hidden = false;
  switchTab(currentTab);
}

function switchTab(tab) {
  currentTab = tab;
  for (const button of document.querySelectorAll("[data-tab]")) {
    const selected = button.dataset.tab === tab;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    $(`pane-${button.dataset.tab}`).hidden = !selected;
  }
}

async function pollJob(id) {
  clearTimeout(pollTimer);
  try {
    const job = await request(`/api/lectures/${encodeURIComponent(id)}`);
    if (id !== activeId) return;
    renderJob(job);
    const index = jobs.findIndex((entry) => entry.id === id);
    if (index >= 0) jobs[index] = job;
    renderHistory();
    if (job.status === "completed") {
      if (!activeResult) {
        const result = await request(`/api/lectures/${encodeURIComponent(id)}/result`);
        if (id !== activeId) return;
        activeResult = result;
        renderResult(result);
      }
      return;
    }
    if (job.status === "failed") return;
  } catch (error) {
    if (id !== activeId) return;
    notice(`Не удалось получить состояние лекции: ${error.message} Повторим автоматически.`);
  }
  if (id === activeId) pollTimer = setTimeout(() => pollJob(id), 2500);
}

function openJob(id) {
  clearTimeout(pollTimer);
  activeId = id;
  activeResult = null;
  currentTab = "summary";
  $("result-content").hidden = true;
  try { localStorage.setItem("lectorium.activeId", id); } catch { /* Storage may be disabled. */ }
  notice();
  const job = jobs.find((entry) => entry.id === id);
  if (job) renderJob(job);
  renderHistory();
  pollJob(id);
}

$("audio-file").addEventListener("change", (event) => selectFile(event.target.files[0]));
for (const name of ["dragenter", "dragover"]) $("dropzone").addEventListener(name, (event) => { event.preventDefault(); $("dropzone").classList.add("drag-over"); });
for (const name of ["dragleave", "drop"]) $("dropzone").addEventListener(name, (event) => { event.preventDefault(); $("dropzone").classList.remove("drag-over"); });
$("dropzone").addEventListener("drop", (event) => {
  if (uploading || sourceMode !== "file") return;
  const file = event.dataTransfer.files[0];
  if (file) {
    // Keep the native required input in sync so browser form validation accepts drops.
    const transfer = new DataTransfer(); transfer.items.add(file); $("audio-file").files = transfer.files;
    selectFile(file);
  }
});
$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (uploading || (sourceMode === "file" && !selectedFile)) return;
  const youtubeUrl = $("youtube-url").value.trim();
  if (sourceMode === "youtube" && !validYouTubeUrl(youtubeUrl)) {
    $("youtube-url").setCustomValidity("Введите HTTPS-ссылку на видео youtube.com или youtu.be");
    $("youtube-url").reportValidity();
    return;
  }
  uploading = true; $("submit-button").disabled = true; $("audio-file").disabled = true; $("youtube-url").disabled = true;
  for (const button of document.querySelectorAll("[data-source]")) button.disabled = true;
  $("submit-button").textContent = sourceMode === "youtube" ? "Добавляем ссылку…" : "Загружаем запись…";
  notice();
  let url = "/api/lectures";
  let options;
  if (sourceMode === "youtube") {
    url += "/youtube";
    options = {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({url: youtubeUrl, language: $("language").value === "auto" ? null : $("language").value})};
  } else {
    const data = new FormData(); data.append("file", selectedFile);
    if ($("language").value !== "auto") data.append("language", $("language").value);
    options = {method: "POST", body: data};
  }
  try {
    const job = await request(url, options);
    jobs.unshift(job); openJob(job.id);
    $("job-panel").scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "nearest"});
    loadHistory().catch(() => {});
  } catch (error) { notice(`${sourceMode === "youtube" ? "Не удалось добавить видео" : "Не удалось загрузить запись"}: ${error.message}`); }
  finally {
    uploading = false; $("submit-button").disabled = false; $("audio-file").disabled = sourceMode !== "file"; $("youtube-url").disabled = sourceMode !== "youtube";
    for (const button of document.querySelectorAll("[data-source]")) button.disabled = false;
    $("submit-button").textContent = "Создать конспект";
  }
});
$("youtube-url").addEventListener("input", () => $("youtube-url").setCustomValidity(""));
const sourceButtons = [...document.querySelectorAll("[data-source]")];
for (const button of sourceButtons) {
  button.addEventListener("click", () => switchSource(button.dataset.source));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const index = event.key === "Home" ? 0 : event.key === "End" ? sourceButtons.length - 1 : (sourceButtons.indexOf(button) + 1) % sourceButtons.length;
    switchSource(sourceButtons[index].dataset.source); sourceButtons[index].focus();
  });
}
$("refresh-history").addEventListener("click", () => loadHistory().then(() => toast("Список обновлён")).catch((error) => notice(error.message)));
const tabButtons = [...document.querySelectorAll("[data-tab]")];
for (const button of tabButtons) {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
  button.addEventListener("keydown", (event) => {
    let index = tabButtons.indexOf(button);
    if (event.key === "ArrowRight") index = (index + 1) % tabButtons.length;
    else if (event.key === "ArrowLeft") index = (index - 1 + tabButtons.length) % tabButtons.length;
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = tabButtons.length - 1;
    else return;
    event.preventDefault(); switchTab(tabButtons[index].dataset.tab); tabButtons[index].focus();
  });
}
$("copy-result").addEventListener("click", async () => {
  if (!activeResult) return;
  const text = currentTab === "summary" ? $("pane-summary").innerText : currentTab === "raw" ? $("pane-raw").textContent : currentTab === "segments" ? $("pane-segments").innerText : $("pane-transcript").textContent;
  try { await navigator.clipboard.writeText(displayText(text)); toast("Текст скопирован"); }
  catch { toast("Буфер обмена недоступен. Используйте скачивание TXT."); }
});
window.addEventListener("beforeunload", () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });

(async function init() {
  try {
    await loadHistory();
    let remembered = null;
    try { remembered = localStorage.getItem("lectorium.activeId"); } catch { /* Optional preference. */ }
    const selected = jobs.find((job) => job.id === remembered) || jobs[0];
    if (selected) openJob(selected.id);
  } catch (error) { notice(`Сервис пока недоступен: ${error.message} Проверьте, запущен ли сервер, и обновите страницу.`); }
})();
