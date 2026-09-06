const badge = document.getElementById("state-badge");
const liveText = document.getElementById("live-text");
const popupText = document.getElementById("popup-text");
const confidenceEl = document.getElementById("confidence");
const sessionIdEl = document.getElementById("session-id");
const tableBody = document.querySelector("#log-table tbody");
const overlay = document.getElementById("live-overlay");
const closeLive = document.getElementById("close-live");
const toast = document.getElementById("toast");
const toastText = document.getElementById("toast-text");

let activeSessionId = null;

function setState(state) {
  badge.innerHTML = `<span></span>${state}`;
  badge.className = "badge " + state.toLowerCase();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  })[c]);
}

function addLogRow(row) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${escapeHtml(row.timestamp)}</td>
    <td>${escapeHtml(row.verification_result)}</td>
    <td>${escapeHtml(row.confidence_score)}</td>
    <td>${escapeHtml(row.transcribed_text)}</td>
    <td>${escapeHtml(row.duration_ms)}</td>
    <td>${escapeHtml(row.termination_reason)}</td>
  `;
  tableBody.prepend(tr);
  while (tableBody.rows.length > 10) tableBody.deleteRow(-1);
}

function openLive(sessionId) {
  activeSessionId = sessionId;
  popupText.textContent = "Listening…";
  overlay.classList.add("show");
  overlay.setAttribute("aria-hidden", "false");
}

function closeLivePopup() {
  overlay.classList.remove("show");
  overlay.setAttribute("aria-hidden", "true");
}

function showToast(reason) {
  toastText.textContent = `Session completed (${reason || "ENDED"})`;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3500);
}

closeLive.addEventListener("click", closeLivePopup);

fetch("/api/recent_logs")
  .then(r => r.json())
  .then(rows => rows.reverse().forEach(addLogRow))
  .catch(console.error);

const protocol = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${protocol}://${location.host}/ws/dashboard`);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.event === "state") {
    setState(msg.state);
    sessionIdEl.textContent = msg.session_id || "--";

    if (msg.confidence !== undefined) {
      confidenceEl.textContent = Number(msg.confidence).toFixed(2);
    }

    if (msg.state === "LISTENING") {
      liveText.textContent = "Waiting for a verified voice session…";
    }

    if (msg.state === "STREAMING") {
      openLive(msg.session_id);
    }

    if (msg.state === "REJECTED") {
      closeLivePopup();
      liveText.textContent = "Session rejected.";
    }

    if (msg.state === "IDLE" && msg.session_id === activeSessionId) {
      closeLivePopup();
      showToast(msg.termination_reason);
      activeSessionId = null;
    }
  }

  if (msg.event === "partial_text") {
    if (!overlay.classList.contains("show")) openLive(msg.session_id);
    popupText.textContent = msg.text || "Listening…";
    liveText.textContent = msg.text || "Listening…";
  }

  if (msg.event === "final_text") {
    popupText.textContent = msg.text || "No speech detected";
    liveText.textContent = msg.text || "No speech detected";
  }

  if (msg.event === "log_row") {
    addLogRow(msg.row);
  }
};

ws.onopen = () => console.log("Connected to dashboard socket");
ws.onclose = () => console.log("Dashboard socket closed");
