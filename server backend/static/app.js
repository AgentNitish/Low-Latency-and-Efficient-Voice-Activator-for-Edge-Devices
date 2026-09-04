const badge = document.getElementById("state-badge");
const liveText = document.getElementById("live-text");
const confidenceEl = document.getElementById("confidence");
const sessionIdEl = document.getElementById("session-id");
const tableBody = document.querySelector("#log-table tbody");

function setState(state) {
  badge.textContent = state;
  badge.className = "badge " + state.toLowerCase();
}

function addLogRow(row) {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${row.timestamp || ""}</td>
    <td>${row.verification_result || ""}</td>
    <td>${row.confidence_score ?? ""}</td>
    <td>${row.transcribed_text || ""}</td>
    <td>${row.duration_ms ?? ""}</td>
    <td>${row.termination_reason || ""}</td>
  `;
  tableBody.prepend(tr);
  while (tableBody.rows.length > 10) tableBody.deleteRow(-1);
}

// Load the last 10 sessions when the page opens
fetch("/api/recent_logs").then(r => r.json()).then(rows => rows.reverse().forEach(addLogRow));

// Connect to the live telemetry socket
const ws = new WebSocket(`ws://${location.host}/ws/dashboard`);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.event === "state") {
    setState(msg.state);
    sessionIdEl.textContent = msg.session_id || "--";
    if (msg.confidence !== undefined) confidenceEl.textContent = msg.confidence.toFixed(2);
    if (msg.state === "LISTENING") liveText.textContent = "Waiting for a session…";
  }

  if (msg.event === "partial_text") {
    liveText.textContent = msg.text;
  }

  if (msg.event === "final_text") {
    liveText.textContent = msg.text;
  }

  if (msg.event === "log_row") {
    addLogRow(msg.row);
  }
};

ws.onopen = () => console.log("Connected to dashboard socket");
ws.onclose = () => console.log("Dashboard socket closed - refresh page to reconnect");
