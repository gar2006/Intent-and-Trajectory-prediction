const chart = document.getElementById("trajectoryChart");
const sampleSelect = document.getElementById("sampleSelect");
const agentTypeBadge = document.getElementById("agentTypeBadge");
const sampleToken = document.getElementById("sampleToken");
const agentId = document.getElementById("agentId");
const summaryText = document.getElementById("summaryText");
const intentLabel = document.getElementById("intentLabel");
const intentBars = document.getElementById("intentBars");
const riskBadge = document.getElementById("riskBadge");
const riskScore = document.getElementById("riskScore");
const riskReasons = document.getElementById("riskReasons");
const modeCards = document.getElementById("modeCards");

const COLORS = {
  past: "#111111",
  future: "#239b56",
  mode1: "#cf3c3c",
  mode2: "#f39c12",
  mode3: "#2c6fb7",
};

const data = await fetch("./data/demo_data.json").then((response) => response.json());

for (const [index, record] of data.entries()) {
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = `Sample ${index + 1}: ${record.agent_type_label} · ${record.predicted_intent_label}`;
  sampleSelect.appendChild(option);
}

sampleSelect.addEventListener("change", () => {
  renderRecord(data[Number(sampleSelect.value)]);
});

renderRecord(data[0]);

function renderRecord(record) {
  agentTypeBadge.textContent = record.agent_type_label;
  agentTypeBadge.className = `agent-type-badge ${record.agent_type}`;
  sampleToken.textContent = `Sample token: ${record.sample_token}`;
  agentId.textContent = `Agent id: ${record.agent_id}`;
  summaryText.textContent = record.summary;
  intentLabel.textContent = record.predicted_intent_label;
  riskBadge.textContent = record.risk_label;
  riskBadge.className = `risk-badge ${record.risk_level}`;
  riskScore.textContent = `Risk score: ${Math.round(record.risk_score * 100)} / 100`;

  renderIntentBars(record.intent_probabilities);
  renderRiskReasons(record.risk_reasons);
  renderModeCards(record.ranked_modes);
  renderChart(record);
}

function renderIntentBars(intentProbabilities) {
  intentBars.innerHTML = "";
  for (const item of intentProbabilities) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${formatIntent(item.intent)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${item.probability * 100}%"></div></div>
      <strong>${Math.round(item.probability * 100)}%</strong>
    `;
    intentBars.appendChild(row);
  }
}

function renderRiskReasons(reasons) {
  riskReasons.innerHTML = "";
  reasons.forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reason;
    riskReasons.appendChild(item);
  });
}

function renderModeCards(modes) {
  modeCards.innerHTML = "";
  modes.forEach((mode, index) => {
    const card = document.createElement("div");
    card.className = "mode-card";
    const endpoint = mode.endpoint.map((value) => value.toFixed(1)).join(", ");
    card.innerHTML = `
      <h3>${mode.label}</h3>
      <p class="mode-meta">
        Endpoint estimate: (${endpoint}) meters<br />
        Confidence: ${Math.round(mode.probability * 100)}%
      </p>
    `;
    card.style.borderLeft = `6px solid ${getModeColor(index)}`;
    modeCards.appendChild(card);
  });
}

function renderChart(record) {
  chart.innerHTML = "";

  const width = 600;
  const height = 600;
  const padding = 48;
  const points = [
    ...record.history_xy,
    ...record.future_xy,
    ...record.ranked_modes.flatMap((mode) => mode.trajectory),
  ];

  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const xMin = Math.min(...xs, -10);
  const xMax = Math.max(...xs, 10);
  const yMin = Math.min(...ys, -10);
  const yMax = Math.max(...ys, 10);
  const xSpan = Math.max(xMax - xMin, 1);
  const ySpan = Math.max(yMax - yMin, 1);
  const span = Math.max(xSpan, ySpan) * 1.15;
  const centerX = (xMin + xMax) / 2;
  const centerY = (yMin + yMax) / 2;

  const scaleX = (x) => ((x - (centerX - span / 2)) / span) * (width - padding * 2) + padding;
  const scaleY = (y) => height - (((y - (centerY - span / 2)) / span) * (height - padding * 2) + padding);

  drawAxes(scaleX, scaleY, span, centerX, centerY);
  drawPath(record.history_xy, COLORS.past, 5);
  drawPath(record.future_xy, COLORS.future, 5);
  record.ranked_modes.forEach((mode, index) => {
    drawPath(mode.trajectory, getModeColor(index), 4, "10 8");
  });
  drawOrigin(scaleX(0), scaleY(0));

  function drawAxes(scaleXFn, scaleYFn, localSpan, cX, cY) {
    const gridCount = 8;
    for (let i = 0; i <= gridCount; i += 1) {
      const value = cX - localSpan / 2 + (localSpan / gridCount) * i;
      const x = scaleXFn(value);
      appendLine(x, padding, x, height - padding, "#ddd3c1", 1);
    }
    for (let i = 0; i <= gridCount; i += 1) {
      const value = cY - localSpan / 2 + (localSpan / gridCount) * i;
      const y = scaleYFn(value);
      appendLine(padding, y, width - padding, y, "#ddd3c1", 1);
    }
  }

  function drawPath(pointsList, color, strokeWidth, dash = "") {
    const d = pointsList
      .map((point, index) => `${index === 0 ? "M" : "L"} ${scaleX(point[0]).toFixed(2)} ${scaleY(point[1]).toFixed(2)}`)
      .join(" ");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", color);
    path.setAttribute("stroke-width", String(strokeWidth));
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    if (dash) {
      path.setAttribute("stroke-dasharray", dash);
    }
    chart.appendChild(path);

    pointsList.forEach((point, index) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", scaleX(point[0]));
      circle.setAttribute("cy", scaleY(point[1]));
      circle.setAttribute("r", index === pointsList.length - 1 ? "5" : "3");
      circle.setAttribute("fill", color);
      chart.appendChild(circle);
    });
  }

  function drawOrigin(x, y) {
    appendLine(x - 8, y, x + 8, y, "#333", 2);
    appendLine(x, y - 8, x, y + 8, "#333", 2);
  }

  function appendLine(x1, y1, x2, y2, stroke, widthValue) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("stroke", stroke);
    line.setAttribute("stroke-width", String(widthValue));
    chart.appendChild(line);
  }
}

function getModeColor(index) {
  return [COLORS.mode1, COLORS.mode2, COLORS.mode3][index] || COLORS.mode3;
}

function formatIntent(intent) {
  return intent.replaceAll("_", " ");
}
