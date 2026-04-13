const dom = {
  qaForm: document.getElementById("qaForm"),
  askBtn: document.getElementById("askBtn"),
  questionInput: document.getElementById("questionInput"),
  statusText: document.getElementById("statusText"),
  conversation: document.getElementById("conversation"),
  conversationPlaceholder: document.getElementById("conversationPlaceholder"),
  suggestedChips: document.getElementById("suggestedChips"),
  modeBadge: document.getElementById("modeBadge"),
};

/* ── Markdown renderer ── */
marked.setOptions({ breaks: true, gfm: true });

/* ── Track active Chart.js instances for cleanup ── */
const activeCharts = new Map();
let msgIndex = 0;

function sanitizeAnswerText(answer) {
  if (typeof answer !== "string") return "";

  const lines = answer.split(/\r?\n/);
  const diagnosticLine = /^\s*(?:response mode:\s*.*|(?:climate-)?fallback(?: mode enabled)?\s*\(.*\))\s*$/i;

  let removedPrefix = false;
  while (lines.length > 0 && diagnosticLine.test(lines[0])) {
    lines.shift();
    removedPrefix = true;
  }

  if (removedPrefix) {
    while (lines.length > 0 && !lines[0].trim()) {
      lines.shift();
    }
  }

  const cleaned = lines.join("\n").trim();
  return cleaned || answer.trim();
}

/* ── API call ── */
async function askBackend(question) {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    let details = "";
    try { details = await response.text(); } catch (_) { details = ""; }
    throw new Error(`Request failed (${response.status}) ${details}`.trim());
  }

  const data = await response.json();
  const cleanedAnswer = sanitizeAnswerText(data.answer);
  if (cleanedAnswer) {
    return {
      answer: cleanedAnswer,
      mode: typeof data.mode === "string" ? data.mode : "unknown",
      dataMode: typeof data.data_mode === "string" ? data.data_mode : "unknown",
      charts: Array.isArray(data.charts) ? data.charts : [],
    };
  }
  throw new Error("Backend did not return a valid answer field.");
}

/* ══════════════════════════════════════════════
   CHART RENDERING (Chart.js)
   ══════════════════════════════════════════════ */

function renderCharts(container, charts) {
  if (!charts || charts.length === 0) return;

  const chartsWrap = document.createElement("div");
  chartsWrap.className = "charts-grid";

  charts.forEach((chartConfig, idx) => {
    const chartCard = document.createElement("div");
    chartCard.className = "chart-card";
    chartCard.style.animationDelay = `${0.2 + idx * 0.15}s`;

    const title = document.createElement("div");
    title.className = "chart-title";
    title.textContent = chartConfig.title || "Chart";

    const canvasWrap = document.createElement("div");
    canvasWrap.className = "chart-canvas-wrap";

    const canvas = document.createElement("canvas");
    const chartId = `chart-${Date.now()}-${idx}`;
    canvas.id = chartId;

    canvasWrap.appendChild(canvas);
    chartCard.appendChild(title);
    chartCard.appendChild(canvasWrap);
    chartsWrap.appendChild(chartCard);

    // Destroy old chart with same logical id if exists
    if (activeCharts.has(chartConfig.id)) {
      activeCharts.get(chartConfig.id).destroy();
      activeCharts.delete(chartConfig.id);
    }

    // Defer chart creation to after DOM paint
    requestAnimationFrame(() => {
      const ctx = canvas.getContext("2d");
      const chartOptions = buildChartOptions(chartConfig);
      const chart = new Chart(ctx, chartOptions);
      activeCharts.set(chartConfig.id, chart);
    });
  });

  container.appendChild(chartsWrap);
}

function buildChartOptions(config) {
  const isDual = config.dualAxis === true;
  const isDonut = config.type === "doughnut";
  const isRadar = config.type === "radar";
  const inferredPrimaryLabel =
    typeof config?.datasets?.[0]?.label === "string" ? config.datasets[0].label : "Value";
  const inferredSecondaryLabel =
    typeof config?.datasets?.[1]?.label === "string" ? config.datasets[1].label : "Secondary Value";
  const xAxisLabel = typeof config?.xAxisLabel === "string" ? config.xAxisLabel : "Year";
  const yAxisLabel = typeof config?.yAxisLabel === "string" ? config.yAxisLabel : inferredPrimaryLabel;
  const y1AxisLabel = typeof config?.y1AxisLabel === "string" ? config.y1AxisLabel : inferredSecondaryLabel;

  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 1200,
      easing: "easeOutQuart",
    },
    plugins: {
      legend: {
        display: true,
        position: isDonut ? "bottom" : "top",
        labels: {
          color: "#8a95aa",
          font: { family: "Inter", size: 11 },
          padding: 12,
          usePointStyle: true,
          pointStyleWidth: 8,
        },
      },
      tooltip: {
        backgroundColor: "rgba(14, 18, 28, 0.92)",
        titleColor: "#e8ecf2",
        bodyColor: "#8a95aa",
        borderColor: "rgba(100, 130, 180, 0.2)",
        borderWidth: 1,
        cornerRadius: 8,
        padding: 10,
        titleFont: { family: "Space Grotesk", weight: "600" },
        bodyFont: { family: "Inter" },
      },
    },
  };

  // Scales for line/bar
  if (!isDonut && !isRadar) {
    baseOptions.scales = {
      x: {
        title: {
          display: true,
          text: xAxisLabel,
          color: "#7d8798",
          font: { family: "Space Grotesk", size: 11, weight: "600" },
        },
        ticks: { color: "#5a6577", font: { size: 10 } },
        grid: { color: "rgba(100,130,180,0.06)" },
      },
      y: {
        position: "left",
        title: {
          display: true,
          text: yAxisLabel,
          color: "#ef5350",
          font: { family: "Space Grotesk", size: 11, weight: "600" },
        },
        ticks: { color: "#ef5350", font: { size: 10 } },
        grid: { color: "rgba(100,130,180,0.06)" },
      },
    };

    if (isDual) {
      baseOptions.scales.y1 = {
        position: "right",
        title: {
          display: true,
          text: y1AxisLabel,
          color: "#42a5f5",
          font: { family: "Space Grotesk", size: 11, weight: "600" },
        },
        ticks: { color: "#42a5f5", font: { size: 10 } },
        grid: { drawOnChartArea: false },
      };
    }
  }

  // Radar specific
  if (isRadar) {
    baseOptions.scales = {
      r: {
        ticks: { color: "#5a6577", backdropColor: "transparent", font: { size: 9 } },
        grid: { color: "rgba(100,130,180,0.1)" },
        pointLabels: { color: "#8a95aa", font: { size: 11 } },
      },
    };
  }

  // Donut specific
  if (isDonut) {
    baseOptions.cutout = "65%";
  }

  return {
    type: config.type,
    data: {
      labels: config.labels,
      datasets: config.datasets.map((ds) => ({
        ...ds,
        pointRadius: config.type === "line" ? 3 : undefined,
        pointHoverRadius: config.type === "line" ? 6 : undefined,
        pointBackgroundColor: ds.borderColor,
      })),
    },
    options: baseOptions,
  };
}

/* ══════════════════════════════════════════════
   UI HELPERS
   ══════════════════════════════════════════════ */

function setLoading(isLoading) {
  dom.askBtn.disabled = isLoading;
  dom.askBtn.classList.toggle("is-loading", isLoading);
  dom.statusText.textContent = isLoading ? "Generating response..." : "Endpoint: POST /api/ask";
  if (isLoading) dom.questionInput.blur();
}

function hidePlaceholder() {
  if (dom.conversationPlaceholder) {
    dom.conversationPlaceholder.style.animation = "fadeOut 0.4s ease forwards";
    setTimeout(() => {
      if (dom.conversationPlaceholder) dom.conversationPlaceholder.style.display = "none";
    }, 400);
  }
}

function addUserMessage(question) {
  hidePlaceholder();
  msgIndex++;
  const bubble = document.createElement("div");
  bubble.className = "msg msg-user";
  bubble.innerHTML = `
    <div class="msg-label">You</div>
    <div class="msg-body">${escapeHtml(question)}</div>
  `;
  dom.conversation.appendChild(bubble);
  scrollToBottom();
}

function addTypingIndicator() {
  const indicator = document.createElement("div");
  indicator.className = "msg msg-ai typing-indicator";
  indicator.id = "typingIndicator";
  indicator.style.animationDelay = "0.15s";
  indicator.innerHTML = `
    <div class="msg-label">🌍 Climate AI</div>
    <div class="msg-body">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </div>
  `;
  dom.conversation.appendChild(indicator);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) {
    el.style.animation = "msgDisappear 0.3s ease forwards";
    setTimeout(() => el.remove(), 300);
  }
}

function addAIMessage(answer, mode, dataMode, charts) {
  removeTypingIndicator();

  setTimeout(() => {
    msgIndex++;
    const bubble = document.createElement("div");
    bubble.className = "msg msg-ai";

    const renderedMarkdown = marked.parse(answer);
    const modeLabel = mode === "greeting" ? "greeting" : `${mode} | data: ${dataMode}`;

    bubble.innerHTML = `
      <div class="msg-label">🌍 Climate AI <span class="mode-tag">${modeLabel}</span></div>
      <div class="msg-body markdown-body">${renderedMarkdown}</div>
    `;
    dom.conversation.appendChild(bubble);

    // Render charts below the markdown
    if (charts && charts.length > 0) {
      renderCharts(bubble, charts);
    }

    // Animate list items
    requestAnimationFrame(() => {
      const items = bubble.querySelectorAll("li");
      items.forEach((li, i) => {
        li.style.opacity = "0";
        li.style.animation = `fadeUp 0.4s ease ${0.1 + i * 0.06}s forwards`;
      });
    });

    scrollToBottom();
    // Scroll again after charts render
    setTimeout(() => scrollToBottom(), 600);
  }, 350);
}

function addErrorMessage(errorText) {
  removeTypingIndicator();
  setTimeout(() => {
    const bubble = document.createElement("div");
    bubble.className = "msg msg-ai msg-error";
    bubble.innerHTML = `
      <div class="msg-label">⚠️ Error</div>
      <div class="msg-body">${escapeHtml(errorText)}</div>
    `;
    dom.conversation.appendChild(bubble);
    scrollToBottom();
  }, 350);
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    dom.conversation.scrollTo({ top: dom.conversation.scrollHeight, behavior: "smooth" });
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function updateModeBadge(mode) {
  const badge = dom.modeBadge;
  badge.style.animation = "none";
  badge.offsetHeight;
  badge.style.animation = "badgePop 0.5s cubic-bezier(0.16, 1, 0.3, 1)";
  badge.textContent = mode;
  badge.className = "status-badge";
  if (mode === "mixtral") badge.classList.add("badge-mixtral");
  else if (mode === "hf-api") badge.classList.add("badge-hf");
  else if (mode === "fallback") badge.classList.add("badge-fallback");
  else if (mode === "greeting") badge.classList.add("badge-greeting");
}

function animateChipClick(chip) {
  chip.style.transform = "scale(0.92)";
  chip.style.transition = "transform 0.1s ease";
  setTimeout(() => { chip.style.transform = ""; chip.style.transition = ""; }, 150);
}

/* ══════════════════════════════════════════════
   EVENT WIRING
   ══════════════════════════════════════════════ */

async function handleQuestion(question) {
  if (!question.trim()) return;

  dom.questionInput.value = "";
  
  // Collapse hero section for a better reading experience
  const card = document.querySelector(".card");
  if (card && !card.classList.contains("card-collapsed")) {
    card.classList.add("card-collapsed");
    dom.conversation.classList.add("conversation-expanded");
  }

  addUserMessage(question);
  await new Promise((r) => setTimeout(r, 200));
  addTypingIndicator();
  setLoading(true);

  try {
    const result = await askBackend(question);
    addAIMessage(result.answer, result.mode, result.dataMode, result.charts);
    updateModeBadge(result.mode);
    dom.statusText.textContent = `Last response: ${result.mode} mode | Endpoint: POST /api/ask`;
  } catch (error) {
    addErrorMessage(`Could not get AI answer. Make sure the backend is running.\n\nError: ${error.message}`);
    dom.statusText.textContent = "Endpoint: POST /api/ask";
  } finally {
    setLoading(false);
  }
}


function wireEvents() {
  dom.qaForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await handleQuestion(dom.questionInput.value.trim());
  });

  dom.suggestedChips.addEventListener("click", async (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    animateChipClick(chip);
    const question = chip.dataset.question;
    if (question) {
      dom.questionInput.value = question;
      await handleQuestion(question);
    }
  });

  dom.questionInput.addEventListener("focus", () => dom.questionInput.parentElement.classList.add("is-focused"));
  dom.questionInput.addEventListener("blur", () => dom.questionInput.parentElement.classList.remove("is-focused"));
}

/* ── Inject dynamic keyframes ── */
function injectDynamicStyles() {
  const style = document.createElement("style");
  style.textContent = `
    @keyframes fadeOut { from { opacity:1; transform:translateY(0); } to { opacity:0; transform:translateY(10px); } }
    @keyframes msgDisappear { from { opacity:1; transform:scale(1); } to { opacity:0; transform:scale(0.95); } }
    @keyframes badgePop { 0% { transform:scale(0.8); opacity:0.5; } 60% { transform:scale(1.1); } 100% { transform:scale(1); opacity:1; } }
  `;
  document.head.appendChild(style);
}

injectDynamicStyles();
wireEvents();
