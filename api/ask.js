const fs = require("fs");
const path = require("path");

const PROJECT_ROOT = process.cwd();

const CLIMATE_KEYWORDS = new Set([
  "climate",
  "temperature",
  "temp",
  "rain",
  "rainfall",
  "precip",
  "weather",
  "monsoon",
  "trend",
  "hottest",
  "wettest",
  "warming",
  "cooling",
  "drought",
  "flood",
  "flooding",
  "heatwave",
  "evs",
  "celsius",
  "mm",
  "year",
  "years",
  "compare",
  "difference",
  "anomaly",
  "correlation",
  "emission",
  "emissions",
  "co2",
  "greenhouse",
  "sustainability",
  "environment",
  "environmental",
]);

const DEMO_ROWS = [
  { year: 2005, avg_temperature_c: 24.91, total_rainfall_mm: 1152 },
  { year: 2006, avg_temperature_c: 24.96, total_rainfall_mm: 1108 },
  { year: 2007, avg_temperature_c: 25.01, total_rainfall_mm: 1197 },
  { year: 2008, avg_temperature_c: 25.04, total_rainfall_mm: 1079 },
  { year: 2009, avg_temperature_c: 25.11, total_rainfall_mm: 1023 },
  { year: 2010, avg_temperature_c: 25.14, total_rainfall_mm: 1249 },
  { year: 2011, avg_temperature_c: 25.2, total_rainfall_mm: 1182 },
  { year: 2012, avg_temperature_c: 25.19, total_rainfall_mm: 1120 },
  { year: 2013, avg_temperature_c: 25.27, total_rainfall_mm: 1164 },
  { year: 2014, avg_temperature_c: 25.34, total_rainfall_mm: 1095 },
  { year: 2015, avg_temperature_c: 25.42, total_rainfall_mm: 1048 },
  { year: 2016, avg_temperature_c: 25.48, total_rainfall_mm: 1230 },
  { year: 2017, avg_temperature_c: 25.51, total_rainfall_mm: 1138 },
  { year: 2018, avg_temperature_c: 25.57, total_rainfall_mm: 1176 },
  { year: 2019, avg_temperature_c: 25.66, total_rainfall_mm: 1117 },
  { year: 2020, avg_temperature_c: 25.73, total_rainfall_mm: 1204 },
  { year: 2021, avg_temperature_c: 25.78, total_rainfall_mm: 1143 },
  { year: 2022, avg_temperature_c: 25.82, total_rainfall_mm: 1211 },
  { year: 2023, avg_temperature_c: 25.87, total_rainfall_mm: 1184 },
  { year: 2024, avg_temperature_c: 25.93, total_rainfall_mm: 1168 },
];

let cachedDataset = null;

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function normalizeQuestion(question) {
  return String(question || "").replace(/\s+/g, " ").trim().toLowerCase();
}

function directionFromChange(change, tolerance = 1e-9) {
  if (change > tolerance) return "increasing";
  if (change < -tolerance) return "decreasing";
  return "stable";
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      const next = line[i + 1];
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
      continue;
    }

    current += char;
  }

  values.push(current);
  return values;
}

function parseYearlyCsv(content) {
  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length < 2) return [];

  const headers = splitCsvLine(lines[0]).map((h) => h.trim().toLowerCase());
  const yearIdx = headers.indexOf("year");
  const tempIdx = headers.findIndex((h) => h.includes("avg_temperature"));
  const rainIdx = headers.findIndex((h) => h.includes("total_rainfall"));

  if (yearIdx < 0 || tempIdx < 0 || rainIdx < 0) return [];

  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const parts = splitCsvLine(lines[i]);
    const year = Number(parts[yearIdx]);
    const avgTemperature = Number(parts[tempIdx]);
    const totalRainfall = Number(parts[rainIdx]);

    if (!Number.isFinite(year) || !Number.isFinite(avgTemperature) || !Number.isFinite(totalRainfall)) {
      continue;
    }

    rows.push({
      year,
      avg_temperature_c: avgTemperature,
      total_rainfall_mm: totalRainfall,
    });
  }

  rows.sort((a, b) => a.year - b.year);
  return rows;
}

function loadDataset() {
  if (cachedDataset) return cachedDataset;

  const yearlyPath = path.join(PROJECT_ROOT, "data", "processed", "yearly_climate_india.csv");
  let rows = [];
  let dataMode = "demo";

  if (fs.existsSync(yearlyPath)) {
    try {
      const raw = fs.readFileSync(yearlyPath, "utf-8");
      rows = parseYearlyCsv(raw);
      if (rows.length > 0) {
        dataMode = "project";
      }
    } catch (_error) {
      rows = [];
    }
  }

  if (rows.length === 0) {
    rows = [...DEMO_ROWS];
  }

  cachedDataset = { rows, dataMode };
  return cachedDataset;
}

function inferYearSlice(rows, question) {
  const matches = String(question || "").match(/\b(?:19|20)\d{2}\b/g) || [];
  const years = matches.map((y) => Number(y)).filter((y) => Number.isFinite(y));

  if (years.length >= 2) {
    const sorted = years.slice(0, 2).sort((a, b) => a - b);
    const subset = rows.filter((row) => row.year >= sorted[0] && row.year <= sorted[1]);
    if (subset.length > 0) return subset;
  }

  if (years.length === 1) {
    const subset = rows.filter((row) => row.year === years[0]);
    if (subset.length > 0) return subset;
  }

  const count = Math.min(10, rows.length);
  return rows.slice(rows.length - count);
}

function isClimateQuestion(normalizedQuestion) {
  if (/\b(?:19|20)\d{2}\b/.test(normalizedQuestion)) return true;
  for (const keyword of CLIMATE_KEYWORDS) {
    if (normalizedQuestion.includes(keyword)) return true;
  }
  return false;
}

function isGreetingOrSmalltalk(normalizedQuestion) {
  const greetings = new Set([
    "hi",
    "hii",
    "hello",
    "hey",
    "yo",
    "hola",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "help",
    "thanks",
    "thank you",
  ]);

  if (greetings.has(normalizedQuestion)) return true;
  const words = normalizedQuestion.split(" ").filter(Boolean);
  return words.length <= 2 && words.some((word) => word === "hi" || word === "hello" || word === "hey");
}

function pickPhrase(seedText, options) {
  if (!options.length) return "";
  let sum = 0;
  for (const ch of seedText) sum += ch.charCodeAt(0);
  return options[sum % options.length];
}

function greetingAnswer(question) {
  const opener = pickPhrase(question, ["Hi!", "Hello!", "Hey!"]);
  const suggestion = pickPhrase(question, [
    "Try: Explain photosynthesis in 3 points.",
    "Try: Compare rainfall between 2010 and 2020.",
    "Try: Solve 125*48.",
  ]);

  return `${opener} I can answer general questions and climate-data questions. For climate questions, I include data representation.\n\n${suggestion}`;
}

function evaluateMath(question) {
  const normalized = normalizeQuestion(question);
  if (!/(solve|calculate|what is|math)/.test(normalized)) return null;

  const expression = normalized.replace(/[^0-9+\-*/().% ]/g, "").replace(/\s+/g, "");
  if (!expression || !/^[0-9+\-*/().%]+$/.test(expression)) return null;

  let value;
  try {
    value = Function(`"use strict"; return (${expression});`)();
  } catch (_error) {
    return null;
  }

  if (!Number.isFinite(value)) return null;
  const rendered = Number.isInteger(value) ? String(value) : Number(value).toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  return `Result: ${rendered}`;
}

function subsetMetrics(subset) {
  const ordered = [...subset].sort((a, b) => a.year - b.year);
  const first = ordered[0];
  const last = ordered[ordered.length - 1];
  const yearSpan = Math.max(1, last.year - first.year);

  const tempChange = last.avg_temperature_c - first.avg_temperature_c;
  const rainChange = last.total_rainfall_mm - first.total_rainfall_mm;

  let hottest = ordered[0];
  let wettest = ordered[0];
  let tempSum = 0;
  let rainSum = 0;

  for (const row of ordered) {
    tempSum += row.avg_temperature_c;
    rainSum += row.total_rainfall_mm;
    if (row.avg_temperature_c > hottest.avg_temperature_c) hottest = row;
    if (row.total_rainfall_mm > wettest.total_rainfall_mm) wettest = row;
  }

  return {
    start_year: first.year,
    end_year: last.year,
    row_count: ordered.length,
    temp_start: first.avg_temperature_c,
    temp_end: last.avg_temperature_c,
    temp_change: tempChange,
    temp_slope: tempChange / yearSpan,
    temp_direction: directionFromChange(tempChange),
    rain_start: first.total_rainfall_mm,
    rain_end: last.total_rainfall_mm,
    rain_change: rainChange,
    rain_slope: rainChange / yearSpan,
    rain_direction: directionFromChange(rainChange),
    avg_temp: tempSum / ordered.length,
    avg_rain: rainSum / ordered.length,
    hottest_year: hottest.year,
    hottest_temp: hottest.avg_temperature_c,
    wettest_year: wettest.year,
    wettest_rain: wettest.total_rainfall_mm,
  };
}

function extractFocus(question) {
  const q = String(question || "").toLowerCase();
  let focusTemperature = /(temperature|temp|heat|warming)/.test(q);
  let focusRainfall = /(rain|rainfall|precip|monsoon)/.test(q);
  const focusExtreme = /(hottest|wettest|highest|lowest|extreme|max|min)/.test(q);
  const focusCompare = /(compare|difference|vs|versus|between|change)/.test(q);

  if (!focusTemperature && !focusRainfall) {
    focusTemperature = true;
    focusRainfall = true;
  }

  return {
    temperature: focusTemperature,
    rainfall: focusRainfall,
    extreme: focusExtreme,
    compare: focusCompare,
  };
}

function implicationText(tempDirection, rainDirection) {
  if (tempDirection === "increasing" && rainDirection === "decreasing") {
    return "This suggests warming with moisture stress, increasing drought and heat-risk pressure.";
  }
  if (tempDirection === "increasing" && rainDirection === "increasing") {
    return "This suggests warmer and wetter conditions; adaptation should cover heat and intense rainfall.";
  }
  if (tempDirection === "stable" && rainDirection === "stable") {
    return "Both variables are relatively stable in this selected period; monitor anomalies closely.";
  }
  return "The variables move differently, so local seasonal and land-use context is important.";
}

function formatMarkdownTable(subset) {
  const header = "| Year | AvgTemperatureC | TotalRainfallMm |";
  const divider = "|---:|---:|---:|";
  const rows = subset.map((row) => {
    const year = String(row.year);
    const temp = row.avg_temperature_c.toFixed(3);
    const rain = row.total_rainfall_mm.toFixed(3);
    return `| ${year} | ${temp} | ${rain} |`;
  });
  return [header, divider, ...rows].join("\n");
}

function buildClimateAnswer(question, subset) {
  const focus = extractFocus(question);
  const metrics = subsetMetrics(subset);

  const lines = [
    "1) Trend Insight",
    `Selected period: ${metrics.start_year} to ${metrics.end_year} (${metrics.row_count} years).`,
  ];

  if (focus.temperature) {
    lines.push(
      `Temperature: ${metrics.temp_start.toFixed(2)} C -> ${metrics.temp_end.toFixed(2)} C (${metrics.temp_change >= 0 ? "+" : ""}${metrics.temp_change.toFixed(2)} C), ${metrics.temp_direction} at about ${metrics.temp_slope >= 0 ? "+" : ""}${metrics.temp_slope.toFixed(4)} C/year.`
    );
  }

  if (focus.rainfall) {
    lines.push(
      `Rainfall: ${metrics.rain_start.toFixed(2)} mm -> ${metrics.rain_end.toFixed(2)} mm (${metrics.rain_change >= 0 ? "+" : ""}${metrics.rain_change.toFixed(2)} mm), ${metrics.rain_direction} at about ${metrics.rain_slope >= 0 ? "+" : ""}${metrics.rain_slope.toFixed(2)} mm/year.`
    );
  }

  if (focus.extreme) {
    lines.push(
      `Extremes: hottest year ${metrics.hottest_year} (${metrics.hottest_temp.toFixed(2)} C), wettest year ${metrics.wettest_year} (${metrics.wettest_rain.toFixed(2)} mm).`
    );
  }

  if (focus.compare && metrics.row_count > 1) {
    lines.push(
      `Start vs end comparison: temperature ${metrics.temp_start.toFixed(2)} -> ${metrics.temp_end.toFixed(2)}, rainfall ${metrics.rain_start.toFixed(2)} -> ${metrics.rain_end.toFixed(2)}.`
    );
  }

  lines.push("");
  lines.push("2) Data Representation");
  lines.push(formatMarkdownTable(subset));
  lines.push("");
  lines.push("3) Climate-change Meaning");
  lines.push(implicationText(metrics.temp_direction, metrics.rain_direction));
  lines.push(
    `Average in this period: ${metrics.avg_temp.toFixed(2)} C temperature and ${metrics.avg_rain.toFixed(2)} mm rainfall.`
  );

  return lines.join("\n");
}

function buildCharts(subset, question) {
  const charts = [];
  const q = String(question || "").toLowerCase();
  const years = subset.map((row) => row.year);
  const temps = subset.map((row) => Number(row.avg_temperature_c.toFixed(2)));
  const rains = subset.map((row) => Number(row.total_rainfall_mm.toFixed(1)));

  if (/(temp|hot|warm|heat|celsius|cool)/.test(q)) {
    charts.push({
      id: "tempLine",
      type: "line",
      title: "Temperature Trend (°C)",
      labels: years,
      datasets: [
        {
          label: "Avg Temperature (°C)",
          data: temps,
          borderColor: "#ef5350",
          backgroundColor: "rgba(239,83,80,0.1)",
          fill: true,
          tension: 0.35,
        },
      ],
    });
  }

  if (/(rain|precip|monsoon|flood|drought|water|wet)/.test(q)) {
    charts.push({
      id: "rainBar",
      type: "bar",
      title: "Annual Rainfall (mm)",
      labels: years,
      datasets: [
        {
          label: "Total Rainfall (mm)",
          data: rains,
          backgroundColor: "rgba(66,165,245,0.6)",
          borderColor: "#42a5f5",
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    });
  }

  if (/(trend|compare|overall|analysis|climate|both|data|show)/.test(q)) {
    charts.push({
      id: "dualLine",
      type: "line",
      title: "Temperature & Rainfall Trends",
      labels: years,
      datasets: [
        {
          label: "Avg Temperature (°C)",
          data: temps,
          borderColor: "#ef5350",
          backgroundColor: "rgba(239,83,80,0.08)",
          fill: false,
          tension: 0.35,
          yAxisID: "y",
        },
        {
          label: "Total Rainfall (mm)",
          data: rains,
          borderColor: "#42a5f5",
          backgroundColor: "rgba(66,165,245,0.08)",
          fill: false,
          tension: 0.35,
          yAxisID: "y1",
        },
      ],
      dualAxis: true,
    });
  }

  if (!charts.length) {
    charts.push({
      id: "defaultLine",
      type: "line",
      title: "Climate Overview",
      labels: years,
      datasets: [
        {
          label: "Temperature (°C)",
          data: temps,
          borderColor: "#ef5350",
          fill: false,
          tension: 0.35,
          yAxisID: "y",
        },
        {
          label: "Rainfall (mm)",
          data: rains,
          borderColor: "#42a5f5",
          fill: false,
          tension: 0.35,
          yAxisID: "y1",
        },
      ],
      dualAxis: true,
    });
  }

  return charts;
}

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

function genericFallbackAnswer(question) {
  const mathResult = evaluateMath(question);
  if (mathResult) return mathResult;

  const normalized = normalizeQuestion(question);
  if (/(code|python|program|bug|error)/.test(normalized)) {
    return "Share your exact code snippet and full error message. I will give a direct corrected version and explain what to change.";
  }

  return "I could not fetch a reliable specific answer right now. Rephrase with clearer detail, for example: 'Explain photosynthesis in 5 bullet points' or 'Compare SQL and NoSQL with examples'.";
}

function buildClimateContext(subset) {
  const metrics = subsetMetrics(subset);
  return [
    `Period: ${metrics.start_year}-${metrics.end_year}`,
    `Temperature change: ${metrics.temp_change.toFixed(2)} C (${metrics.temp_direction})`,
    `Rainfall change: ${metrics.rain_change.toFixed(2)} mm (${metrics.rain_direction})`,
    `Hottest year: ${metrics.hottest_year}`,
    `Wettest year: ${metrics.wettest_year}`,
    "Data table:",
    formatMarkdownTable(subset),
  ].join("\n");
}

async function askHuggingFace(question, isClimate, climateContext) {
  const token = process.env.HF_TOKEN || process.env.HUGGINGFACEHUB_API_TOKEN;
  if (!token) return null;

  const model = process.env.HF_MODEL || "mistralai/Mistral-7B-Instruct-v0.3";
  const prompt = isClimate
    ? [
        "You are an EVS climate assistant. Use the provided context for data-backed answers.",
        "Answer directly and clearly.",
        `Question: ${question}`,
        "",
        "Climate context:",
        climateContext,
      ].join("\n")
    : [
        "You are a helpful AI assistant. Answer clearly and directly.",
        `Question: ${question}`,
      ].join("\n");

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 18000);

  try {
    const response = await fetch(`https://api-inference.huggingface.co/models/${encodeURIComponent(model)}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          max_new_tokens: 320,
          temperature: 0.3,
          return_full_text: false,
        },
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      return null;
    }

    const payload = await response.json();

    if (Array.isArray(payload) && payload.length > 0 && typeof payload[0]?.generated_text === "string") {
      return payload[0].generated_text.trim();
    }

    if (typeof payload?.generated_text === "string") {
      return payload.generated_text.trim();
    }

    return null;
  } catch (_error) {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

async function parsePayload(req) {
  if (req.body && typeof req.body === "object") {
    return req.body;
  }

  if (typeof req.body === "string" && req.body.trim()) {
    return JSON.parse(req.body);
  }

  const chunks = [];
  await new Promise((resolve, reject) => {
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", resolve);
    req.on("error", reject);
  });

  const raw = Buffer.concat(chunks).toString("utf-8").trim();
  if (!raw) return {};
  return JSON.parse(raw);
}

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  if (req.method !== "POST") {
    sendJson(res, 405, { error: "Method not allowed. Use POST /api/ask" });
    return;
  }

  try {
    const payload = await parsePayload(req);
    const question = String(payload?.question || "").trim();

    if (!question) {
      sendJson(res, 400, { error: "question is required" });
      return;
    }

    const normalized = normalizeQuestion(question);
    const { rows, dataMode } = loadDataset();

    if (isGreetingOrSmalltalk(normalized)) {
      sendJson(res, 200, {
        answer: greetingAnswer(question),
        mode: "greeting",
        data_mode: dataMode,
        charts: [],
      });
      return;
    }

    const climate = isClimateQuestion(normalized);
    const subset = climate ? inferYearSlice(rows, question) : null;
    const charts = climate && subset ? buildCharts(subset, question) : [];

    let answer = "";
    let mode = "fallback";

    if (climate && subset) {
      const context = buildClimateContext(subset);
      const hfAnswer = await askHuggingFace(question, true, context);
      if (hfAnswer) {
        answer = `${hfAnswer}\n\nData Representation (source values used):\n${formatMarkdownTable(subset)}`;
        mode = "hf-api";
      } else {
        answer = buildClimateAnswer(question, subset);
      }
    } else {
      const hfAnswer = await askHuggingFace(question, false, "");
      if (hfAnswer) {
        answer = hfAnswer;
        mode = "hf-api";
      } else {
        answer = genericFallbackAnswer(question);
      }
    }

    sendJson(res, 200, {
      answer: sanitizeAnswerText(answer),
      mode,
      data_mode: dataMode,
      charts,
    });
  } catch (error) {
    sendJson(res, 500, {
      error: `Unexpected server error: ${error instanceof Error ? error.message : String(error)}`,
    });
  }
};