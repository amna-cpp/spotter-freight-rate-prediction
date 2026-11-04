const $ = (selector) => document.querySelector(selector);
const form = $("#load-form");
const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});
const count = new Intl.NumberFormat("en-US");
const dateLabel = (date) =>
  new Date(`${date}T12:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
const presets = {
  december: {
    pickup: "Lexington",
    delivery: "Fort Wayne",
    distance: 360,
    weight: 32000,
    equipment: "Dry Van",
    date: "2025-12-04",
  },
  longhaul: {
    pickup: "Los Angeles",
    delivery: "Dallas",
    distance: 1435,
    weight: 38000,
    equipment: "Reefer",
    date: "2025-11-18",
  },
};
let study;
let requestNumber = 0;
let chartScale;

function applyPreset(load) {
  for (const [key, value] of Object.entries(load))
    form.elements[key].value = value;
}

async function predict(event) {
  event?.preventDefault();
  if (!form.reportValidity()) return;
  const thisRequest = ++requestNumber;
  const load = Object.fromEntries(new FormData(form));
  const button = $("#submit-button");
  button.disabled = true;
  button.querySelector("span").textContent = "Estimating…";
  $("#form-error").hidden = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(load),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(result.error || "The estimate could not be calculated.");
    if (thisRequest !== requestNumber) return;
    $("#estimated-rate").textContent = currency.format(result.predicted_rate);
    $("#per-mile").textContent =
      `${currency.format(result.rate_per_mile)} / mile`;
    $("#summary-pickup").textContent = load.pickup;
    $("#summary-delivery").textContent = load.delivery;
    $("#load-summary").textContent =
      `${count.format(Number(load.distance))} mi · ${load.equipment} · ${count.format(result.weight_used)} lb · ${dateLabel(load.date)}`;
    $("#weight-note").hidden = !result.weight_imputed;
    $("#weight-note").textContent = result.weight_imputed
      ? `Weight filled using the training-set equipment median: ${count.format(result.weight_used)} lb.`
      : "";
  } catch (error) {
    $("#form-error").textContent =
      error instanceof SyntaxError || error instanceof TypeError
        ? "The estimator is unavailable. Please try again."
        : error.message;
    $("#form-error").hidden = false;
  } finally {
    if (thisRequest === requestNumber) {
      button.disabled = false;
      button.querySelector("span").textContent = "Estimate rate";
    }
  }
}

function drawChart() {
  const rows = study.december;
  const values = rows.map((row) => row.predicted_rate);
  const step = Math.max(5, Math.ceil((Math.max(...values) - Math.min(...values)) / 15) * 5);
  const low = Math.floor(Math.min(...values) / step) * step - step;
  const high = Math.ceil(Math.max(...values) / step) * step + step;
  chartScale = (value) => 154 - ((value - low) / (high - low)) * 127;
  const x = (index) => 44 + index * (588 / 30);
  const y = chartScale;
  let content =
    "<title>Fixed-load December predictions</title><desc>Daily estimates from the submitted model, with only the pickup date changing.</desc>";
  for (let value = low; value <= high; value += step) {
    content += `<line x1="44" y1="${y(value)}" x2="632" y2="${y(value)}" stroke="#dedfd3" stroke-width="1"/><text class="chart-label" x="0" y="${y(value) + 4}">$${value}</text>`;
  }
  for (const day of [1, 8, 15, 22, 31])
    content += `<text class="chart-label" x="${x(day - 1)}" y="180" text-anchor="${day === 1 ? "start" : day === 31 ? "end" : "middle"}">Dec ${day}</text>`;
  const path = rows
    .map(
      (row, index) =>
        `${index === 0 ? "M" : "L"}${x(index)},${y(row.predicted_rate)}`,
    )
    .join(" ");
  content += `<path d="${path}" fill="none" stroke="#5d6944" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/><line id="selected-line" y1="20" y2="155" stroke="#93482e" stroke-dasharray="3 4" opacity=".5"/><circle id="selected-point" r="4.5" fill="#93482e" stroke="#fffef9" stroke-width="2"/>`;
  $("#december-chart").innerHTML = content;
  selectDay();
}

function selectDay() {
  const index = Number($("#chart-day").value) - 1;
  const row = study.december[index];
  const x = 44 + index * (588 / 30);
  const y = chartScale(row.predicted_rate);
  $("#selected-line").setAttribute("x1", x);
  $("#selected-line").setAttribute("x2", x);
  $("#selected-point").setAttribute("cx", x);
  $("#selected-point").setAttribute("cy", y);
  $("#day-readout").textContent =
    `Dec ${index + 1} · ${currency.format(row.predicted_rate)}`;
  $("#chart-day").setAttribute(
    "aria-valuetext",
    `December ${index + 1}, ${currency.format(row.predicted_rate)}`,
  );
}

async function start() {
  form.addEventListener("submit", predict);
  try {
    const response = await fetch("/data.json");
    if (!response.ok) throw new Error("Could not load the study.");
    study = await response.json();
    const result = study.metrics.october;
    const baseline = study.metrics.october_baseline;
    $("#baseline-mae").textContent = currency.format(baseline.mae);
    $("#selected-mae").textContent = currency.format(result.mae);
    $("#context-mae").textContent = currency.format(result.mae);
    $("#mae-improvement").textContent =
      `${(100 * (1 - result.mae / baseline.mae)).toFixed(1)}% lower MAE on the same October loads.`;
    $("#within-tolerance").textContent =
      `${result.within_10pct_pct.toFixed(2)}% of predictions within 10% of the actual price.`;
    $("#metric-mape").textContent = `${result.mape_pct.toFixed(2)}%`;
    $("#metric-rmse").textContent = currency.format(result.rmse);
    $("#metric-r2").textContent = result.r2.toFixed(3);
    for (const id of ["pickup", "delivery"]) {
      const select = $(`#${id}`);
      select.replaceChildren(
        ...study.cities.map((city) => new Option(city, city)),
      );
    }
    applyPreset(presets.december);
    drawChart();
    $("#chart-day").addEventListener("input", selectDay);
    $("#use-chart-load").addEventListener("click", () => {
      applyPreset({
        ...presets.december,
        date: study.december[Number($("#chart-day").value) - 1].date,
      });
      predict();
      if (window.innerWidth < 740)
        $("#estimate").scrollIntoView({ behavior: "smooth" });
    });
    document.querySelectorAll("[data-preset]").forEach((button) =>
      button.addEventListener("click", () => {
        applyPreset(presets[button.dataset.preset]);
        predict();
      }),
    );
    await predict();
  } catch {
    $("#form-error").textContent =
      "The demo could not load. Refresh the page to try again.";
    $("#form-error").hidden = false;
  }
}

start();
