const content = document.getElementById("content");

function render(result) {
  if (!result) {
    content.textContent = "No analysis yet.";
    return;
  }
  const parsed = result.parsed_job || {};
  const links = (parsed.apply_url ? [parsed.apply_url] : []).map((link) => `<li>${link}</li>`).join("");
  content.innerHTML = `
    <strong>${parsed.title || "Untitled role"}</strong>
    <div class="muted">${parsed.company || "Unknown company"} · ${parsed.location || "Unknown location"}</div>
    <p>${parsed.raw_text_preview || ""}</p>
    <div><strong>Portal:</strong> ${parsed.portal_type || "generic"}</div>
    <div><strong>Worker status:</strong> ${result.worker_status || "not started"}</div>
    <div><strong>Keywords:</strong> ${(result.tailored_resume?.keywords || []).join(", ")}</div>
    <ul class="list">${links}</ul>
  `;
}

async function loadLatest() {
  const data = await chrome.storage.local.get(["latestApplicationResult"]);
  render(data.latestApplicationResult || null);
}

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === "local" && changes.latestApplicationResult) {
    render(changes.latestApplicationResult.newValue);
  }
});

loadLatest();
