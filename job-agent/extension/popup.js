const statusEl = document.getElementById("status");
const analyzeButton = document.getElementById("analyze-button");
const openPanelButton = document.getElementById("open-panel");

function setStatus(message) {
  statusEl.textContent = message;
}

analyzeButton.addEventListener("click", async () => {
  setStatus("Analyzing page...");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const response = await chrome.runtime.sendMessage({ type: "ANALYZE_JOB", tabId: tab.id });
    if (!response?.ok) {
      throw new Error(response?.error || "Unknown extension error");
    }
    setStatus(`Saved analysis for ${response.result?.parsed_job?.title || "job"}.`);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
});

openPanelButton.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  await chrome.sidePanel.open({ tabId: tab.id });
});
