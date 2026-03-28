function findTitle() {
  const candidates = [
    document.querySelector("h1"),
    document.querySelector("[data-qa='job-title']"),
    document.querySelector("meta[property='og:title']"),
    document.querySelector("title")
  ].filter(Boolean);
  for (const candidate of candidates) {
    const text = candidate.content || candidate.textContent || "";
    if (text.trim()) {
      return text.trim();
    }
  }
  return "";
}

function findCompany() {
  const selectors = [
    "[data-qa='company-name']",
    "[class*='company']",
    "[data-testid*='company']"
  ];
  for (const selector of selectors) {
    const node = document.querySelector(selector);
    if (node?.textContent?.trim()) {
      return node.textContent.trim();
    }
  }
  return "";
}

function findLocation() {
  const selectors = [
    "[data-qa='location']",
    "[class*='location']",
    "[data-testid*='location']"
  ];
  for (const selector of selectors) {
    const node = document.querySelector(selector);
    if (node?.textContent?.trim()) {
      return node.textContent.trim();
    }
  }
  return "";
}

function findApplyLinks() {
  const actions = [];
  const seen = new Set();
  const clickTargets = [...document.querySelectorAll("a, button")];
  for (const node of clickTargets) {
    const text = (node.textContent || "").trim();
    const href = node.href || "";
    const signature = `${text}|${href}`;
    if (seen.has(signature)) continue;
    if (/apply|submit application|easy apply|start application/i.test(text) || /apply/i.test(href)) {
      actions.push({ text, url: href || location.href });
      seen.add(signature);
    }
  }
  return actions.slice(0, 10);
}

function extractJobPage() {
  return {
    title: findTitle(),
    company: findCompany(),
    location: findLocation(),
    pageUrl: location.href,
    visibleText: getVisibleText(),
    applyLinks: findApplyLinks(),
    metadata: {
      pageTitle: document.title,
      metaDescription: getMetaContent("description"),
      ogTitle: getMetaContent("og:title"),
      portalType: detectPortalType(location.href)
    }
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "EXTRACT_JOB_PAGE") {
    return false;
  }
  try {
    sendResponse({ ok: true, payload: extractJobPage() });
  } catch (error) {
    sendResponse({ ok: false, error: error.message });
  }
  return false;
});
