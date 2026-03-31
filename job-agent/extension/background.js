const BACKEND_URL = "http://127.0.0.1:8000";

async function extractFromTab(tabId) {
  const response = await chrome.tabs.sendMessage(tabId, { type: "EXTRACT_JOB_PAGE" });
  if (!response?.ok) {
    throw new Error(response?.error || "Failed to extract job data");
  }
  return response.payload;
}

async function runWorkflow(payload) {
  const profileSeed = await chrome.storage.local.get(["candidateProfile"]);
  const candidateProfile = profileSeed.candidateProfile || {
    profile_id: "default-profile",
    full_name: "Candidate Name",
    email: "candidate@example.com",
    phone: "555-123-4567",
    location: "Chicago, IL",
    linkedin_url: "https://linkedin.com/in/example",
    website_url: "",
    master_resume_text: "Summary\nAnalytics professional.\n\nTechnical Skills\nSQL, Python\n\nEducation\nUniversity Name\n\nProfessional Experience\nCompany A\n- Built reporting workflows.\n\nProjects\n- Analytics project.",
    work_authorization: {
      authorized_us: true,
      need_sponsorship_now: false,
      need_sponsorship_future: false
    },
    default_answers: {
      veteran_status: "Decline to answer",
      disability_status: "Decline to answer"
    },
    metadata: {}
  };

  const response = await fetch(`${BACKEND_URL}/run-application`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_url: payload.pageUrl,
      raw_text: payload.visibleText,
      title_hint: payload.title,
      company_hint: payload.company,
      location_hint: payload.location,
      page_metadata: payload.metadata,
      apply_links: payload.applyLinks.map((item) => item.url),
      profile: candidateProfile,
      submit_mode: "review_first",
      run_worker: false
    })
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Backend request failed: ${response.status} ${errorBody}`);
  }

  return response.json();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "ANALYZE_JOB") {
    return false;
  }

  (async () => {
    try {
      const extraction = await extractFromTab(message.tabId);
      const result = await runWorkflow(extraction);
      await chrome.storage.local.set({
        latestExtraction: extraction,
        latestApplicationResult: result
      });
      sendResponse({ ok: true, result });
    } catch (error) {
      sendResponse({ ok: false, error: error.message });
    }
  })();

  return true;
});
