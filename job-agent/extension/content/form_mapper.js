function collectFormDescriptors() {
  return [...document.querySelectorAll("input, textarea, select")].slice(0, 100).map((field) => ({
    label: document.querySelector(`label[for="${field.id}"]`)?.textContent?.trim() || "",
    name: field.name || "",
    id: field.id || "",
    placeholder: field.placeholder || "",
    nearbyText: field.closest("div, label, fieldset")?.textContent?.trim()?.slice(0, 120) || ""
  }));
}
