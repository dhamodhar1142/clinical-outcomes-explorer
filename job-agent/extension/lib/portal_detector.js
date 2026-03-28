function detectPortalType(url) {
  const lowered = (url || location.href).toLowerCase();
  if (lowered.includes("workday")) return "workday";
  if (lowered.includes("greenhouse")) return "greenhouse";
  if (lowered.includes("lever")) return "lever";
  return "generic";
}
