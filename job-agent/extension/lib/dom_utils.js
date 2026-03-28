function getVisibleText() {
  const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT);
  const chunks = [];
  while (walker.nextNode()) {
    const value = walker.currentNode.nodeValue?.replace(/\s+/g, " ").trim();
    if (!value) continue;
    chunks.push(value);
  }
  return chunks.join(" ").slice(0, 50000);
}

function getMetaContent(name) {
  const node =
    document.querySelector(`meta[property="${name}"]`) ||
    document.querySelector(`meta[name="${name}"]`);
  return node?.content || "";
}
