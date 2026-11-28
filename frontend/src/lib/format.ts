export function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return "0 B";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

export function formatPercent(
  ratio: number | null | undefined,
  fallback = 0
): string {
  const value =
    typeof ratio === "number" && Number.isFinite(ratio) && ratio > 0
      ? ratio
      : fallback;
  const pct = value <= 1 ? value * 100 : value;
  return `${Math.round(pct)}%`;
}

export function formatSubdomainLeaf(subdomain?: string | null): string {
  if (!subdomain) return "";
  return subdomain.split(".").pop() || subdomain;
}
