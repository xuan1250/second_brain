import { useCallback, useEffect, useMemo } from "react";
import type { ActiveCitation, HeadingOutlineItem } from "../lib/types";

export function normalizeMarkdownMatch(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function useMarkdownCitationHighlight({
  containerRef,
  markdown,
  isCitedDoc,
  activeCitation,
}: {
  containerRef: React.RefObject<HTMLDivElement | null>;
  markdown?: string;
  isCitedDoc: boolean;
  activeCitation: ActiveCitation | null;
}) {
  const targetSnippet = useMemo(() => {
    if (!isCitedDoc || !activeCitation) return "";
    const rawExcerpt = (activeCitation.excerpt || "").trim();
    return rawExcerpt.length >= 12
      ? normalizeMarkdownMatch(rawExcerpt.slice(0, 90))
      : "";
  }, [isCitedDoc, activeCitation]);

  const targetHeadingLeaf = useMemo(() => {
    if (!isCitedDoc || !activeCitation?.heading_path) return "";
    const parts = activeCitation.heading_path
      .split(">")
      .map((s) => s.trim())
      .filter(Boolean);
    return parts.length > 0
      ? normalizeMarkdownMatch(parts[parts.length - 1])
      : "";
  }, [isCitedDoc, activeCitation?.heading_path]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !markdown || !isCitedDoc || !activeCitation) {
      return;
    }

    container
      .querySelectorAll(".citation-pulse-highlight")
      .forEach((el) => el.classList.remove("citation-pulse-highlight"));

    const timer = window.setTimeout(() => {
      let matchedElement: HTMLElement | null = null;

      if (targetSnippet) {
        const snippetWords = targetSnippet.split(" ").slice(0, 7).join(" ");
        const blocks = Array.from(
          container.querySelectorAll<HTMLElement>(
            "p, li, tr, td, blockquote, pre, h1, h2, h3, h4, h5, h6"
          )
        );
        for (const el of blocks) {
          const normText = normalizeMarkdownMatch(el.textContent || "");
          if (
            normText &&
            snippetWords.length >= 6 &&
            normText.includes(snippetWords)
          ) {
            matchedElement = el;
            break;
          }
        }
      }

      if (!matchedElement && targetHeadingLeaf) {
        const headings = Array.from(
          container.querySelectorAll<HTMLElement>("h1, h2, h3, h4, h5, h6")
        );
        for (const h of headings) {
          const normHeading = normalizeMarkdownMatch(h.textContent || "");
          if (
            normHeading &&
            (normHeading.includes(targetHeadingLeaf) ||
              targetHeadingLeaf.includes(normHeading))
          ) {
            matchedElement = h;
            break;
          }
        }
      }

      if (matchedElement) {
        matchedElement.classList.add("citation-pulse-highlight");
        matchedElement.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      }
    }, 120);

    return () => window.clearTimeout(timer);
  }, [
    containerRef,
    markdown,
    isCitedDoc,
    activeCitation,
    targetSnippet,
    targetHeadingLeaf,
  ]);

  const scrollToHeading = useCallback(
    (heading: HeadingOutlineItem) => {
      const container = containerRef.current;
      if (!container) return;
      const normTarget = normalizeMarkdownMatch(heading.text);
      const headingEls = Array.from(
        container.querySelectorAll<HTMLElement>("h1, h2, h3, h4, h5, h6")
      );
      const found = headingEls.find(
        (el) => normalizeMarkdownMatch(el.textContent || "") === normTarget
      );
      if (found) {
        container
          .querySelectorAll(".citation-pulse-highlight")
          .forEach((el) => el.classList.remove("citation-pulse-highlight"));
        found.classList.add("citation-pulse-highlight");
        found.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    },
    [containerRef]
  );

  return { scrollToHeading };
}
