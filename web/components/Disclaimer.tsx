import { DISCLAIMER } from "@/lib/modes";

export function Disclaimer() {
  return (
    <footer
      role="contentinfo"
      data-testid="disclaimer"
      className="fixed inset-x-0 bottom-0 z-40 border-t border-amber-300 bg-amber-50 px-4 py-2 text-center text-sm font-medium text-amber-900"
    >
      {DISCLAIMER}
    </footer>
  );
}
