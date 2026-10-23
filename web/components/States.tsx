export function LoadingSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div data-testid="loading" role="status" aria-live="polite" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <span className="sr-only">Searching</span>
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="animate-pulse rounded-lg border border-slate-200 bg-white p-3">
          <div className="aspect-square rounded-md bg-slate-200" />
          <div className="mt-3 h-3 w-1/2 rounded bg-slate-200" />
          <div className="mt-2 h-3 w-3/4 rounded bg-slate-200" />
        </div>
      ))}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div
      data-testid="empty-state"
      className="rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center text-sm text-slate-600"
    >
      {message}
    </div>
  );
}

export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      data-testid="error-banner"
      role="alert"
      className="flex flex-wrap items-center gap-3 rounded-lg border border-rose-300 bg-rose-50 px-4 py-3 text-sm text-rose-900"
    >
      <span className="flex-1">{message}</span>
      {onRetry && (
        <button
          type="button"
          data-testid="retry-button"
          onClick={onRetry}
          className="rounded-md border border-rose-400 bg-white px-3 py-1 text-rose-900 hover:bg-rose-100"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function WarningsList({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <ul
      data-testid="warnings"
      className="list-disc space-y-0.5 rounded-md border border-amber-200 bg-amber-50 py-2 pl-7 pr-3 text-xs text-amber-900"
    >
      {warnings.map((w, i) => (
        <li key={i}>{w}</li>
      ))}
    </ul>
  );
}
