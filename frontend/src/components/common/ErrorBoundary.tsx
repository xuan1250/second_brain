import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public override state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public override componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("[ErrorBoundary caught an error]:", error, errorInfo);
    this.setState({ errorInfo });
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  public override render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-full w-full flex-col items-center justify-center p-6 text-center bg-slate-950 text-slate-200">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-rose-500/15 border border-rose-500/30 text-rose-400 mb-3">
            <AlertTriangle className="h-6 w-6" />
          </div>
          <h2 className="text-sm font-bold text-white mb-1">
            {this.props.fallbackTitle || "Something went wrong in this view"}
          </h2>
          <p className="max-w-md text-xs text-slate-400 mb-4 leading-relaxed">
            {this.state.error?.message || "An unexpected rendering error occurred."}
          </p>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={this.handleRetry}
              className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white shadow-xs hover:bg-indigo-500 transition cursor-pointer"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Try Again</span>
            </button>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-700 transition cursor-pointer"
            >
              <span>Reload Page</span>
            </button>
          </div>

          {import.meta.env?.DEV && this.state.error?.stack && (
            <details className="mt-4 max-w-lg text-left">
              <summary className="text-[11px] text-slate-500 cursor-pointer hover:text-slate-400">
                Show Error Stack
              </summary>
              <pre className="mt-2 max-h-40 overflow-auto rounded bg-slate-900 p-2 text-[10px] text-rose-300 font-mono">
                {this.state.error.stack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}
