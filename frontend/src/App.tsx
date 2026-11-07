import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkspaceProvider } from "./lib/workspaceStore";
import { WorkspaceLayout } from "./components/layout/WorkspaceLayout";
import { ErrorBoundary } from "./components/common/ErrorBoundary";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export const App: React.FC = () => {
  return (
    <ErrorBoundary fallbackTitle="Second Brain Application Error">
      <QueryClientProvider client={queryClient}>
        <WorkspaceProvider>
          <WorkspaceLayout />
        </WorkspaceProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

export default App;

