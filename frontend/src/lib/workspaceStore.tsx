import React from "react";
import {
  ChatFilterProvider,
  DEFAULT_DOMAIN_FILTER,
  type ChatFilterContextValue,
  useChatFilter,
} from "./store/ChatFilterContext";
import {
  LlmSelectionProvider,
  type LlmSelectionContextValue,
  useLlmSelection,
} from "./store/LlmSelectionContext";
import {
  SettingsModalProvider,
  type SettingsModalContextValue,
  useSettingsModal,
} from "./store/SettingsModalContext";
import {
  type InspectorTab,
  type ViewerContextValue,
  ViewerProvider,
  useViewer,
} from "./store/ViewerContext";

export { DEFAULT_DOMAIN_FILTER };
export type { InspectorTab };
export { useChatFilter, useLlmSelection, useSettingsModal, useViewer };

export type WorkspaceStateContextValue = ViewerContextValue &
  ChatFilterContextValue &
  LlmSelectionContextValue &
  SettingsModalContextValue;

export const WorkspaceProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return (
    <ViewerProvider>
      <ChatFilterProvider>
        <LlmSelectionProvider>
          <SettingsModalProvider>{children}</SettingsModalProvider>
        </LlmSelectionProvider>
      </ChatFilterProvider>
    </ViewerProvider>
  );
};

export function useWorkspace(): WorkspaceStateContextValue {
  const viewer = useViewer();
  const chatFilter = useChatFilter();
  const llm = useLlmSelection();
  const settings = useSettingsModal();

  return {
    ...viewer,
    ...chatFilter,
    ...llm,
    ...settings,
  };
}
