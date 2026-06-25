from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import gradio as gr

from fixed.config import CONFIG, STATIC_DIR
from fixed.agent_runtime import AgentRuntime


runtime = AgentRuntime()
CSS_PATH = STATIC_DIR / "app.css"
MAX_CONVERSATION_BUTTONS = 12
DEFAULT_PENDING_STATUS = "답변을 진행중입니다"
ENTER_TO_SEND_HEAD = """
<style>
@media (min-width: 981px) {
  html,
  body {
    height: 100dvh !important;
    overflow-y: hidden !important;
  }
}
</style>
<script>
function setKananaTextareaValue(textarea, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(textarea, value);
  textarea.dispatchEvent(new Event("input", { bubbles: true }));
  textarea.dispatchEvent(new Event("change", { bubbles: true }));
}

(function setupKananaPendingCleanup() {
  if (window.__kananaPendingCleanupReady) return;
  window.__kananaPendingCleanupReady = true;

  function isKananaPendingMessage(message) {
    const small = message?.querySelector("small");
    const status = (small?.textContent || "").trim();
    return status === "답변을 진행중입니다" || (status.startsWith("현재 ") && status.endsWith(" 실행 중"));
  }

  function hideKananaPendingPanel(panel) {
    panel.dataset.kananaHiddenPending = "true";
    panel.hidden = true;
    panel.style.display = "none";
    panel.setAttribute("aria-hidden", "true");
  }

  function revealKananaMessagePanel(panel) {
    if (panel.dataset.kananaHiddenPending !== "true") return;
    panel.hidden = false;
    panel.style.display = "";
    panel.removeAttribute("aria-hidden");
    delete panel.dataset.kananaHiddenPending;
  }

  window.cleanupKananaPendingMessages = function cleanupKananaPendingMessages() {
    const chatbot = document.querySelector("#kanana-chatbot");
    if (!chatbot) return;

    const messageGroups = new Set();
    chatbot.querySelectorAll("[data-testid='bot']").forEach((botMessage) => {
      const messagePanel = botMessage.closest(".message");
      if (messagePanel?.parentElement) {
        messageGroups.add(messagePanel.parentElement);
      }
    });

    messageGroups.forEach((group) => {
      const panels = Array.from(group.children).filter((child) => child.classList?.contains("message"));
      panels.forEach(revealKananaMessagePanel);
      const pendingPanels = panels.filter(isKananaPendingMessage);
      const finalPanels = panels.filter((panel) => !isKananaPendingMessage(panel) && (panel.textContent || "").trim());

      if (finalPanels.length > 0) {
        pendingPanels.forEach(hideKananaPendingPanel);
        return;
      }

      pendingPanels.slice(0, -1).forEach(hideKananaPendingPanel);
    });
  };

  function observeKananaPendingMessages() {
    if (!document.body) return;
    const observer = new MutationObserver(window.cleanupKananaPendingMessages);
    observer.observe(document.body, { childList: true, characterData: true, subtree: true });
    window.cleanupKananaPendingMessages();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", observeKananaPendingMessages, { once: true });
  } else {
    observeKananaPendingMessages();
  }
})();

document.addEventListener("keydown", function(event) {
  const target = event.target;
  if (!(target instanceof HTMLTextAreaElement)) return;
  if (!target.closest("#kanana-input")) return;
  if (event.key !== "Enter" || event.isComposing) return;

  event.preventDefault();
  event.stopPropagation();

  if (event.shiftKey) {
    const start = target.selectionStart ?? target.value.length;
    const end = target.selectionEnd ?? target.value.length;
    const nextValue = `${target.value.slice(0, start)}\n${target.value.slice(end)}`;
    setKananaTextareaValue(target, nextValue);
    target.setSelectionRange(start + 1, start + 1);
    return;
  }

  const sendRoot = document.querySelector("#kanana-send");
  const sendButton =
    sendRoot?.matches("button") ? sendRoot : sendRoot?.querySelector("button");

  if (sendButton && !sendButton.disabled) {
    sendButton.dispatchEvent(new MouseEvent("click", {
      bubbles: true,
      cancelable: true,
      view: window
    }));
  }
}, true);
</script>
"""
DELETE_CONVERSATION_CONFIRM_JS = """
function(conversationId) {
  if (!conversationId) {
    alert("삭제할 저장된 대화를 먼저 선택해 주세요.");
    return [""];
  }
  const confirmed = confirm("선택한 저장된 대화를 영구 삭제할까요?\\n이 작업은 되돌릴 수 없습니다.");
  return [confirmed ? conversationId : ""];
}
"""


def _trace_message(trace: dict[str, Any]) -> str:
    events = trace.get("events", [])
    lines = []
    for event in events[-8:]:
        label = event.get("event")
        tool_name = event.get("tool_name")
        if tool_name:
            lines.append(f"- `{label}` · `{tool_name}`")
        else:
            lines.append(f"- `{label}`")
    return "\n".join(lines) if lines else "- trace 없음"


def _sqlite_schedule_memory_enabled() -> bool:
    return int(getattr(runtime, "active_week", CONFIG.active_week) or 1) >= 3


def _saved_schedule_lines(limit: int = 8) -> list[str]:
    if not _sqlite_schedule_memory_enabled():
        return []
    list_schedules = getattr(runtime.app_store, "list_schedules", None)
    if list_schedules is None:
        return []
    rows = list_schedules(limit=limit, kind="personal_schedule")
    lines: list[str] = []
    for row in rows:
        date = row.get("date") or "날짜 미정"
        start_time = row.get("start_time") or "시간 미정"
        end_time = row.get("end_time") or ""
        time_range = f"{start_time}-{end_time}" if end_time else start_time
        title = row.get("title") or "제목 없음"
        attendees = row.get("attendees") or []
        attendee_text = f" · 참석자: {', '.join(attendees)}" if attendees else ""
        lines.append(f"- {date} {time_range} · {title}{attendee_text}")
    return lines


def _saved_schedule_markdown(limit: int = 8) -> str:
    if not _sqlite_schedule_memory_enabled():
        return "Week 1 main 브랜치에서는 현재 대화 안의 임시 일정만 사용합니다."
    lines = _saved_schedule_lines(limit=limit)
    if not lines:
        return "저장된 일정이 아직 없습니다."
    return "\n".join(lines)


def _chat_notice() -> list[dict[str, str]]:
    return []


def _pending_assistant_message(status_text: str = DEFAULT_PENDING_STATUS) -> dict[str, str]:
    return {"role": "assistant", "content": f"...\n\n<small>{html.escape(status_text)}</small>"}


def _is_pending_assistant_message(message: dict[str, Any] | None) -> bool:
    if not message or message.get("role") != "assistant":
        return False
    content = str(message.get("content") or "").strip()
    has_default_status = DEFAULT_PENDING_STATUS in content
    has_tool_status = "현재 " in content and " 실행 중" in content
    return content.startswith("...") and (has_default_status or has_tool_status)


def _replace_pending_status(history: list[dict[str, Any]], status_text: str) -> list[dict[str, Any]]:
    pending_message = _pending_assistant_message(status_text)
    if history and _is_pending_assistant_message(history[-1]):
        return [*history[:-1], pending_message]
    return [*history, pending_message]


def _replace_pending_with_answer(history: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
    assistant_message = {"role": "assistant", "content": answer}
    if history and _is_pending_assistant_message(history[-1]):
        return [*history[:-1], assistant_message]
    return [*history, assistant_message]


def _saved_chatbot_history(conversation_id: str) -> list[dict[str, str]]:
    return [
        message
        for message in runtime.load_messages_for_chatbot(conversation_id)
        if not _is_pending_assistant_message(message)
    ]


def _conversation_rows() -> list[dict[str, str]]:
    rows = runtime.app_store.list_conversations()
    return [
        {
            "conversation_id": row["conversation_id"],
            "title": row["title"] or "새 대화",
            "preview": (row.get("last_message") or "").replace("\n", " ")[:54],
        }
        for row in rows[:MAX_CONVERSATION_BUTTONS]
    ]


def _conversation_button_updates(selected_id: str | None = None) -> list[Any]:
    rows = _conversation_rows()
    updates: list[Any] = []
    for index in range(MAX_CONVERSATION_BUTTONS):
        if index < len(rows):
            row = rows[index]
            label = row["title"]
            if row["conversation_id"] == selected_id:
                label = f"● {label}"
            updates.append(gr.update(value=label, visible=True))
        else:
            updates.append(gr.update(value="", visible=False))
    return updates


def queue_user_message(
    message: str,
    history: list[dict[str, Any]] | None,
    conversation_id: str | None,
) -> tuple:
    history = history or []
    message = (message or "").strip()
    if not message:
        return (
            history,
            {},
            conversation_id or "",
            gr.update(value="", interactive=True),
            "",
            gr.update(interactive=True),
            _saved_schedule_markdown(),
            *_conversation_button_updates(conversation_id),
        )

    active_conversation_id = runtime.ensure_conversation(conversation_id or None, message)
    history = [
        *history,
        {"role": "user", "content": message},
        _pending_assistant_message(),
    ]
    return (
        history,
        {"mode": "pending"},
        active_conversation_id,
        gr.update(value="", interactive=False),
        message,
        gr.update(interactive=False),
        _saved_schedule_markdown(),
        *_conversation_button_updates(active_conversation_id),
    )


def finish_agent_response(
    pending_message: str,
    history: list[dict[str, Any]] | None,
    conversation_id: str | None,
) -> Any:
    history = history or []
    pending_message = (pending_message or "").strip()
    if not pending_message:
        yield (
            history,
            {},
            conversation_id or "",
            gr.update(interactive=True),
            "",
            gr.update(interactive=True),
            _saved_schedule_markdown(),
            *_conversation_button_updates(conversation_id),
        )
        return

    active_conversation_id = conversation_id or None
    for event in runtime.stream_agent(pending_message, active_conversation_id):
        if event.status_text:
            history = _replace_pending_status(history, event.status_text)
            yield (
                history,
                {"mode": "pending", "status": event.status_text},
                conversation_id or "",
                gr.update(interactive=False),
                pending_message,
                gr.update(interactive=False),
                _saved_schedule_markdown(),
                *_conversation_button_updates(conversation_id),
            )
        if event.result:
            history = _replace_pending_with_answer(history, event.result.answer)
            yield (
                history,
                event.result.trace,
                event.result.conversation_id,
                gr.update(interactive=True),
                "",
                gr.update(interactive=True),
                _saved_schedule_markdown(),
                *_conversation_button_updates(event.result.conversation_id),
            )
            return


def new_chat() -> tuple:
    return (_chat_notice(), {}, "", _saved_schedule_markdown(), *_conversation_button_updates(None))


def load_chat(conversation_id: str | None) -> tuple:
    if not conversation_id:
        return (_chat_notice(), "", _saved_schedule_markdown(), *_conversation_button_updates(None))
    return (
        _saved_chatbot_history(conversation_id),
        conversation_id,
        _saved_schedule_markdown(),
        *_conversation_button_updates(conversation_id),
    )


def archive_chat(conversation_id: str | None) -> tuple:
    runtime.archive_conversation(conversation_id)
    return (_chat_notice(), {}, "", _saved_schedule_markdown(), *_conversation_button_updates(None))


def delete_chat(conversation_id: str | None) -> tuple:
    if conversation_id:
        runtime.delete_conversation(conversation_id)
    return (_chat_notice(), {}, "", _saved_schedule_markdown(), *_conversation_button_updates(None))


def conversation_id_at(index: int) -> str:
    rows = _conversation_rows()
    if 0 <= index < len(rows):
        return rows[index]["conversation_id"]
    return ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Kanana Schedule Agent") as demo:
        conversation_id = gr.Textbox(value="", visible=False, elem_id="selected-conversation-id", container=False)
        pending_message = gr.State("")
        gr.HTML(
            f"""
            <div class="kanana-topbar">
              <div class="brand-lockup">
                <span>Smart Schedule Agent</span>
              </div>
            </div>
            """
        )
        with gr.Tabs(elem_id="main-tabs"):
            with gr.Tab("채팅"):
                with gr.Row(elem_id="kanana-shell"):
                    with gr.Column(scale=1, min_width=250, elem_classes=["sidebar"]):
                        new_btn = gr.Button("새 대화", elem_classes=["primary-action"])
                        gr.HTML("<div class='conversation-list-title'>대화</div>", container=False)
                        conversation_buttons = [
                            gr.Button(
                                "",
                                visible=False,
                                elem_classes=["conversation-list-item"],
                            )
                            for _ in range(MAX_CONVERSATION_BUTTONS)
                        ]
                        gr.HTML("<div class='conversation-list-title'>저장된 일정</div>", container=False)
                        saved_schedules = gr.Markdown(
                            value=_saved_schedule_markdown(),
                            show_label=False,
                            elem_id="saved-schedule-list",
                            elem_classes=["saved-schedule-list"],
                        )
                        archive_btn = gr.Button("현재 대화 보관", elem_classes=["ghost-action"])
                        delete_btn = gr.Button("저장된 대화 삭제", elem_classes=["danger-action"])
                    with gr.Column(scale=4, min_width=560, elem_classes=["chat-panel"]):
                        chatbot = gr.Chatbot(
                            value=_chat_notice(),
                            height=680,
                            show_label=False,
                            elem_id="kanana-chatbot",
                            placeholder="",
                        )
                        with gr.Row(elem_classes=["composer"]):
                            textbox = gr.Textbox(
                                placeholder="",
                                show_label=False,
                                lines=2,
                                elem_id="kanana-input",
                            )
                            send_btn = gr.Button("↑", elem_id="kanana-send", elem_classes=["send-button"])
            with gr.Tab("상세"):
                with gr.Row(elem_classes=["details-layout"]):
                    with gr.Column(scale=1, min_width=720, elem_classes=["detail-card", "trace-detail-card"]):
                        gr.HTML("<div class='trace-title'>마지막 에이전트 실행 Trace</div>")
                        trace_json = gr.JSON(
                            label="trace 페이로드",
                            value={},
                            elem_id="trace-json",
                            open=True,
                            min_height=620,
                            max_height=780,
                        )

        send_outputs = [
            chatbot,
            trace_json,
            conversation_id,
            textbox,
            pending_message,
            send_btn,
            saved_schedules,
            *conversation_buttons,
        ]
        finish_outputs = [
            chatbot,
            trace_json,
            conversation_id,
            textbox,
            pending_message,
            send_btn,
            saved_schedules,
            *conversation_buttons,
        ]
        send_btn.click(
            queue_user_message,
            inputs=[textbox, chatbot, conversation_id],
            outputs=send_outputs,
            queue=False,
        ).then(
            finish_agent_response,
            inputs=[pending_message, chatbot, conversation_id],
            outputs=finish_outputs,
            show_progress="hidden",
        )
        new_btn.click(new_chat, outputs=[chatbot, trace_json, conversation_id, saved_schedules, *conversation_buttons])
        archive_btn.click(
            archive_chat,
            inputs=[conversation_id],
            outputs=[chatbot, trace_json, conversation_id, saved_schedules, *conversation_buttons],
        )
        delete_btn.click(
            delete_chat,
            inputs=[conversation_id],
            outputs=[chatbot, trace_json, conversation_id, saved_schedules, *conversation_buttons],
            js=DELETE_CONVERSATION_CONFIRM_JS,
            queue=False,
        )
        for index, conversation_button in enumerate(conversation_buttons):
            conversation_button.click(
                lambda idx=index: conversation_id_at(idx),
                outputs=[conversation_id],
                show_progress="hidden",
            ).then(
                load_chat,
                inputs=[conversation_id],
                outputs=[chatbot, conversation_id, saved_schedules, *conversation_buttons],
                show_progress="hidden",
            )
        demo.load(
            lambda: (_saved_schedule_markdown(), *_conversation_button_updates(None)),
            outputs=[saved_schedules, *conversation_buttons],
        )
    return demo


if __name__ == "__main__":
    if not CONFIG.has_openai_key:
        print("주의: 프롬프트 기반 에이전트 채팅에는 .env의 PROXY_TOKEN이 필요합니다.")
    build_demo().launch(css_paths=[str(CSS_PATH)], head=ENTER_TO_SEND_HEAD)
