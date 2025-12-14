from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Tuple

import gradio as gr

from ..graph.plan_graph import (
    build_graph, 
    GraphState, 
    solve_with_budget,
    set_thinking_callback,
    get_thinking_log,
    clear_thinking_log,
)


def build_compiled_app(config_path: str | None = None) -> Any:
    """
    Build and compile the orchestrator graph once.
    """
    default_cfg = Path(__file__).resolve().parents[2] / "configs" / "openrouter.yaml"
    cfg_path = Path(config_path) if config_path else default_cfg
    g = build_graph(cfg_path.as_posix())
    return g.compile()


def answer_question_with_thinking(
    question: str, 
    config_path: str, 
    timeout_s: float = 300.0,
    web_enabled: bool = False,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Invoke with thinking log and return (answer, thinking_log).
    """
    thinking_updates: List[Tuple[str, str]] = []
    
    def thinking_callback(stage: str, content: str) -> None:
        thinking_updates.append((stage, content))
    
    out = solve_with_budget(
        problem=question, 
        config_path=config_path, 
        timeout_s=timeout_s,
        thinking_callback=thinking_callback,
        web_enabled=web_enabled,
    )
    ans = getattr(out, "final_answer", "") or ""
    return str(ans).strip(), thinking_updates


def format_thinking_log(thinking_log: List[Tuple[str, str]]) -> str:
    """Format thinking log for display."""
    if not thinking_log:
        return "*No thinking log available*"
    
    lines = []
    for stage, content in thinking_log:
        stage_display = stage.replace("_", " ").title()
        
        emoji = "🔄"
        if "start" in stage:
            emoji = "🚀"
        elif "complete" in stage:
            emoji = "✅"
        elif "error" in stage:
            emoji = "❌"
        elif "fallback" in stage or "retry" in stage:
            emoji = "🔁"
        elif "timeout" in stage:
            emoji = "⏱️"
        elif "phase" in stage:
            emoji = "📋"
        elif "worker" in stage:
            emoji = "⚙️"
        elif "model" in stage:
            emoji = "🤖"
        
        # For multi-line payloads (e.g., RAG chunk lists, timeline reasoning), render content on a new line.
        c = "" if content is None else str(content)
        if "\n" in c or stage.startswith("rag_") or stage == "rag_chunks" or stage.startswith("timeline_"):
            lines.append(f"{emoji} **{stage_display}**:\n{c}")
        else:
            lines.append(f"{emoji} **{stage_display}**: {c}")
    
    return "\n\n".join(lines)


def make_ui(app: Any, config_path: str) -> gr.Blocks:
    """
    Construct a Gradio interface with thinking panel.
    """
    custom_css = """
    <style>
    body { 
        background: radial-gradient(1200px 600px at 20% 0%, #0ea5e922, transparent 60%),
                    radial-gradient(1200px 600px at 80% 0%, #a855f722, transparent 60%),
                    linear-gradient(180deg, #0b1020, #0b0f18); 
    }
    .gradio-container { max-width: 1400px !important; margin: auto; }
    .thinking-panel { 
        background: #0f172a !important; 
        border: 1px solid #1e293b; 
        border-radius: 8px;
        padding: 12px;
        font-size: 13px;
        line-height: 1.6;
        max-height: 600px;
        overflow-y: auto;
    }
    .main-title { 
        text-align: center; 
        background: linear-gradient(90deg, #0ea5e9, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 1.5em;
    }
    </style>
    """

    with gr.Blocks(title="MAS Orchestrator") as demo:
        gr.HTML(custom_css)
        gr.HTML("<h1 class='main-title'>🧠 Multi-Agent Reasoning System</h1>")
        gr.HTML("<p class='subtitle'>Ask math, logic, humanities, and knowledge questions. Watch the agents think in real-time.</p>")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask any question (math, logic, humanities, science)...",
                        label="Your Question",
                        scale=4,
                        show_label=False,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", size="sm")
                    web_toggle = gr.Checkbox(
                        value=False,
                        label="Enable Web (search + fetch page text)",
                        info="When off, the system will not use DuckDuckGo or fetch URLs.",
                        scale=2,
                    )
                    timeout_slider = gr.Slider(
                        minimum=60, 
                        maximum=600, 
                        value=300, 
                        step=30,
                        label="Timeout (seconds)",
                        scale=2,
                    )
                
                gr.Examples(
                    examples=[
                        "What is 12 * 13?",
                        "Who wrote The Hobbit and what year was it published?",
                        "Explain the significance of the French Revolution.",
                        "Solve: If a train travels at 60 mph for 2.5 hours, how far does it go?",
                        "What causes the seasons on Earth?",
                        "Evaluate: 3^4 + 2^5",
                        "Compare and contrast democracy and authoritarianism.",
                    ],
                    inputs=msg,
                    label="Example Questions",
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Agent Thinking Process")
                thinking_output = gr.Markdown(
                    value="*Ask a question to see the agents' thinking process...*",
                )
                
                with gr.Accordion("ℹ️ How It Works", open=False):
                    gr.Markdown("""
**The Multi-Agent Reasoning System uses:**

1. **Supervisor Agent** - Decomposes your question into subtasks
2. **Math Worker** - Handles arithmetic and calculations  
3. **QA Worker** - Handles factual knowledge and humanities
4. **Logic Worker** - Handles reasoning and deduction
5. **Verifier** - Double-checks numeric answers

**The thinking panel shows:**
- 🚀 Start events
- 🤖 Model queries
- ⚙️ Worker dispatches
- 🔁 Fallback attempts
- ✅ Completions
- ❌ Errors (if any)
                    """)
        
        def respond(message: str, history: list, timeout: float, web_enabled: bool):
            if not message.strip():
                return history, "*Please enter a question*"
            
            # Gradio 6.x uses dict format: {"role": "user/assistant", "content": "..."}
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "⏳ Processing..."}
            ]
            yield history, "🔄 **Starting analysis...**"
            
            answer, thinking_log = answer_question_with_thinking(
                message, 
                config_path=config_path, 
                timeout_s=timeout,
                web_enabled=web_enabled,
            )
            
            # Update the last assistant message with the answer
            history[-1] = {"role": "assistant", "content": answer}
            thinking_display = format_thinking_log(thinking_log)
            
            yield history, thinking_display
        
        def clear_chat():
            return [], "*Ask a question to see the agents' thinking process...*"
        
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, timeout_slider, web_toggle],
            outputs=[chatbot, thinking_output],
        ).then(
            lambda: "",
            outputs=msg,
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot, timeout_slider, web_toggle],
            outputs=[chatbot, thinking_output],
        ).then(
            lambda: "",
            outputs=msg,
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, thinking_output],
        )
        
        gr.Markdown(
            """
---
**Tip:** Set `OPENROUTER_API_KEY` in your environment. 
Config: `apps/mas/configs/openrouter.yaml`
            """
        )
    
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS Orchestrator Chat UI")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "configs" / "openrouter.yaml"),
        help="Path to OpenRouter config YAML",
    )
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--server-port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    app = build_compiled_app(args.config)
    ui = make_ui(app, args.config)
    ui.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
