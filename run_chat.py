import argparse
import threading
import time
from typing import List

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input, RichLog, Static

from src.fuzzy_sentic.conversation import (
    SPEAKERS,
    Classification,
    ConversationResult,
    Message,
    classify_conversation,
)
from src.fuzzy_sentic.interfaces import SenticPipelineInterface

SPEAKER_COLORS = {SPEAKERS[0]: "cyan", SPEAKERS[1]: "magenta"}
LABEL_COLORS = {"positiva": "green", "negativa": "red", "neutra": "yellow"}
BAR_WIDTH = 21


def polarity_bar(polarity: float) -> str:
    marker = round((polarity + 1.0) / 2.0 * (BAR_WIDTH - 1))
    center = BAR_WIDTH // 2
    color = "green" if polarity > 0 else "red"
    cells = []
    for i in range(BAR_WIDTH):
        if i == marker:
            cells.append("[bold white]●[/]")
        elif min(center, marker) <= i <= max(center, marker):
            cells.append(f"[{color}]█[/]")
        else:
            cells.append("[dim]░[/]")
    return "−1 " + "".join(cells) + " +1"


def describe(c: Classification) -> str:
    color = LABEL_COLORS[c.label]
    return f"[{color}]{c.label}[/] ({c.polarity:+.2f})"


def render_panel(
    result: ConversationResult | None, model_name: str, busy: bool, elapsed: float | None = None
) -> str:
    lines = []
    if result is None:
        lines.append("[dim]Envie uma mensagem para começar.[/]")
    else:
        conv = result.conversation
        color = LABEL_COLORS[conv.label]
        lines += [
            f"[b]Conversa:[/] [b {color}]{conv.label.upper()}[/]  ({conv.polarity:+.2f})",
            polarity_bar(conv.polarity),
            "",
        ]
        for speaker, classification in result.by_speaker.items():
            lines.append(f"[{SPEAKER_COLORS[speaker]}]{speaker}[/]: {describe(classification)}")
        if result.last_message is not None:
            lines.append(f"Última msg: {describe(result.last_message)}")
        lines += ["", "[b]Emoções dominantes:[/]"]
        if conv.emotions:
            lines += [f"  {name} ({degree:.2f})" for name, degree in conv.emotions[:6]]
        else:
            lines.append("  [dim]nenhuma acima do α-cut[/]")
    lines += ["", f"[dim]Modelo: {model_name}[/]"]
    if elapsed is not None:
        lines.append(f"[dim]Tempo de processamento: {elapsed * 1000:.0f} ms[/]")
    if busy:
        lines.append("[dim italic]analisando…[/]")
    return "\n".join(lines)


class ChatApp(App):
    CSS = """
    #conversa { width: 3fr; border: round $primary; border-title-color: $primary; }
    #painel { width: 2fr; border: round $secondary; border-title-color: $secondary; padding: 0 1; }
    #mensagens { height: 1fr; }
    #entrada { dock: bottom; }
    """
    BINDINGS = [
        Binding("ctrl+l", "clear", "Limpar conversa"),
        Binding("ctrl+q", "quit", "Sair"),
    ]

    def __init__(self, pipeline: SenticPipelineInterface) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.model_name = pipeline.__class__.__name__
        self.messages: List[Message] = []
        self.turn = 0
        self.result: ConversationResult | None = None
        self.elapsed: float | None = None
        self._lock = threading.Lock()

    @property
    def speaker(self) -> str:
        return SPEAKERS[self.turn % 2]

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="conversa"):
                yield RichLog(id="mensagens", markup=True, wrap=True)
                yield Input(id="entrada")
            yield Static(id="painel")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#conversa").border_title = "Conversa"
        self.query_one("#painel").border_title = "Classificação"
        self._refresh_prompt()
        self._refresh_panel(busy=False)
        self.query_one("#entrada", Input).focus()

    def _refresh_prompt(self) -> None:
        self.query_one("#entrada", Input).placeholder = f"{self.speaker} > digite e tecle Enter"

    def _refresh_panel(self, busy: bool) -> None:
        self.query_one("#painel", Static).update(render_panel(self.result, self.model_name, busy, self.elapsed))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        message = Message(self.speaker, text)
        self.messages.append(message)
        color = SPEAKER_COLORS[message.speaker]
        self.query_one("#mensagens", RichLog).write(f"[b {color}]{message.speaker}:[/] {text}")
        self.turn += 1
        self._refresh_prompt()
        self._refresh_panel(busy=True)
        self.classify(list(self.messages))

    @work(thread=True, exclusive=True)
    def classify(self, messages: List[Message]) -> None:
        with self._lock:
            start = time.perf_counter()
            result = classify_conversation(self.pipeline, messages)
            elapsed = time.perf_counter() - start
        self.call_from_thread(self._show_result, result, len(messages), elapsed)

    def _show_result(self, result: ConversationResult, n_messages: int, elapsed: float) -> None:
        # Descarta resultados atrasados (de antes da última mensagem ou de antes de limpar)
        if n_messages != len(self.messages):
            return
        self.result = result
        self.elapsed = elapsed
        self._refresh_panel(busy=False)

    def action_clear(self) -> None:
        self.messages.clear()
        self.turn = 0
        self.result = None
        self.elapsed = None
        self.query_one("#mensagens", RichLog).clear()
        self._refresh_prompt()
        self._refresh_panel(busy=False)


def load_pipeline(name: str) -> SenticPipelineInterface:
    if name == "baseline":
        from src.fuzzy_sentic.models.baseline import LoiaSenatoreBaseline
        return LoiaSenatoreBaseline(alpha_cut=0.5)
    from src.fuzzy_sentic.models.neosentic import NeoSenticFuzzy
    return NeoSenticFuzzy(alpha_cut=0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat com classificação de sentimento em tempo real.")
    parser.add_argument("--modelo", choices=["neo", "baseline"], default="neo")
    args = parser.parse_args()
    print("Carregando modelo...")
    ChatApp(load_pipeline(args.modelo)).run()


if __name__ == "__main__":
    main()
