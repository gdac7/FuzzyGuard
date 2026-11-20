import asyncio

import numpy as np
import pytest
from textual.widgets import Input, Static

from run_chat import ChatApp
from src.fuzzy_sentic.conversation import (
    Message,
    as_sentence,
    classify_conversation,
    dominant_emotions,
    polarity_label,
)
from src.fuzzy_sentic.interfaces import SenticPipelineInterface


class KeywordPipeline(SenticPipelineInterface):
    """Modelo falso: +0.8 por "love", -0.8 por "hate"; Joy ativa com "love"."""

    def __init__(self, alpha_cut: float = 0.5) -> None:
        super().__init__(alpha_cut=alpha_cut)
        self.calls = []

    def analyze_text(self, text):
        self.calls.append(text)
        words = text.lower().split()
        score = 0.8 * sum("love" in w for w in words) - 0.8 * sum("hate" in w for w in words)
        table = np.zeros((6, 4))
        if any("love" in w for w in words):
            table[1, 0] = 0.9
        return float(np.clip(score, -1, 1)), table


def test_as_sentence_adds_final_period():
    assert as_sentence("  hello there ") == "hello there."
    assert as_sentence("really?") == "really?"


@pytest.mark.parametrize("p, label", [(0.5, "positiva"), (-0.5, "negativa"), (0.05, "neutra")])
def test_polarity_label(p, label):
    assert polarity_label(p) == label


def test_dominant_emotions_respects_alpha_cut():
    table = np.zeros((6, 4))
    table[1, 0], table[4, 3] = 0.9, 0.3
    assert dominant_emotions(table, 0.5) == [("Joy", 0.9)]


def test_classify_conversation_by_speaker():
    pipeline = KeywordPipeline()
    messages = [Message("Pessoa A", "I love it"), Message("Pessoa B", "I hate it")]
    result = classify_conversation(pipeline, messages)
    assert result.by_speaker["Pessoa A"].label == "positiva"
    assert result.by_speaker["Pessoa B"].label == "negativa"
    assert result.last_message.label == "negativa"
    assert "I love it. I hate it." in pipeline.calls


def test_chat_app_alternates_speakers_and_updates_panel():
    async def scenario():
        app = ChatApp(KeywordPipeline())
        async with app.run_test() as pilot:
            entrada = app.query_one("#entrada", Input)
            assert "Pessoa A" in entrada.placeholder

            await pilot.click("#entrada")
            await pilot.press(*"I love you", "enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Pessoa B" in entrada.placeholder
            assert "POSITIVA" in str(app.query_one("#painel", Static).render())
            assert "Tempo de processamento" in str(app.query_one("#painel", Static).render())

            await pilot.press(*"I hate hate this", "enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert [m.speaker for m in app.messages] == ["Pessoa A", "Pessoa B"]
            assert "NEGATIVA" in str(app.query_one("#painel", Static).render())

            await pilot.press("ctrl+l")
            await pilot.pause()
            assert app.messages == [] and "Pessoa A" in entrada.placeholder

    asyncio.run(scenario())
