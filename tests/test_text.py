from types import SimpleNamespace

import pytest

from lecture_service.asr import merge_overlap
from lecture_service.summarizer import (
    QwenSummarizer,
    cleanup_language_changed,
    format_paragraphs_lossless,
    split_text_tokens,
    strip_thinking,
    summary_word_target,
)


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(value) for value in text]


def test_overlap_merges_punctuation_case_and_preserves_new_content():
    tail, count = merge_overlap("Первая тема. Машинное обучение важно!", "машинное обучение важно. Теперь перейдем к оценке.")
    assert tail == "Теперь перейдем к оценке."
    assert count == 3


def test_distant_repeated_content_is_not_deleted():
    previous = "Градиент — производная функции. Далее изучим матрицы и векторы."
    current = "Градиент — производная функции. Вернемся к этому определению."
    assert merge_overlap(previous, current) == (current, 0)


def test_overlap_tolerates_one_asr_spelling_error_but_keeps_new_sentence():
    previous = "Далее стохастический градиентный спуск обновляет параметры."
    current = "Стохастический градиентныи спуск обновляет параметры. Размер шага важен."
    assert merge_overlap(previous, current) == ("Размер шага важен.", 5)


def test_similar_but_different_boundary_statement_is_not_deleted():
    previous = "Модель с высокой точностью предсказывает параметры."
    current = "Модель с низкой точностью предсказывает параметры. Это другая модель."
    assert merge_overlap(previous, current) == (current, 0)


def test_single_word_repeat_is_preserved():
    assert merge_overlap("Обозначим это и", "и рассмотрим пример") == ("и рассмотрим пример", 0)


def test_token_split_preserves_every_character_and_last_part():
    text = ("Определение функции. Число 42 и формула x².\n\nСледующая тема — матрицы. " * 83) + "УНИКАЛЬНЫЙ КОНЕЦ"
    tokenizer = CharacterTokenizer()
    pieces = split_text_tokens(text, tokenizer, 100)
    assert len(pieces) > 20
    assert "".join(pieces) == text
    assert all(len(tokenizer.encode(piece)) <= 100 for piece in pieces)
    assert pieces[-1].endswith("УНИКАЛЬНЫЙ КОНЕЦ")


def test_no_whitespace_text_still_has_bounded_chunks():
    text = "абв" * 1000
    parts = split_text_tokens(text, CharacterTokenizer(), 80)
    assert "".join(parts) == text
    assert max(map(len, parts)) <= 80


def test_thinking_not_leaked_and_unfinished_thinking_is_error():
    assert strip_thinking("<think>private</think>\n## Итог") == "## Итог"
    with pytest.raises(RuntimeError, match="незавершённое"):
        strip_thinking("<think>never finishes")


def test_summarizer_processes_all_source_parts_and_keeps_partial_notes():
    settings = SimpleNamespace(summary_chunk_tokens=100, summary_max_new_tokens=80)
    summarizer = QwenSummarizer(settings)
    summarizer.tokenizer = CharacterTokenizer()
    summarizer.load = lambda: None
    seen_sources = []

    def generate(text, task, max_new_tokens):
        if "полную отредактированную" in task:
            seen_sources.append(text)
            return text.strip()
        return "### Тема\n- Идея."

    summarizer._generate = generate
    source = "Предложение лекции. " * 30 + "УНИКАЛЬНЫЙ КОНЕЦ"
    result = summarizer.process(source)
    assert "".join(seen_sources) == source
    assert result["metadata"]["summary_source_parts"] == len(seen_sources)
    assert f"## Часть {len(seen_sources)}" in result["summary"]
    assert "УНИКАЛЬНЫЙ КОНЕЦ" in result["transcript"]


def test_cleanup_dropping_numbers_preserves_authoritative_source():
    settings = SimpleNamespace(summary_chunk_tokens=500, summary_max_new_tokens=100)
    summarizer = QwenSummarizer(settings)
    summarizer.tokenizer = CharacterTokenizer()
    summarizer.load = lambda: None
    summarizer._generate = lambda text, task, limit: "Градиентный спуск обновляет параметры."
    source = "Градиентный спуск обновляет параметры за 42 итерации."
    result = summarizer.process(source)
    assert result["transcript"] == source
    assert len(result["metadata"]["cleanup_warnings"]) == 1


def test_wall_of_text_paragraph_fallback_preserves_every_original_character():
    sentence = "Число 3.1415, термин F1 и два  пробела сохраняются. "
    original = sentence * 24
    formatted = format_paragraphs_lossless(original)
    assert "\n\n" in formatted
    assert formatted.replace("\n\n", "") == original
    assert formatted.count("3.1415") == original.count("3.1415")
    assert formatted.split() == original.split()


def test_existing_semantic_paragraphs_and_short_text_are_unchanged():
    original = "Первое определение.\n\nВторое определение. " * 30
    assert format_paragraphs_lossless(original) == original
    assert format_paragraphs_lossless("Одна короткая фраза.") == "Одна короткая фраза."


def test_summary_length_target_adapts_to_source_and_is_capped():
    assert summary_word_target("слово " * 225) == 112
    assert summary_word_target("слово " * 40) == 20
    assert summary_word_target("слово " * 2000) == 180


def test_observed_193_word_note_is_rewritten_once_without_posthoc_truncation():
    summarizer = QwenSummarizer(SimpleNamespace(summary_max_new_tokens=900))
    source = "источник " * 224
    long_note = "повтор " * 193
    concise_note = "кратко " * 89 + "ПОСЛЕДНИЙ_ВЫВОД"
    answers = iter([long_note, concise_note])
    inputs = []

    def generate(text, task, max_new_tokens):
        inputs.append(text)
        return next(answers)

    summarizer._generate = generate
    note, target, compressed, warning = summarizer._make_note(source, lambda stage, fraction: None)
    assert note == concise_note
    assert note.endswith("ПОСЛЕДНИЙ_ВЫВОД")
    assert inputs == [source, source]
    assert target == 112 and compressed
    assert warning is None


def test_verbose_summary_is_preserved_with_warning_after_one_retry():
    summarizer = QwenSummarizer(SimpleNamespace(summary_max_new_tokens=900))
    calls = []

    def generate(text, task, max_new_tokens):
        calls.append(task)
        return "слишком длинно " * 100

    summarizer._generate = generate
    note, target, compressed, warning = summarizer._make_note(
        "источник " * 225, lambda stage, fraction: None)
    assert len(calls) == 2
    assert note == "слишком длинно " * 100
    assert target == 112 and compressed
    assert warning["target_words"] == 112
    assert warning["actual_words"] == 200
    assert warning["maximum_preferred_words"] == 140


def test_process_records_verbose_part_warning_without_failing_whole_lecture():
    settings = SimpleNamespace(summary_chunk_tokens=8000, summary_max_new_tokens=900)
    summarizer = QwenSummarizer(settings)
    summarizer.tokenizer = CharacterTokenizer()
    summarizer.load = lambda: None
    source = "Источник содержит все определения и заключительный важный риск. " * 25
    long_note = "Содержательное определение и важный итог. " * 40
    calls = []

    def generate(text, task, max_new_tokens):
        calls.append(task)
        return source if "полную отредактированную" in task else long_note

    summarizer._generate = generate
    result = summarizer.process(source)
    assert len(calls) == 3  # Cleanup, note, and exactly one rewrite.
    assert result["summary"].endswith(long_note)
    assert result["metadata"]["summary_compression_calls"] == 1
    assert len(result["metadata"]["summary_length_warnings"]) == 1
    warning = result["metadata"]["summary_length_warnings"][0]
    assert warning["part"] == 1
    assert warning["actual_words"] > warning["maximum_preferred_words"]


def test_process_formats_model_wall_of_text_without_rewriting_it():
    settings = SimpleNamespace(summary_chunk_tokens=8000, summary_max_new_tokens=900)
    summarizer = QwenSummarizer(settings)
    summarizer.tokenizer = CharacterTokenizer()
    summarizer.load = lambda: None
    source = "Первое определение содержит число 42 и термин. " * 30
    tasks = []

    def generate(text, task, max_new_tokens):
        tasks.append(task)
        return source if "полную отредактированную" in task else "### Определение\n- Число 42."

    summarizer._generate = generate
    result = summarizer.process(source)
    assert result["transcript"].replace("\n\n", "") == source
    assert result["metadata"]["paragraph_formatting_fallback_parts"] == [1]
    assert "Сохрани исходный язык" in tasks[0]


@pytest.mark.parametrize("source,translated", [
    ("Each neuron stores 42 values and the next layer combines these inputs. " * 5,
     "Каждый нейрон хранит 42 значения и следующий слой объединяет эти входные данные. " * 5),
    ("Каждый нейрон хранит 42 значения и следующий слой объединяет эти входные данные. " * 5,
     "Each neuron stores 42 values and the next layer combines these inputs. " * 5),
])
def test_cleanup_translation_is_rejected_even_when_numbers_and_length_survive(source, translated):
    source = source.strip()
    settings = SimpleNamespace(summary_chunk_tokens=8000, summary_max_new_tokens=900)
    summarizer = QwenSummarizer(settings)
    summarizer.tokenizer = CharacterTokenizer()
    summarizer.load = lambda: None
    summarizer._generate = lambda text, task, limit: (
        translated if "полную отредактированную" in task else "### Итог\n- Нейрон хранит значения.")
    assert len(translated.split()) >= len(source.split()) * 0.65
    assert "42" in translated
    result = summarizer.process(source)
    assert result["transcript"] == source
    assert result["metadata"]["cleanup_warnings"] == [
        "Part 1: source text retained after cleanup validation; source language changed"]


def test_cleanup_accepts_mixed_technical_terms_and_auto_infers_source_script():
    russian = "Функция активации ReLU возвращает max(0, x), sigmoid ограничивает значения. " * 4
    edited_russian = "Функция активации ReLU вычисляет max(0, x), sigmoid ограничивает значения. " * 4
    english = "The ReLU function returns max(0, x), while sigmoid limits values. " * 4
    assert not cleanup_language_changed(russian, edited_russian)
    assert not cleanup_language_changed(english, english.replace("returns", "computes"))
    assert cleanup_language_changed(english, russian)
    assert cleanup_language_changed(russian, english)
    # A bilingual fragment does not identify a dominant source language.
    bilingual = "Hello world. Привет мир. " * 4
    assert not cleanup_language_changed(bilingual, "Hello world. " * 8)
    assert not cleanup_language_changed("ReLU 42", "Релу 42")
