# tests/unit/test_chat_template_files.py
"""chat_template_files checks that need no model directory."""


_CHATML = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
           "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def test_prefix_stability_flags_what_costs_every_turn_its_cache():
    """Plan W3's lint. A generation prompt sharing nothing with the next
    turn's render of the reply (the audit's Qwen3.8 sidecar leaked a newline
    there), and a history that re-renders differently, both fail; a template
    whose generation prompt pre-fills a block the history then drops (Qwen3.5,
    gemma-4) keeps its role header and passes."""
    from heylook_llm.chat_template_files import prefix_stability

    assert prefix_stability(_CHATML)[0] is True
    prefilled = _CHATML.replace("<|im_start|>assistant\n{% endif %}",
                                "<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}")
    assert prefix_stability(prefilled)[0] is True
    # (an expression, not a literal newline: trim_blocks eats a newline right
    # after a block tag, the very whitespace rule the real defect tripped on)
    leaked = _CHATML.replace("{% if add_generation_prompt %}<|im_start|>",
                             "{% if add_generation_prompt %}{{ '\\n' }}<|im_start|>")
    stable, why = prefix_stability(leaked)
    assert stable is False and "shares nothing" in why
    rewrites = _CHATML.replace("{{ m.content }}", "{{ m.content }}{% if loop.last %} (brief){% endif %}")
    stable, why = prefix_stability(rewrites)
    assert stable is False and "differently" in why
    assert prefix_stability(None)[0] is None


def test_template_sources_mark_download_state_and_in_force(tmp_path):
    """Plan W3: each copy says where it came from. A file whose bytes still
    hash to huggingface_hub's download record (a small file's etag is its git
    blob SHA-1) is downloaded; an edit makes it modified; no record is no
    record. The in-force copy is the one whose body the ladder resolved."""
    import hashlib

    from heylook_llm.chat_template_files import template_sources

    body = "{% for m in messages %}{{ m.content }}{% endfor %}"
    (tmp_path / "chat_template.jinja").write_text(body)
    record = tmp_path / ".cache" / "huggingface" / "download"
    record.mkdir(parents=True)
    blob = hashlib.sha1(b"blob %d\0" % len(body.encode()) + body.encode()).hexdigest()
    (record / "chat_template.jinja.metadata").write_text(f"abc123\n{blob}\n0\n")
    (tmp_path / "chat_template.heylook.jinja").write_text(body + "!")
    cfg = {"model_path": str(tmp_path)}

    rows = {r["source"]: r for r in template_sources("m", "mlx", cfg, body + "!")}
    assert rows["heylook override"]["in_force"] and rows["heylook override"]["provenance"] == "heylook override"
    assert rows["chat_template.jinja"]["provenance"] == "downloaded" and rows["chat_template.jinja"]["download_commit"] == "abc123"
    assert not rows["chat_template.jinja"]["in_force"]

    (tmp_path / "chat_template.jinja").write_text(body + " edited")
    rows = {r["source"]: r for r in template_sources("m", "mlx", cfg, None)}
    assert rows["chat_template.jinja"]["provenance"] == "modified since download"
    (record / "chat_template.jinja.metadata").unlink()
    assert {r["source"]: r for r in template_sources("m", "mlx", cfg, None)}["chat_template.jinja"]["provenance"] \
        == "no download record"
