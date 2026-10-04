"""チャットテンプレート対応(2026-09-22、全タブ共通)のテスト。

tokenizer_utils.pyのテンプレート描画・フォールバック・停止トークン解決・特殊トークンの
エンコードを、実モデルなしで検証する(小さな自作トークナイザーと一時フォルダを使う)。
jinja2/tokenizersが無い環境では該当テストをスキップする。

実行: python python/tests/test_chat_template.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import tokenizer_utils as tu  # noqa: E402

# 一般的なChatML形式のテンプレート(Qwen2.5と同様、systemが無ければデフォルトを入れる)
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{% endif %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

LLAMA3_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _model_dir(tokenizer_config: dict | None, generation_config: dict | None = None) -> Path:
    d = Path(tempfile.mkdtemp())
    if tokenizer_config is not None:
        (d / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    if generation_config is not None:
        (d / "generation_config.json").write_text(json.dumps(generation_config), encoding="utf-8")
    return d


def test_select_chat_template_variants():
    assert tu.select_chat_template({"chat_template": "abc"}) == "abc"
    assert tu.select_chat_template({"chat_template": [{"name": "tool_use", "template": "t"}, {"name": "default", "template": "d"}]}) == "d"
    assert tu.select_chat_template({"chat_template": [{"name": "x", "template": "t"}]}) == "t"
    assert tu.select_chat_template({}) is None
    assert tu.select_chat_template({"chat_template": "  "}) is None


def test_detect_family():
    assert tu.detect_template_family(CHATML_TEMPLATE) == "chatml"
    assert tu.detect_template_family(LLAMA3_TEMPLATE) == "llama3"
    assert tu.detect_template_family("<start_of_turn>") == "gemma"
    assert tu.detect_template_family("[INST]") == "mistral"
    assert tu.detect_template_family("unknown") is None


def test_builtin_renderers():
    assert tu.render_builtin_chat_prompt("chatml", "hi") == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert tu.render_builtin_chat_prompt("chatml", "hi", "sys").startswith("<|im_start|>system\nsys<|im_end|>\n")
    assert tu.render_builtin_chat_prompt("llama3", "hi").startswith("<|begin_of_text|><|start_header_id|>user")
    assert tu.render_builtin_chat_prompt("gemma", "hi").endswith("<start_of_turn>model\n")
    assert tu.render_builtin_chat_prompt("mistral", "hi") == "<s>[INST] hi [/INST]"


def test_render_with_jinja_chatml():
    if not _has("jinja2"):
        print("skip (jinja2なし)")
        return
    d = _model_dir({"chat_template": CHATML_TEMPLATE, "eos_token": "<|im_end|>"})
    out = tu.render_chat_prompt(d, "こんにちは")
    assert out == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nこんにちは<|im_end|>\n<|im_start|>assistant\n"
    )


def test_render_with_jinja_llama3_bos_dict():
    if not _has("jinja2"):
        return
    d = _model_dir({"chat_template": LLAMA3_TEMPLATE, "bos_token": {"content": "<|begin_of_text|>"}})
    out = tu.render_chat_prompt(d, " hi ")
    assert out == (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_raise_exception_in_template_is_reported():
    if not _has("jinja2"):
        return
    d = _model_dir({"chat_template": "{{ raise_exception('only user/assistant') }}"})
    try:
        tu.render_chat_prompt(d, "x")
    except tu.ChatTemplateError as e:
        assert "only user/assistant" in str(e)
        return
    raise AssertionError("expected ChatTemplateError")


def test_missing_template_raises_and_is_tokenizer_error():
    d = _model_dir({"eos_token": "</s>"})
    try:
        tu.render_chat_prompt(d, "x")
    except tu.TokenizerLoadError as e:  # ChatTemplateErrorはTokenizerLoadErrorのサブクラス
        assert isinstance(e, tu.ChatTemplateError)
        return
    raise AssertionError("expected ChatTemplateError")


def test_fallback_when_jinja_unavailable(monkeypatch=None):
    d = _model_dir({"chat_template": CHATML_TEMPLATE})
    original = tu._render_with_jinja

    def no_jinja(*_a, **_k):
        raise ImportError("no jinja2")

    tu._render_with_jinja = no_jinja
    try:
        out = tu.render_chat_prompt(d, "hi")
    finally:
        tu._render_with_jinja = original
    assert out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


class _FakeVocabTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def token_to_id(self, token: str):
        return self.vocab.get(token)


def test_get_stop_token_ids():
    d = _model_dir({"eos_token": "<|im_end|>"}, {"eos_token_id": [151645, 151643]})
    tok = _FakeVocabTokenizer({"<|im_end|>": 151645, "<|endoftext|>": 151643, "<|im_start|>": 151644})
    assert tu.get_stop_token_ids(d, tok) == [151645, 151643]
    d2 = _model_dir({"eos_token": {"content": "<eos>"}}, {"eos_token_id": 1})
    tok2 = _FakeVocabTokenizer({"<eos>": 1, "<end_of_turn>": 107})
    assert tu.get_stop_token_ids(d2, tok2) == [1, 107]
    assert tu.get_stop_token_ids(_model_dir(None), _FakeVocabTokenizer({})) == []


def test_resolve_stop_token_ids_off_by_default():
    tok = _FakeVocabTokenizer({"<|im_end|>": 5})
    d = _model_dir({})
    assert tu.resolve_stop_token_ids({}, d, tok) is None
    assert tu.resolve_stop_token_ids({"use_chat_template": True}, d, None) is None
    assert tu.resolve_stop_token_ids({"use_chat_template": True}, d, tok) == [5]


def _tiny_tokenizer():
    """特殊トークン(<|im_start|>/<|im_end|>)を持つ小さなトークナイザー。BOSを自動付与する
    post-processorを付けて、add_special_tokens=Falseで二重付与が防げることも確認できるようにする。"""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders  # noqa: PLC0415

    vocab = {"[UNK]": 0, "<bos>": 1, "<|im_start|>": 2, "<|im_end|>": 3}
    for ch in "abcdefghijklmnopqrstuvwxyz\n":
        vocab[ch] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(["<bos>", "<|im_start|>", "<|im_end|>"])
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
    return tok


def test_encode_text_with_template_uses_special_tokens_once():
    if not (_has("jinja2") and _has("tokenizers")):
        print("skip (jinja2/tokenizersなし)")
        return
    tok = _tiny_tokenizer()
    d = _model_dir({"chat_template": CHATML_TEMPLATE})
    plain = tu.encode_text(tok, "hi")
    assert plain[0] == 1  # 従来どおり: post-processorのBOSが付く
    ids = tu.encode_text(tok, "hi", model_dir=d, use_chat_template=True)
    assert ids.count(2) == 3 and ids.count(3) == 2  # im_start×3(system/user/assistant)・im_end×2
    assert 1 not in ids  # add_special_tokens=Falseで自動BOSは付かない
    assert ids[0] == 2


def test_decode_tokens_shows_special_token_text():
    if not _has("tokenizers"):
        return
    tok = _tiny_tokenizer()
    assert tu.decode_tokens(tok, [2]) == "<|im_start|>"  # 単独の特殊トークンは表記を返す
    assert tu.decode_tokens(tok, [tok.token_to_id("h"), tok.token_to_id("i")]) == "hi"
    assert tu.decode_tokens(tok, [2, tok.token_to_id("h")]) == "h"  # 複数なら従来どおり除外


def test_encode_text_requires_model_dir():
    try:
        tu.encode_text(object(), "x", use_chat_template=True)
    except tu.ChatTemplateError:
        return
    raise AssertionError("expected ChatTemplateError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
