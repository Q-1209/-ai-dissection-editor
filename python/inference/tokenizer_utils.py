"""
tokenizer_utils.py

トークナイズ(テキスト <-> トークンID)だけは、Hugging Face製の軽量ライブラリ
`tokenizers` を例外的に使用する。理由: BPE(Byte Pair Encoding)の完全自前実装は
Phase1のコア価値(ノードを自由に繋ぎ替えて実行する)とは無関係な労力が大きく、
`tokenizers`自体はトークナイズ専用の小さなライブラリで、モデルの推論ロジックは
一切含まないため、「推論エンジンは自前実装」という方針への影響は小さいと判断した
(詳細はプロジェクトのREADME参照)。

対応フォーマット: HuggingFace形式の tokenizer.json ファイル。

2026-09-22追加: チャットテンプレート対応(全タブ共通、漣さんとの確認事項)。
Instructモデルは「<|im_start|>user ... <|im_end|><|im_start|>assistant」のような
会話形式に包まれた入力を前提に学習されているため、生のテキストを渡すと「応答」ではなく
「文章の続き書き」モードで動く(Sycophancy Reducer実機検証で判明、claude-notes.md参照)。
use_chat_template=Trueのとき、モデルフォルダの tokenizer_config.json にある
chat_template(Jinja2テンプレート)でユーザー入力を包んでからエンコードする。
- Jinja2が使える場合(torchの依存ライブラリなので通常は入っている): モデル付属の
  テンプレートをそのまま描画する(transformersのapply_chat_templateと同じ考え方)
- Jinja2が無い場合: 代表的な形式(ChatML/Llama3/Gemma/Mistral)を組み込みで描画する
  フォールバック。テンプレート文字列中の特徴的な記号から形式を判定する
transformersライブラリには依存しない(プロジェクト方針どおり)。
"""

from __future__ import annotations

import json
from pathlib import Path
from .i18n import tr


class TokenizerLoadError(Exception):
    """tokenizer.jsonが見つからない、またはtokenizersライブラリが未インストールの場合。"""


class ChatTemplateError(TokenizerLoadError):
    """チャットテンプレートが見つからない・描画できない場合(2026-09-22追加)。

    TokenizerLoadErrorのサブクラスにしているのは、既存の各エントリポイントの
    except TokenizerLoadError節でそのままGUIにエラー表示されるようにするため。
    """


def load_tokenizer(model_dir: Path):
    """モデルディレクトリから tokenizer.json を読み込む。

    戻り値の型は tokenizers.Tokenizer (実行時にのみimportするため、ここでは
    型ヒントを省略している)。
    """
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise TokenizerLoadError(
            tr(f"tokenizer.json が見つかりません: {tokenizer_path}。"
            f"このモデルディレクトリにHuggingFace形式のトークナイザーファイルが"
            f"含まれているか確認してください。", f"tokenizer.json not found: {tokenizer_path}. " f"Check that the model folder contains a Hugging Face " f"tokenizer file.")
        )

    try:
        from tokenizers import Tokenizer  # noqa: PLC0415
    except ImportError as e:
        raise TokenizerLoadError(
            tr("tokenizersライブラリがインストールされていません。"
            "`pip install tokenizers` を実行してください。", "The tokenizers library is not installed. " "Run `pip install tokenizers`.")
        ) from e

    return Tokenizer.from_file(str(tokenizer_path))


# ---------------------------------------------------------------------------
# チャットテンプレート(2026-09-22追加)
# ---------------------------------------------------------------------------

def _token_content(value) -> str | None:
    """tokenizer_config.jsonのbos_token/eos_tokenは文字列または{"content": ...}の辞書。"""
    if value is None:
        return None
    if isinstance(value, dict):
        content = value.get("content")
        return content if isinstance(content, str) else None
    if isinstance(value, str):
        return value
    return None


def load_tokenizer_config(model_dir: Path) -> dict:
    """tokenizer_config.jsonを読み込む(無ければ空辞書)。"""
    path = Path(model_dir) / "tokenizer_config.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_chat_template(tokenizer_config: dict) -> str | None:
    """tokenizer_config.jsonからチャットテンプレート文字列を取り出す。

    chat_templateは文字列、または[{"name": "default", "template": "..."}, ...]の
    リスト形式(複数テンプレートを持つモデル)のどちらか。リストの場合は"default"を優先する。
    """
    template = tokenizer_config.get("chat_template")
    if isinstance(template, str) and template.strip():
        return template
    if isinstance(template, list):
        named = {t.get("name"): t.get("template") for t in template if isinstance(t, dict)}
        chosen = named.get("default") or next((v for v in named.values() if v), None)
        if isinstance(chosen, str) and chosen.strip():
            return chosen
    return None


def detect_template_family(template: str | None) -> str | None:
    """テンプレート文字列中の特徴的な記号から、代表的な会話形式を判定する(フォールバック用)。"""
    if not template:
        return None
    if "<|im_start|>" in template:
        return "chatml"  # Qwen系など
    if "<|start_header_id|>" in template:
        return "llama3"
    if "<start_of_turn>" in template:
        return "gemma"
    if "[INST]" in template:
        return "mistral"
    return None


def render_builtin_chat_prompt(
    family: str, user_text: str, system_prompt: str | None = None, bos_token: str | None = None
) -> str:
    """Jinja2が使えない場合のフォールバック: 代表的な会話形式を組み込みで描画する。

    いずれも「ユーザーの発話1つ + アシスタントの応答開始位置まで」(add_generation_prompt=True
    相当)を返す。モデル付属テンプレートが自動で入れるデフォルトのシステムプロンプト
    (Qwen2.5の"You are Qwen..."等)までは再現しないため、Jinja2で描画した場合と
    完全一致はしない(実用上の差は小さい想定)。
    """
    if family == "chatml":
        prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
        return f"{prefix}<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    if family == "llama3":
        bos = bos_token or "<|begin_of_text|>"
        prefix = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            if system_prompt
            else ""
        )
        return (
            f"{bos}{prefix}<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if family == "gemma":
        bos = bos_token or "<bos>"
        # Gemmaはsystemロールを持たないため、システムプロンプトはユーザー発話の先頭に連結する
        body = f"{system_prompt}\n\n{user_text}" if system_prompt else user_text
        return f"{bos}<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"
    if family == "mistral":
        bos = bos_token or "<s>"
        body = f"{system_prompt}\n\n{user_text}" if system_prompt else user_text
        return f"{bos}[INST] {body} [/INST]"
    raise ChatTemplateError(tr(f"未対応のチャットテンプレート形式です: {family}", f"Unsupported chat template format: {family}"))


def _render_with_jinja(template: str, messages: list[dict], bos_token: str | None, eos_token: str | None) -> str:
    """モデル付属のJinja2テンプレートを描画する(transformersのapply_chat_templateと同じ設定)。

    Jinja2が無い場合はImportErrorをそのまま投げる(呼び出し元でフォールバックする)。
    """
    from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: PLC0415

    def raise_exception(message):
        raise ChatTemplateError(tr(f"チャットテンプレートがエラーを返しました: {message}", f"The chat template raised an error: {message}"))

    def tojson(value, indent=None):
        return json.dumps(value, ensure_ascii=False, indent=indent)

    def strftime_now(fmt):
        from datetime import datetime  # noqa: PLC0415

        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    compiled = env.from_string(template)
    return compiled.render(
        messages=messages,
        add_generation_prompt=True,
        bos_token=bos_token or "",
        eos_token=eos_token or "",
        tools=None,
    )


def render_chat_prompt(model_dir: Path, user_text: str, system_prompt: str | None = None) -> str:
    """ユーザー入力をモデルのチャットテンプレートで包んだ文字列を返す。

    テンプレートが無い(ベースモデル等)場合はChatTemplateError。
    """
    config = load_tokenizer_config(Path(model_dir))
    template = select_chat_template(config)
    if template is None:
        raise ChatTemplateError(
            tr("このモデルの tokenizer_config.json に chat_template がありません"
            "(Instructモデルではなくベースモデルの可能性があります)。"
            "「チャットテンプレート適用」をオフにして実行してください。", "This model's tokenizer_config.json has no chat_template " "(it may be a base model rather than an Instruct model). " "Turn off \"Use chat template\" and run again.")
        )
    bos_token = _token_content(config.get("bos_token"))
    eos_token = _token_content(config.get("eos_token"))

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    try:
        return _render_with_jinja(template, messages, bos_token, eos_token)
    except ImportError:
        family = detect_template_family(template)
        if family is None:
            raise ChatTemplateError(
                tr("Jinja2ライブラリが無く、このモデルのチャットテンプレート形式も自動判定できませんでした。"
                "`pip install jinja2` を実行してください。", "Jinja2 is not installed and this model's chat template format could not be detected. " "Run `pip install jinja2`.")
            ) from None
        return render_builtin_chat_prompt(family, user_text, system_prompt, bos_token)
    except ChatTemplateError:
        raise
    except Exception as e:  # noqa: BLE001 (Jinja2の構文エラー等)
        family = detect_template_family(template)
        if family is None:
            raise ChatTemplateError(tr(f"チャットテンプレートの描画に失敗しました: {e}", f"Failed to render the chat template: {e}")) from e
        return render_builtin_chat_prompt(family, user_text, system_prompt, bos_token)


# 生成停止に使う代表的な「発話終了」トークン(語彙に存在するものだけを採用する)
_KNOWN_END_OF_TURN_TOKENS = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|end_of_text|>"]


def get_stop_token_ids(model_dir: Path, tokenizer) -> list[int]:
    """チャットテンプレート適用時に、生成を止めるべきトークンIDの一覧を返す(2026-09-22追加)。

    generation_config.jsonのeos_token_id(整数またはリスト)、tokenizer_config.jsonの
    eos_token、代表的な発話終了トークン(語彙に存在するもの)を合わせて重複なく返す。
    tokenizerはtoken_to_id(str) -> int | None を持つオブジェクト(tokenizers.Tokenizer)。
    """
    ids: list[int] = []

    def add(tid) -> None:
        if isinstance(tid, int) and tid not in ids:
            ids.append(tid)

    gen_config_path = Path(model_dir) / "generation_config.json"
    if gen_config_path.exists():
        with open(gen_config_path, encoding="utf-8") as f:
            gen_config = json.load(f)
        eos = gen_config.get("eos_token_id")
        if isinstance(eos, list):
            for tid in eos:
                add(tid)
        else:
            add(eos)

    eos_text = _token_content(load_tokenizer_config(Path(model_dir)).get("eos_token"))
    candidates = ([eos_text] if eos_text else []) + _KNOWN_END_OF_TURN_TOKENS
    for token in candidates:
        add(tokenizer.token_to_id(token))
    return ids


def encode_text(
    tokenizer, text: str, *, model_dir: Path | None = None, use_chat_template: bool = False
) -> list[int]:
    """テキストをトークンID列に変換する。

    use_chat_template=True(2026-09-22追加)の場合、model_dirのチャットテンプレートで
    包んでからエンコードする。テンプレート自体がBOS等の特殊トークンを含むため、
    その場合はトークナイザー側の特殊トークン自動付与(add_special_tokens)を無効にして
    二重付与を防ぐ。
    """
    if use_chat_template:
        if model_dir is None:
            raise ChatTemplateError(tr("use_chat_template=True の場合は model_dir が必要です。", "model_dir is required when use_chat_template=True."))
        prompt = render_chat_prompt(Path(model_dir), text)
        return tokenizer.encode(prompt, add_special_tokens=False).ids
    encoding = tokenizer.encode(text)
    return encoding.ids


def decode_tokens(tokenizer, token_ids: list[int]) -> str:
    """トークンID列をテキストに変換する。

    1トークンだけを変換して空文字列になった場合(<|im_start|>等の特殊トークンは
    デコード時に除外されるため)は、トークン表記そのもの(id_to_token)を返す。
    トークン単位の表示(Sensitivity Map・Token Journey等)で特殊トークンが空欄に
    ならないようにするため(2026-09-22追加、チャットテンプレート対応に伴う)。
    """
    text = tokenizer.decode(token_ids)
    if text == "" and len(token_ids) == 1:
        raw = tokenizer.id_to_token(token_ids[0])
        if raw is not None:
            return raw
    return text


def resolve_stop_token_ids(request: dict, model_dir: Path, tokenizer) -> list[int] | None:
    """各エントリポイントの生成呼び出し用: チャットテンプレート適用時のみ停止トークンを返す。

    use_chat_template=Falseの場合、またはtokenizerが無い(input_ids直接指定)場合はNoneを返し、
    従来どおりmax_new_tokens分を最後まで生成する(既存の挙動を変えない)。
    """
    if not request.get("use_chat_template", False) or tokenizer is None:
        return None
    return get_stop_token_ids(Path(model_dir), tokenizer)
