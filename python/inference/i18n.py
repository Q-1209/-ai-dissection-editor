"""
i18n.py — エラー文などの表示言語(2026-10-13追加、英語UI)。

画面の言語はElectronが環境変数 AIDE_LANG ("ja" / "en") で子プロセスに渡す(electron/main.ts)。
未設定のとき(テスト・コマンドラインから直接実行したとき)は従来どおり日本語。

使い方: raise GraphValidationError(tr("日本語の文", "English message"))
"""

from __future__ import annotations

import os


def current_lang() -> str:
    return "en" if os.environ.get("AIDE_LANG", "").lower() == "en" else "ja"


def tr(ja: str, en: str) -> str:
    """表示言語に合わせて ja / en のどちらかを返す。"""
    return en if current_lang() == "en" else ja
