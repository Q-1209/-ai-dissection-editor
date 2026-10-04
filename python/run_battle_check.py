#!/usr/bin/env python3
"""
run_battle_check.py

Multi-Model Battle Viewの動作確認を簡単に行うためのCLIラッパー。

run_battle_view.py本体はElectron経由でstdin JSONを受け取る設計(GUIからの利用が
前提)だが、手動での動作確認や、結果をClaudeにそのまま読ませて検証させたい場合には、
モデルディレクトリを列挙するだけで実行できるようにする(run_token_journey_check.pyと
同じ狙い)。

各モデルディレクトリのconfig.jsonを読み、フォルダ名をラベルとして使う。
比較対象は生成結果・内部状態の両方をデフォルトで実行する。

使い方:
    python run_battle_check.py --model_dirs "C:\\path\\to\\model_a" "C:\\path\\to\\model_b" --input_text "こんにちは"
    (省略時、結果はこのファイルと同じディレクトリの battle_result.json に保存される)

出力したJSONファイルを Claude に渡せば(パスを伝える、またはlogs/配下に置いて
次のチャットで参照させる)、Claudeはスクリーンショットを目視するよりも確実に
結果を読み取って検証できる。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_battle_view import run_battle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dirs", required=True, nargs="+",
        help="比較するHuggingFace形式モデルディレクトリを2個以上スペース区切りで指定(1個でも可)",
    )
    parser.add_argument("--input_text", default="こんにちは", help="全モデル共通の入力文")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument(
        "--skip_generation", action="store_true", help="生成結果の比較をスキップする(内部状態のみ)"
    )
    parser.add_argument(
        "--skip_internals", action="store_true", help="内部状態の比較をスキップする(生成結果のみ)"
    )
    parser.add_argument(
        "--output", default=None,
        help="結果JSONの保存先パス。省略時はこのスクリプトと同じディレクトリのbattle_result.json",
    )
    args = parser.parse_args()

    models = []
    for model_dir_str in args.model_dirs:
        model_dir = Path(model_dir_str)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        models.append({"model_dir": str(model_dir), "config": config, "label": model_dir.name})

    request = {
        "models": models,
        "input_text": args.input_text,
        "compare_generation": not args.skip_generation,
        "compare_internals": not args.skip_internals,
        "max_new_tokens": args.max_new_tokens,
    }

    result = run_battle(request)

    output_path = (
        Path(args.output) if args.output else Path(__file__).resolve().parent / "battle_result.json"
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"結果を書き出しました: {output_path}")
    for r in result["results"]:
        if r["success"]:
            gen = f" 生成結果: {r['generated_text']!r}" if r.get("generated_text") is not None else ""
            print(f"  [{r['label']}] OK{gen}")
        else:
            print(f"  [{r['label']}] 失敗: {r['error_type']}: {r['message']}")


if __name__ == "__main__":
    main()
