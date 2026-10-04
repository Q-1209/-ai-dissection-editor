#!/usr/bin/env python3
"""
run_token_journey_check.py

Token Journeyの動作確認を簡単に行うためのCLIラッパー。

compute_token_journey.py本体はElectron経由でstdin JSONを受け取る設計(GUIからの
利用が前提)だが、手動での動作確認や、結果をClaudeにそのまま読ませて検証させたい
場合には、いちいちノード+エッジのJSONを手で組み立てるのは面倒。

このスクリプトは、モデルディレクトリと入力テキストだけを指定すれば、
デフォルトの全層グラフ(embedding -> layer_block(全層、折りたたみ状態) -> lm_head、
toExecutionGraph.tsのデフォルト構成と同じ)を自動生成してcompute_token_journey.pyの
run_token_journey()を呼び出し、結果をJSONファイルに保存する。

さらに、GUIのテーブルを目視で確認する代わりに機械的に検証できるよう、
「同じトークンidが一定ブロック数以上連続して選ばれ続けている(=層を追っても
意味の変遷が全く起きていない=収束の疑いがある)」区間を自動検出して
`collapse_warnings`として結果JSONに含める(そのブロック区間の平均top-1類似度
`mean_similarity`も含めるので、「高い類似度で自信を持って同じ単語を選び続けている」
のか「どの候補も低い類似度=実質ノイズの中でたまたま同じ単語が勝ち続けている」
のかを区別できる。後者は残差接続によりレイヤー間の隠れ状態がほとんど回転しない
という、この手法(logit lens)自体の既知の限界であり、必ずしもバグではない
可能性がある)。

--compare を付けると、logit lens補正(最終RMSNormの適用)ありと補正なしの両方を
1回で実行し、`comparison`として差分サマリ(補正の有無で最大の収束連続長・平均
類似度がどう変わったか)を結果JSONに追加する。補正が実際に効いているかを
Claudeが目視なしで確認できる。

使い方:
    python run_token_journey_check.py --model_dir "C:\\path\\to\\model" --input_text "こんにちは"
    python run_token_journey_check.py --model_dir "C:\\path\\to\\model" --input_text "こんにちは" --compare
    (省略時、結果はこのファイルと同じディレクトリの token_journey_result.json に保存される)

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

from compute_token_journey import run_token_journey  # noqa: E402


def build_default_graph(config: dict) -> tuple[list[dict], list[dict]]:
    """embedding -> layer_block(全層、折りたたみ状態) -> lm_head という、
    src/lib/toExecutionGraph.tsのデフォルト(expandedLayers空集合)と同じ構成のグラフを作る。
    """
    num_layers = config["num_hidden_layers"]
    nodes = [{"id": "model.embed_tokens", "role": "embedding", "layer_index": None}]
    for i in range(num_layers):
        nodes.append({"id": f"model.layers.{i}", "role": "layer_block", "layer_index": i})
    nodes.append({"id": "lm_head", "role": "lm_head", "layer_index": None})
    edges = [{"source": nodes[i]["id"], "target": nodes[i + 1]["id"]} for i in range(len(nodes) - 1)]
    return nodes, edges


def detect_collapses(
    journey: dict[str, list[list[dict]]],
    block_order: list[str],
    seq_len: int,
    min_run: int = 4,
) -> list[dict]:
    """トークン位置ごとに、top-1のtoken_idが同じ値のままmin_runブロック以上
    連続している区間を検出する(層を追っても最近傍単語が全く変化しない=
    収束の疑いがあるサイン)。区間内の平均top-1類似度(mean_similarity)も
    合わせて記録する: 値が高ければ「自信を持って同じ単語に居座っている」、
    低ければ「どの候補も微妙な差でたまたま同じ単語が勝ち続けている(=実質
    ノイズ、logit lens手法自体の既知の限界の可能性)」と判断する材料になる。

    embedding直後の1〜2ブロックだけ元の入力トークンが連続するのは正常な
    挙動なので誤検知しうるが、min_run(デフォルト4)以上の長い連続だけを
    警告対象にすることで、実運用上問題になる「延々と同じ単語」パターンだけを拾う。
    """
    warnings: list[dict] = []

    for pos in range(seq_len):
        prev_token: int | None = None
        run_len = 0
        run_start_idx = 0
        run_similarities: list[float] = []

        def flush(end_idx: int) -> None:
            if run_len >= min_run:
                warnings.append(
                    {
                        "position": pos,
                        "token_id": prev_token,
                        "block_range": [block_order[run_start_idx], block_order[end_idx]],
                        "run_length": run_len,
                        "mean_similarity": sum(run_similarities) / len(run_similarities)
                        if run_similarities
                        else None,
                    }
                )

        for i, block_id in enumerate(block_order):
            entries = journey.get(block_id, [])
            top = entries[pos][0] if pos < len(entries) and entries[pos] else None
            token_id = top["token_id"] if top else None
            similarity = top["similarity"] if top else None

            if token_id is not None and token_id == prev_token:
                run_len += 1
                run_similarities.append(similarity)
            else:
                flush(i - 1)
                run_start_idx = i
                run_len = 1 if token_id is not None else 0
                run_similarities = [similarity] if token_id is not None else []
            prev_token = token_id

        flush(len(block_order) - 1)

    return warnings


def summarize_collapses(warnings: list[dict]) -> dict:
    """collapse_warningsのリストから、比較用の要約統計を作る(--compare用)。"""
    if not warnings:
        return {"count": 0, "max_run_length": 0, "mean_similarity_of_longest_run": None}
    longest = max(warnings, key=lambda w: w["run_length"])
    return {
        "count": len(warnings),
        "max_run_length": longest["run_length"],
        "mean_similarity_of_longest_run": longest["mean_similarity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="HuggingFace形式モデルディレクトリ")
    parser.add_argument("--input_text", default="こんにちは", help="追跡する入力文")
    parser.add_argument("--top_k", type=int, default=3, help="各位置・各ブロックで保存する近傍単語の件数")
    parser.add_argument(
        "--min_run",
        type=int,
        default=4,
        help="この件数以上ブロックが連続して同じtoken_idになったら収束警告を出す",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="結果JSONの保存先パス。省略時はこのスクリプトと同じディレクトリのtoken_journey_result.json",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="logit lens補正(最終RMSNorm適用)ありと補正なしの両方を実行し、差分サマリを出力する",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    nodes, edges = build_default_graph(config)

    base_request = {
        "model_dir": str(model_dir),
        "config": config,
        "nodes": nodes,
        "edges": edges,
        "input_text": args.input_text,
        "top_k": args.top_k,
    }

    def run_and_annotate(apply_final_norm: bool) -> dict:
        r = run_token_journey(base_request, apply_final_norm=apply_final_norm)
        if r.get("success"):
            r["collapse_warnings"] = detect_collapses(
                r["journey"], r["block_order"], seq_len=len(r["input_token_ids"]), min_run=args.min_run,
            )
        return r

    if args.compare:
        corrected = run_and_annotate(apply_final_norm=True)
        raw = run_and_annotate(apply_final_norm=False)
        result = {
            "success": corrected.get("success") and raw.get("success"),
            "corrected": corrected,
            "raw": raw,
        }
        if corrected.get("success") and raw.get("success"):
            corrected_summary = summarize_collapses(corrected["collapse_warnings"])
            raw_summary = summarize_collapses(raw["collapse_warnings"])
            result["comparison"] = {
                "corrected": corrected_summary,
                "raw": raw_summary,
                # 補正ありの最大連続長が補正なしより短ければ、補正に効果があったと言える
                "final_norm_reduced_max_run_length": (
                    corrected_summary["max_run_length"] < raw_summary["max_run_length"]
                ),
            }
    else:
        result = run_and_annotate(apply_final_norm=True)

    output_path = (
        Path(args.output) if args.output else Path(__file__).resolve().parent / "token_journey_result.json"
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"結果を書き出しました: {output_path}")
    if args.compare:
        if result["success"]:
            print(f"comparison: {json.dumps(result['comparison'], ensure_ascii=False)}")
        else:
            print("compareのいずれかの実行が失敗しました。JSON内のerror_type/messageを確認してください。")
    elif result.get("success"):
        n_warn = len(result["collapse_warnings"])
        print(f"collapse_warnings: {n_warn}件" + ("（要確認）" if n_warn else "（問題なし）"))
    else:
        print(f"失敗: {result.get('error_type')}: {result.get('message')}")


if __name__ == "__main__":
    main()
