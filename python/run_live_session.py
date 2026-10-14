#!/usr/bin/env python3
"""
run_live_session.py

Phase3: Live Slider(スライダーによるリアルタイム出力操作、2026-09-24着手)のエントリポイント。

他のエントリポイント(1回のリクエスト→1回の応答)と違い、常駐するプロセス。Electronが起動したまま
保持し、1行1JSONのコマンドをstdinで受け取り、1行1JSONのイベントをstdoutに流す。モデルの重みは
最初に1回だけ読み込むので、2回目以降の生成・スイープはすぐに始まる。

生成中もstdinを別スレッドで読み続け、set_controls(スライダーの値)と stop は生成ループが
トークンごとに取り込む。それ以外のコマンドは生成が終わってから順に処理する。

コマンド(stdin、1行1JSON):
  {"cmd": "init", "model_dir", "config", "nodes", "edges", "use_chat_template": false}   ← 最初に必ず1回
  {"cmd": "set_steering", "positive_prompts": [...], "negative_prompts": [...], "target_node_ids": [...]}
  {"cmd": "set_controls", "controls": {...}}           (形式は inference/live_controls.py 参照)
  {"cmd": "generate", "input_text", "max_new_tokens": 40, "compare_baseline": true,
     "temperature": 0.0, "top_k", "top_p", "repetition_penalty", "no_repeat_ngram_size"}
  {"cmd": "stop"}                                      生成中なら打ち切る
  {"cmd": "sweep", "input_text", "target": {...}, "start": 0, "stop": 2, "steps": 9}
  {"cmd": "shutdown"}

イベント(stdout、1行1JSON):
  {"event": "ready", "limits": {...}}  /  {"event": "steering_ready", "vector_norms": {...}}
  {"event": "controls_applied", "version", "controls"}
  {"event": "started"} / {"event": "baseline", "text"} / {"event": "token", "text", "kl", "controls_version", ...}
  {"event": "done", "text", "reason", "mean_kl", "max_kl"}
  {"event": "sweep_result", "target", "points": [{"scale", "kl", "top_token_text", "top_prob"}, ...]}
  {"event": "error", "error_type", "message", "cmd"}

使い方: Electron(electron/main.ts の live:start)から起動される。手で試す場合は
  python python/run_live_session.py  を起動し、上のJSONを1行ずつ貼り付ける。
"""

from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.graph_validation import (  # noqa: E402
    GraphNode,
    GraphValidationError,
    build_input_map_multi,
    topological_sort,
)
from inference.live_controls import (  # noqa: E402
    ControlError,
    Controls,
    make_sweep_scales,
    normalize_controls,
    validate_command,
)
from inference.tokenizer_utils import TokenizerLoadError  # noqa: E402
from inference.i18n import tr  # noqa: E402

_out_lock = threading.Lock()


def emit(event: dict) -> None:
    with _out_lock:
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def emit_error(exc: Exception, cmd: str | None) -> None:
    if isinstance(exc, (GraphValidationError, ControlError)):
        etype = "GraphValidationError" if isinstance(exc, GraphValidationError) else "ControlError"
    elif isinstance(exc, TokenizerLoadError):
        etype = "TokenizerLoadError"
    else:
        etype = "UnexpectedError"
        traceback.print_exc(file=sys.stderr)
    emit({"event": "error", "error_type": etype, "message": str(exc), "cmd": cmd})


def build_graph(request: dict):
    nodes = [
        GraphNode(id=n["id"], role=n["role"], layer_index=n.get("layer_index"), merge_mode=n.get("merge_mode", "add"))
        for n in request["nodes"]
    ]
    edges = [(e["source"], e["target"]) for e in request["edges"]]
    entry = [n.id for n in nodes if n.role == "embedding"]
    exit_ = [n.id for n in nodes if n.role == "lm_head"]
    if len(entry) != 1 or len(exit_) != 1:
        raise GraphValidationError(tr("embeddingノードとlm_headノードがそれぞれ1つずつ必要です。", "Exactly one embedding node and one lm_head node are required."))
    topological_sort(nodes, edges)
    build_input_map_multi(nodes, edges, entry_node_id=entry[0])
    return nodes, edges, entry[0], exit_[0]


class Server:
    def __init__(self, inbox: "queue.Queue[dict | None]"):
        self.inbox = inbox
        self.session = None
        self.controls = Controls()
        self.version = 0
        self.stop_requested = False
        self.pending: list[dict] = []
        self.running = True

    # 生成ループから毎トークン呼ばれる: 届いているset_controls/stopを取り込む
    def _drain(self) -> None:
        # 待ち行列の先頭に、今の生成に向けた set_controls / stop / shutdown が並んでいれば先に取り込む
        while self.pending and self.pending[0].get("cmd") in ("set_controls", "stop", "shutdown"):
            msg = self.pending.pop(0)
            cmd = msg.get("cmd")
            if cmd == "set_controls":
                self._set_controls(msg)
            else:
                self.stop_requested = True
                if cmd == "shutdown":
                    self.running = False
        while True:
            try:
                msg = self.inbox.get_nowait()
            except queue.Empty:
                return
            if msg is None:
                self.stop_requested = True
                self.running = False
                return
            cmd = msg.get("cmd")
            if self.pending:
                # 先に届いた別のコマンドが待っている場合は、順番を守るため後ろに並べる
                # (例: generate→stop の stop は、2つ目のgenerateを止めるためのもの)
                self.pending.append(msg)
            elif cmd == "set_controls":
                self._set_controls(msg)
            elif cmd == "stop":
                self.stop_requested = True
            elif cmd == "shutdown":
                self.stop_requested = True
                self.running = False
            else:
                self.pending.append(msg)

    def _set_controls(self, msg: dict) -> None:
        try:
            self.controls = normalize_controls(msg.get("controls"), self.session.limits)
            self.version += 1
            emit({"event": "controls_applied", "version": self.version, "controls": self.controls.to_dict()})
        except Exception as e:  # noqa: BLE001
            emit_error(e, "set_controls")

    def handle(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        try:
            validate_command(msg)
            if cmd == "shutdown":
                self.running = False
                return
            if cmd == "init":
                self._init(msg)
                return
            if self.session is None:
                raise ControlError(tr("先に init コマンドでモデルを読み込んでください。", "Load the model with the init command first."))
            if cmd == "set_controls":
                self._set_controls(msg)
            elif cmd == "set_steering":
                norms = self.session.set_steering(
                    msg.get("positive_prompts") or [], msg.get("negative_prompts") or [], msg.get("target_node_ids") or []
                )
                emit({"event": "steering_ready", "vector_norms": norms, "warnings": getattr(self.session, "steering_warnings", [])})
            elif cmd == "generate":
                self._generate(msg)
            elif cmd == "sweep":
                scales = make_sweep_scales(msg.get("start", 0.0), msg.get("stop", 2.0), msg.get("steps", 9))
                target = msg.get("target") or {}
                # 対象の妥当性はnormalize_controlsで確認する(未知のノード・範囲外の番号はエラー)
                from inference.live_controls import override_target  # noqa: PLC0415

                normalize_controls(override_target(self.controls, target, 1.0).to_dict(), self.session.limits)
                points, base = self.session.sweep(msg.get("input_text") or "", target, scales, self.controls)
                emit({"event": "sweep_result", "target": target, "points": points, **base})
            elif cmd == "stop":
                pass  # 生成していないときの stop は何もしない
        except Exception as e:  # noqa: BLE001
            emit_error(e, cmd)

    def _init(self, msg: dict) -> None:
        from inference.live_session import LiveSession  # noqa: PLC0415
        from inference.tokenizer_utils import load_tokenizer  # noqa: PLC0415

        model_dir = Path(msg["model_dir"])
        nodes, edges, entry, exit_ = build_graph(msg)
        tokenizer = load_tokenizer(model_dir)
        self.session = LiveSession(
            model_dir, msg["config"], nodes, edges, entry, exit_, tokenizer,
            use_chat_template=bool(msg.get("use_chat_template", False)),
        )
        self.controls = Controls()
        self.version = 0
        lim = self.session.limits
        emit({"event": "ready", "limits": {
            "node_ids": sorted(lim.node_ids), "num_layers": lim.num_layers,
            "intermediate_size": lim.intermediate_size, "num_heads": lim.num_heads,
        }, "use_chat_template": self.session.use_chat_template})

    def _generate(self, msg: dict) -> None:
        from inference.sampling import SamplingConfig  # noqa: PLC0415

        sampling = SamplingConfig(
            temperature=float(msg.get("temperature", 0.0)),
            top_k=msg.get("top_k"),
            top_p=msg.get("top_p"),
            repetition_penalty=float(msg.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=msg.get("no_repeat_ngram_size"),
        )
        self.stop_requested = False

        def get_controls():
            self._drain()
            return self.version, self.controls

        def should_stop():
            self._drain()
            return self.stop_requested

        self.session.generate(
            msg.get("input_text") or "", int(msg.get("max_new_tokens", 40)), sampling,
            get_controls, should_stop, emit, compare_baseline=bool(msg.get("compare_baseline", True)),
        )

    def serve(self) -> None:
        while self.running:
            if self.pending:
                msg = self.pending.pop(0)
            else:
                msg = self.inbox.get()
                if msg is None:
                    break
            self.handle(msg)


def reader(inbox: "queue.Queue[dict | None]", stream) -> None:
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            inbox.put(json.loads(line))
        except json.JSONDecodeError as e:
            emit({"event": "error", "error_type": "InvalidInput", "message": str(e), "cmd": None})
    inbox.put(None)  # stdinが閉じられた=親プロセス終了


def main() -> None:
    # 標準入出力をUTF-8に固定(Windowsの既定はcp932。electron/main.tsも同じ指定で起動する)
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # 2026-10-13修正: DLLを含むモジュール(torch・numpy・tokenizers)は、stdinの読み取りスレッドを始める前に読み込む。
    # Windowsでは、別スレッドがstdin(パイプ)の読み取りで待っている間にDLLを読み込むと、stdinが閉じられるまで
    # 読み込みが止まる(実機で init がモデル読み込み前のnumpyで停止した。stdinを閉じる試験では再現しない)
    import torch  # noqa: F401, PLC0415
    from inference import live_session, sampling  # noqa: F401, PLC0415

    try:
        import tokenizers  # noqa: F401, PLC0415
    except ImportError:
        pass  # 未インストールの場合は init で TokenizerLoadError として報告される

    inbox: "queue.Queue[dict | None]" = queue.Queue()
    threading.Thread(target=reader, args=(inbox, sys.stdin), daemon=True).start()
    Server(inbox).serve()


if __name__ == "__main__":
    main()
    # 2026-10-13修正: 通常の終了処理に任せると、stdin待ちのまま残った読み取りスレッドとtorchの後片付けが
    # 重なり、Windowsで終了コードが 0xC0000005(アクセス違反)になっていた。出力を流しきってから即座に終了する
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
