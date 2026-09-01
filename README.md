# AI Dissection Editor（解剖ビューア MVP）

ローカルで動くTransformer系AIモデル(Llama/Mistral/Qwen/Gemma等、HuggingFace形式)を
読み込み、その内部構造(Attention・MLP・正規化層など)を役割ラベル付きのノードグラフとして
可視化するデスクトップアプリです。

既存のノードエディタ系ツール(ComfyUI等)の多くは「モデルを外部から呼び出して繋ぐ」ことに
留まりますが、本ツールは「モデルを開いて中身を部品として扱う」ことを目指しています。
将来的には複数モデルの連結・融合や、異なるアーキテクチャ間の部品差し替えまで見据えた、
個人開発・自己資金0円からのOSSプロジェクトです。

**現状のスコープは Phase 0（解剖ビューア = 構造の可視化のみ）です。** 推論の実行や
モデル間の連結はまだ実装していません。詳しくは [未実装・今後の課題](#未実装今後の課題phase-1以降)
を参照してください。

**一緒に開発してくれる人を探しています。** Issue・PR・フィードバックいずれも歓迎です。
興味があれば気軽にDiscussionsやIssueで声をかけてください。

## 現在の対応状況

| アーキテクチャ | 状態 |
| --- | --- |
| Llama / Qwen | 実機動作確認済み(Qwen2.5-0.5B-Instruct、24層) |
| Mistral(GQA含む) | ダミーデータで構造検証済み |
| Gemma2 / Gemma3 | ダミーデータで構造検証済み |
| Mamba / SSM系 | 未対応(キー命名規則が根本的に異なるため個別対応が必要) |

## 構成

```
electron/
  main.ts       Electronメインプロセス。ファイル選択ダイアログとPythonサブプロセス起動を担当
  preload.ts    レンダラーに安全なAPIを公開するpreloadスクリプト
python/
  inspect_model.py   config.json + safetensorsヘッダーを読み、ノードグラフJSONを出力
src/
  App.tsx                       メイン画面
  index.css                     デザイントークン(配色・タイポグラフィ)定義
  lib/toFlowGraph.ts            解析結果 -> React Flow用ノード/エッジへの変換(折りたたみ対応)
  components/NormalNode.tsx     通常ノードのカスタムReact Flowコンポーネント
  components/LayerSummaryNode.tsx  レイヤー折りたたみ/展開ヘッダーのカスタムコンポーネント
  components/Legend.tsx         役割ごとの色分け凡例
  electron.d.ts                 window.electronAPIの型定義
```

## 仕組み

1. 「モデルフォルダを開く」でHuggingFace形式のモデルディレクトリ（`config.json` +
   `*.safetensors`）を選択
2. Electronが `python/inspect_model.py` をサブプロセスとして起動
3. Pythonスクリプトは safetensors の**ヘッダーのみ**を読む（重み本体はロードしない）ため、
   数GBのモデルでも一瞬で構造を取得できる
4. state_dictのキー名（例: `model.layers.0.self_attn.q_proj.weight`）を正規表現で
   パースし、役割（Attention/MLP/正規化層など）とレイヤー番号を推定
5. 結果をJSONでElectron経由React側に渡し、React Flowでグラフ描画。デフォルトは全層折りたたみ
   表示で、レイヤーごとにクリックして展開/折りたたみができる

## 対応範囲（現時点）

- モデル形式: safetensors のみ（`.bin` のpickle形式は非対応）
- アーキテクチャ: 上記の対応状況表を参照。判定ロジックはHuggingFace `transformers`の
  Llama系実装をベースにした命名規則（`self_attn.q/k/v/o_proj`、`mlp.gate/up/down_proj`、
  `input_layernorm`、`post_attention_layernorm`、Gemma2/3では追加で
  `pre_feedforward_layernorm`/`post_feedforward_layernorm`）に依存している
- 依存: Python標準ライブラリのみで解析可能（`transformers` のインストール不要）。
  Electron側からPythonを呼ぶため、実行環境に `python3`（Windowsでは`python`でも可、
  `PYTHON_BIN`環境変数で指定）が必要

## セットアップ

```bash
npm install
```

Python側は標準ライブラリのみで動作するため追加インストール不要です。
`inspect_model.py` を直接動かして確認する場合:

```bash
python3 python/inspect_model.py /path/to/huggingface_model_dir
```

## 開発時の起動

```bash
npm run electron:dev
```

Vite dev serverとElectronを同時に起動します（GUI環境が必要）。

## 配布ビルド

```bash
npm run electron:build
```

`electron-builder` で各OS向けパッケージを `release/` に出力します。

## 未実装・今後の課題（Phase 1以降）

- 複数モデルの同時ロードとノード間の実接続（現状は同一モデル内の可視化のみ）
- Mamba/SSM系アーキテクチャへの対応
- レイアウトの改善（現状はレイヤー番号×役割の単純グリッド配置）
- 大規模モデル（分割safetensors・量子化モデル等）でのエッジケース検証

## 貢献方法

Issue・Pull Request・Discussionでのフィードバック、どれも歓迎です。特に以下のような
貢献を歓迎しています。

- 対応アーキテクチャの拡張（Mamba/SSM系、MoE系モデルなど）
- 実モデルでの動作報告・不具合報告
- UI/UXの改善提案
- Phase 1（ノード連結機能）の設計・実装への参加

大きな変更を加える前には、Issueで方向性を相談してもらえると助かります。

## ライセンス

[MIT License](./LICENSE)

