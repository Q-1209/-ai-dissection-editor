# 開発ログ（旧README、2026-10-04に移設）

> 2026-10-04のOSS公開にあたり、これまでのREADME（設計判断・実装詳細・修正したバグの記録）をここに移しました。
> 最新の概要と使い方は [README.ja.md](../README.ja.md) / [README.md](../README.md) を参照してください。
> 記述は書かれた時点のもので、一部は古くなっています（例: Phase2・3はその後完了しています）。

# AI Dissection Editor（解剖ビューア MVP）

ローカルで動くTransformer系AIモデル(Llama/Mistral/Qwen/Gemma等、HuggingFace形式)を
読み込み、その内部構造(Attention・MLP・正規化層など)を役割ラベル付きのノードグラフとして
可視化するデスクトップアプリです。

既存のノードエディタ系ツール(ComfyUI等)の多くは「モデルを外部から呼び出して繋ぐ」ことに
留まりますが、本ツールは「モデルを開いて中身を部品として扱う」ことを目指しています。
将来的には複数モデルの連結・融合や、異なるアーキテクチャ間の部品差し替えまで見据えた、
個人開発・自己資金0円からのOSSプロジェクトです。

**現状のスコープ:**
- **Phase 0（解剖ビューア = 構造の可視化）: 完了。** GUIで実際に動作します
- **Phase 1（ノード連結・実行）: GUI統合済み。** アプリ上部の「解剖」「実行」タブを
  切り替えることで、実行タブでは同一モデル内のAttention/MLPブロックをドラッグで
  自由に繋ぎ替え、「実行」ボタンで実際に推論を実行できます。エンジン自体
  （自前PyTorch実装）は実機（Windows、Qwen2.5-0.5B実重み）で動作確認済みです。
  詳しくは [Phase 1: ノード連結・実行エンジン](#phase-1-ノード連結実行エンジン)
  を参照してください

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
  inspect_model.py   config.json + safetensorsヘッダーを読み、ノードグラフJSONを出力(Phase0)
  execute_graph.py   Electronから呼ばれるPhase1実行のエントリポイント(標準入出力でJSON通信)
  inspect_internals.py  Phase2: 内部状態(Attention重み・活性化値)取得のエントリポイント
  analyze_sensitivity.py  Phase2: Prompt Sensitivity Map(トークン別影響度)のエントリポイント
  compute_token_journey.py  Phase2: Token Journey(トークンの意味変遷追跡)のエントリポイント
  run_phase1_smoke_test.py  Phase1推論エンジンの動作確認スクリプト(手元torch環境で実行)
  inference/          Phase1: 自前PyTorch実装の推論エンジン
    load_weights.py    safetensorsから実テンソル値を読み込む
    layers.py          RMSNorm・RoPE(KVキャッシュ対応)・GQA対応Attention・SwiGLU MLPの自前実装
    graph_validation.py  グラフのトポロジカルソート・検証、合流(複数入力)対応(torch非依存)
    graph_runner.py    ノード+エッジ定義に従って推論を実行する(execute: 1ステップ、
                        generate: KVキャッシュを使った複数トークン逐次生成、torch依存)
    tokenizer_utils.py  tokenizersライブラリを使ったテキスト<->トークンID変換
    sampling.py         temperature・top-k・top-p・repetition penaltyのサンプリング戦略
    sensitivity.py       Phase2: KLダイバージェンス計算・トークン置換ロジック
    token_journey.py     Phase2: コサイン類似度による最近傍単語探索ロジック
  tests/
    test_layers_numpy.py    layers.pyと同じ数式のnumpy版によるロジック検証
    test_graph_runner.py    graph_validation.pyの単体テスト
    test_sampling_numpy.py  sampling.pyと同じロジックのnumpy版による検証
    test_sensitivity_numpy.py  sensitivity.pyと同じロジックのnumpy版による検証
    test_token_journey_numpy.py  token_journey.pyと同じロジックのnumpy版による検証
    test_split_safetensors.py  分割safetensors(複数ファイル)の読み込みロジック検証
src/
  App.tsx                       メイン画面(解剖/実行タブ切り替え)
  index.css                     デザイントークン(配色・タイポグラフィ)定義
  lib/toFlowGraph.ts            Phase0: 解析結果 -> React Flow用ノード/エッジへの変換(折りたたみ対応)
  lib/toExecutionGraph.ts       Phase1: 解析結果 -> ブロック単位の実行用グラフへの変換
  components/NormalNode.tsx     通常ノードのカスタムReact Flowコンポーネント
  components/LayerSummaryNode.tsx  レイヤー折りたたみ/展開ヘッダーのカスタムコンポーネント
  components/Legend.tsx         役割ごとの色分け凡例
  components/AttentionHeatmap.tsx  Phase2: Attention重みのヒートマップ描画
  components/ActivationStatsList.tsx  Phase2: 層ごとの活性化値統計一覧
  components/SensitivityHeatmap.tsx  Phase2: Prompt Sensitivity Mapのトークン別影響度表示
  components/TokenJourneyView.tsx  Phase2: Token Journeyの意味変遷を表形式で表示
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

## Phase 1: ノード連結・実行エンジン

Phase 0の可視化グラフの上で、実際にレイヤーを自由に繋ぎ替えて推論を実行できる
エンジンを実装しています。

**スコープ（合意済みの設計方針）:**
- 対象は**単一モデル内のみ**。複数モデルをまたいだ接続はPhase 2以降に先送り
- レイヤー単位（Attentionブロック・MLPブロック）での順序入れ替え・スキップ接続を
  自由に組める
- ノード同士のshapeが合わない接続は**エラーで拒否**する。未学習のアダプター層を
  自動挿入して強制的につなぐことはしない（学習済み重み同士の組み合わせでない限り、
  出力の意味が保証できなくなるため）
- 実行の成功基準は「計算がエラーなく最後まで通ること」まで。**出力の意味的な妥当性は
  保証しない**（ユーザーが目視で確認する）。これは「解剖して繋いで確かめる」という
  実験ツールとしての性質上の割り切り

**GUI操作:** アプリ上部の「解剖」「実行」タブで切り替えられます。「実行」タブでは、
Attentionブロック・MLPブロック単位のノードがオリジナル構造通りの順序で並んでおり、
ドラッグでエッジを繋ぎ変える（新規接続・削除）ことで自由に順序を組み替えられます。
入力欄にテキストを入力し「実行」ボタンを押すと、Electron経由で`execute_graph.py`が
呼ばれ、実際にPyTorchで複数トークンの逐次生成（KVキャッシュ使用）が実行されます。
結果パネルには生成されたテキスト・入出力トークンID・実際に実行された順序が
表示されます。生成中はグラフ構造（繋ぎ方）を固定し、最初に確定したグラフのまま
最後まで生成し続けます（Phase1合意事項）。

**実装方式:** `transformers`ライブラリには依存せず、RMSNorm・RoPE（回転位置
エンコーディング）・GQA対応Attention・SwiGLU MLPをPyTorchで自前実装しています。
対象アーキテクチャはLlama系（Llama/Qwen/Mistral、RoPE必須）のみです。
トークナイザーだけは例外的にHugging Face製の軽量ライブラリ`tokenizers`を使用します
（BPEの完全自前実装はコストに見合わないと判断）。

**KVキャッシュ実装:** 各Attentionノードは、そのノードID固有のK/Vキャッシュを
保持します（`node_id -> (K cache, V cache)` の辞書）。1トークンずつ生成する際、
過去に計算したK/Vを使い回し、新しいトークン分だけを追加計算することで、生成の
たびに全トークン列を再計算する必要がありません。RoPEは各トークンの絶対位置に
依存するため、キャッシュ利用時も正しい位置のcos/sinをスライスして適用しています。
正しさは、「KVキャッシュを使った逐次計算」と「キャッシュなしの一括計算」が完全に
一致することを検証するテスト（`test_kv_cache_matches_full_recompute`、
`test_kv_cache_with_rope_matches_full_recompute`）で担保しています。

**実行確認について:** この開発環境ではGPU依存必須のPyTorchビルドしか入手できず、
ディスク容量の制約もあり実機での実行確認ができていません。ロジックの正しさは、
`layers.py`と全く同じ数式をnumpyで独立実装した`test_layers_numpy.py`で検証済み
です（RoPEの相対位置依存性、causal maskの遮断、softmaxの正規化、GQAの複製ロジック、
KVキャッシュの正しさなどを検証、計10テスト）。**Windows環境・Qwen2.5-0.5B実重みで、
単一forward pass・繋ぎ替え・KVキャッシュを使った複数トークン生成・実トークナイザー
でのテキスト入出力まで、全機能の実機検証が完了しています**。例えば「日本の首都は」
という入力に対し「- 世界の有名な都市 - 1」という文法的に自然な日本語が生成される
ことを確認済みです。単体でのスモークテストは以下のように実行できます。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers safetensors
python python/run_phase1_smoke_test.py /path/to/model_dir
```

**修正済みの重大バグ（開発中に発見）:**

1. **エッジ接続を無視する不具合**: 初期実装では、`execute()`がノードの
「実行順序リスト」だけを見て機械的に直前の出力を次に渡しており、**エッジの接続
関係を実際には見ていませんでした**。そのため、GUIでエッジを削除してグラフを
分断しても、それが無視されて「実行成功」してしまう不具合がありました（ユーザー
による実機テストで発覚）。`build_input_map()`を新設し、各ノードの入力元を
エッジから厳密に解決、孤立ノード（接続が欠落したノード）は実行前に必ずエラーで
検出するよう修正しています。この修正は回帰テスト（`test_graph_runner.py`の
`test_disconnected_edge_creates_unreachable_node`等）と、実機での再検証（意図した
繋ぎ替えが実行順序に正確に反映されることを確認）の両方で担保しています。

2. **Qwen2のQ/K/Vバイアス欠落**: 実装当初、Attentionの計算はLlama系の設計
（Q/K/V/O projectionすべてバイアスなし）を前提にしていましたが、**Qwen2は
Q/K/V projectionに学習済みのバイアス項を持つ**（`o_proj`のみバイアスなし）という
アーキテクチャ差異があり、これを完全に見落としていました。この結果、実行自体は
エラーなく完走するものの、生成されるテキストが常に意味不明な文字列の羅列に
なるという不具合がありました（数値としては健全に見えるため、「実行成功」の
判定だけでは検出できず、ユーザーによる実際のテキスト出力の確認で発覚）。
`list_available_keys()`を新設し、モデルファイルに実際に存在するバイアスキーを
事前スキャンして動的に判定・適用する仕組みに修正しています。修正後、「日本の
首都は」という入力への次トークン候補に「東」「日本」等の妥当な単語が上位に
出ること、複数トークン生成で文法的に自然な日本語が生成されることを実機で
確認済みです。

**サンプリング戦略:** 貪欲デコード（常に最大確率のトークンを選ぶ）は、同じ文脈に
戻ると同じ予測を繰り返してしまい、生成が短いループに陥りやすいという既知の性質が
あります。これを緩和するため、以下のサンプリング機能を実装しています（実行タブの
「生成設定」パネルから調整可能）。

- **Temperature**: logitsをtemperatureで割ってから確率分布を作る。`0`は貪欲デコード
  （デフォルト）、`1`は変化なし、大きいほどランダム性が増す
- **Top-k**: 確率上位k個のトークンだけに絞ってサンプリングする
- **Top-p (nucleus sampling)**: 累積確率がpを超えるまでの上位トークン集合から
  サンプリングする
- **Repetition penalty**: 既に生成済みのトークンのlogitsを下げ、繰り返しを抑制する
- **No-repeat n-gram**: 指定したnトークンの並び（n-gram）が2回出現するのを禁止する。
  repetition penaltyが既出トークン単体を弱めるだけなのに対し、こちらは既出の並びを
  完全に禁止するため、長い定型句の繰り返し（同じ文パターンの反復）により強く効く。
  貪欲デコード・サンプリングのどちらでも有効

これらは組み合わせて使え、適用順序は no-repeat n-gram → repetition penalty →
temperature → top-k → top-p → サンプリング（HuggingFace transformersの実装慣例に
合わせている）。実装は`inference/sampling.py`、正しさはnumpyでの独立検証
（`test_sampling_numpy.py`、repetition penaltyの効果・temperatureによる分布の
鋭さ変化・top-k/top-pの候補数・no-repeat n-gramが正しい並びだけを禁止すること・
組み合わせ時の確率分布の妥当性など計16テスト）で担保しています。

## Phase 2: 内部状態可視化（着手中）

Phase 1の実行エンジンを拡張し、推論中の内部状態を記録・可視化する機能を追加して
います。アプリの「観察」タブから使えます。「内部状態」「Sensitivity Map」の
2つのサブタブがあります。

### 内部状態（Attention重み・活性化値）

**できること:**
- テキストを入力して「観察」を押すと、1回のforward passを実行しながら各
  Attention/MLPブロックの内部状態を記録
- **Attentionヒートマップ**: どのトークンがどのトークンに注目しているかを、
  層ごとに選んで表示。ヘッド平均（トークン×トークン）と全ヘッド分の両方に対応
- **活性化値統計**: 各ブロックの出力（mean/std/min/max）を層の実行順に一覧表示。
  stdの大きさを棒グラフで視覚化し、層を追うごとに値の広がりがどう変化するかを
  把握できる。線形スケールと対数スケールを切り替えられ、層間でstdの桁が大きく
  異なる場合（下記のMassive Activations参照）でも、小さい値・大きい値の両方を
  同時に見比べられる
- 「実行」タブで組んだグラフ構造（繋ぎ替え後のもの含む）がそのまま使われるため、
  レイヤーを入れ替えた状態での内部状態も観察できる

**実装:** `layers.py`の`attention()`がAttention重みも返すよう拡張し、
`graph_runner.py`の各ブロック実行メソッドに`capture`引数を追加（`None`なら
一切記録せず、Phase 1までの実行と同じパフォーマンス特性を維持）。専用の
エントリポイント`inspect_internals.py`を新設。GUIで折りたたんだままの層
（`layer_block`ロール）は内部でAttention/MLPの2つのサブ状態に分かれて実行される
ため、`executed_order`のノードIDに対応づけて出力するリネーム処理も実装している。

**実機での発見（Windows・Qwen2.5-0.5Bで確認）:** ある日本語の質問文
を入力したところ、Attentionヒートマップは期待通りcausal maskに沿った下三角パターンを
示した。さらに興味深いことに、活性化値のstdが層2で急激に拡大（0.3→7.0）、
層3〜20で16〜17程度の高い値で持続し、層21〜23で急減する、という顕著なパターンが
観測された。これは["Massive Activations in Large Language Models"](https://arxiv.org/abs/2402.17762)
という論文で報告されている既知の現象と一致する（少数の固定次元・特定トークンで
桁違いに大きな活性化値が生じ、浅い層で突然出現し中間層を通じて持続、終盤の層で
減衰する）。実装のバグではなく、モデル自体が持つ性質をツールが正しく観測できて
いることの裏付けになった。

### Prompt Sensitivity Map

**できること:**
- 入力文の各トークンを1つずつ「削除」「パディングに置換」「ランダムトークンに
  置換」「指定トークンIDに置換」のいずれかで変更し、その都度forward passを
  実行して、出力（次トークンのlogits分布）がどれだけ変化したかを測定する
- 影響度は**KLダイバージェンス**（変更前後の分布のズレ）を主指標とし、次トークンの
  最上位予測が変わったかどうかも補足情報として表示する
- 結果はトークンごとの色分けタイル（影響が大きいほど濃い暖色）として表示。
  最上位予測が変わったトークンには枠で印が付く

**実装:** `sensitivity.py`を新設し、KLダイバージェンス計算とトークン置換
（4モード）のロジックを実装。専用エントリポイント`analyze_sensitivity.py`は
入力文の長さぶんの追加forward passを行うため、他のエントリポイントより実行に
時間がかかる。正しさはnumpyでの独立検証（`test_sensitivity_numpy.py`、KL
ダイバージェンスの非負性・単調性・各置換モードのロジックなど計9テスト）で
担保している。**Windows環境・Qwen2.5-0.5Bで実機確認済み。**「こんにちは、
元気ですか？」という入力（delete mode）に対し、文末の「？」のKLダイバージェンスが
他のトークン（0.04〜0.44）と比べて突出して大きい値（5.73）を示し、かつ最上位
予測トークンの変化も検出された。疑問符という文全体の性質（質問文であること）を
決定づけるトークンが最も重要だと判定される、直感的に妥当な結果が得られている。

### Token Journey

**できること:**
- 入力文の全トークンについて、各ブロック（Attention直後・MLP直後）を通過した
  時点での隠れ状態ベクトルを、埋め込み行列（`model.embed_tokens.weight`）の
  各行とのコサイン類似度で比較し、その時点で最も意味が近い単語（nearest
  neighbor、いわゆるlogit lensに近い考え方）を求める
- 結果はトークン位置（行）×ブロック（列）の表として表示。層を追うごとに
  「最も近い単語」がどう変化していくかを見ることで、モデル内部でのトークンの
  意味の変遷を追跡できる。セルの色の濃さはコサイン類似度の高さを表す

**実装:** `token_journey.py`を新設し、コサイン類似度による最近傍探索ロジックを
実装。既存の内部状態可視化と同じ`capture_internals`機構を再利用しており、
1回のforward passで全トークン・全ブロックの情報が得られるため、Prompt
Sensitivity Mapのように入力長に応じて追加のforward passが必要になることはない。
専用エントリポイント`compute_token_journey.py`を新設。正しさはnumpyでの独立
検証（`test_token_journey_numpy.py`、完全一致で類似度1.0になること・コサイン
類似度のスケール不変性・Top-k順序・層を追うごとに最近傍が変化する性質など
計5テスト）で担保している。**実機でのエンドツーエンド確認はまだ行っていない。**

## 今後のロードマップ

Phase 0（解剖ビューア）・Phase 1（ノード連結・実行エンジン）は完了しています。
その先の構想を、開発規模の近いもの同士でフェーズ分けしています（着手時期は未定）。

| フェーズ | 内容 |
| --- | --- |
| Phase 0 | 解剖ビューア（構造の可視化）— **完了** |
| Phase 1 | ノード連結・実行エンジン（KVキャッシュ・サンプリング・分岐合流含む）— **完了** |
| Phase 2 | 内部状態の可視化（Attention・活性化のヒートマップ表示）を基盤に、Prompt Sensitivity Map・複数モデル比較・Token Journey・Neuron Search・Layer Ablation Wizardなどの観察系機能 — **一部着手**（内部状態可視化の基盤部分を実装済み） |
| Phase 3 | 推論時のリアルタイム介入（スライダーでの出力操作）・重みの直接編集・Activation Steering・Sycophancy/Refusal挙動の調整など、モデルを「改造」する機能群 |
| Phase 4 | LoRA/QLoRAファインチューニング、学習データの自動生成、モデルマージなど、モデルを「学習させる」機能群 |
| Phase 5 | 改造・学習前後の差分検証、ハルシネーション検出、バイアス監査、Function Callingのデバッグなど、分析・監査・実務運用を支える機能群 |
| Phase 6 | 公開・初期収益化（旧Phase 4から後ろにスライド） |

**将来的な拡張方向性（別トラック、時期未定）:** ローカルAIをClaude Codeのような
構造化ハーネスで自動最適化する機能、複数のローカルAIを組み合わせてMoE的な
アーキテクチャを視覚的に構築する機能、外部サービス（課題管理・ドキュメント・
PRレビュー等）との連携、拡張機能の追加、チャット形式での自由対話、ノーコードでの
コード/Web/ゲーム/画像生成。これらは解剖・改造・学習という現在のツールの中核とは
別の方向性の機能群であり、実現するかどうか・どのフェーズに組み込むかは未確定です。

## 未実装・今後の課題（Phase 1時点で判明している技術的な積み残し）

- 複数モデルの同時ロードとノード間の実接続（Phase 2、現状は単一モデル内のみ）
- Mamba/SSM系アーキテクチャへの対応
- **MoE（Mixture-of-Experts）アーキテクチャへの対応**: 2025〜2026年時点の
  Mistral最新ラインナップ（Large 3、Medium 3.5、Small 4）やGemma 4は、いずれも
  MoE構成（一部マルチモーダル前提）にシフトしており、現在のPhase 1実装
  （`LlamaGraphExecutor`、密結合Transformer専用）ではそのままでは動作しない。
  対応にはルーティング機構・エキスパート選択ロジックなど設計レベルの拡張が必要で、
  Phase 1のスコープを超える規模の作業になる見込み。密結合構成のMistral
  （`model_type: "mistral"`、`MistralForCausalLM`、例: Ministral-8B-Instruct-2410
  等の旧世代）はアーキテクチャ的にはPhase 1の対応範囲内のはずだが、実機での
  動作確認はまだ行っていない
- ~~レイアウトの改善~~: 解剖タブのレイヤー列幅を、展開状態に応じて動的に計算する
  よう改善済み（折りたたみ時は代表ノード1つぶんの幅、展開時はAttention/MLP
  ブロックの実際の占有幅に応じて次の層の位置を決める）。一部の層だけ展開しても
  他の層と重ならず、無駄な余白も出ない
- **大規模モデル（分割safetensors・量子化モデル等）でのエッジケース検証**:
  分割safetensors（`model-00001-of-00003.safetensors`等、複数ファイルに分割された
  大規模モデル）への対応は、`test_split_safetensors.py`で恒久的な回帰テストとして
  検証済み（3ファイル・10ファイル分割、複数ファイルにまたがるキーの部分読み込み、
  存在しないキーの検出など計5テスト）。`model.safetensors.index.json`（索引ファイル）
  には頼らず、全safetensorsファイルのヘッダーを直接スキャンする設計のため、索引
  ファイルが欠損していても動作する。ただし実際の大規模モデル（数十GB・10ファイル
  以上）での読み込み速度・メモリ使用量、および量子化モデル特有のdtype
  （`F8_E4M3`等）への対応は未検証
- ~~実行タブでの分岐・合流のあるグラフの完全なサポート~~: 複数ノードへの分岐
  （1つの出力を複数のノードへドラッグで接続するだけ、追加実装は不要）と、複数
  ノードの合流（1つのノードに複数のエッジを接続すると、指定した方法で自動的に
  1つのテンソルに合成される）の両方に対応済み。合流の合成方法は
  **add**（要素ごとの加算、残差接続と同じ発想）と**average**（要素ごとの平均）
  から選べ、合流ノードをクリックするとトグルできる（GUI上に`[合流:add/average]`
  というラベルと強調枠で表示される）。合流させる入力同士のshapeが一致しない
  場合はエラーで拒否する。正しさはtorch非依存部分（孤立ノード検出の継続動作、
  合流の入力解決）をこの環境で検証済み（`test_graph_runner.py`に4テスト追加）、
  合成ロジック（add/averageの計算、3入力以上での動作）もnumpyで検証済み
  （`test_layers_numpy.py`に6テスト追加）。**Windows環境・Qwen2.5-0.5B実重みで、
  merge_mode='add'/'average'両方の実機動作確認済み**（両者が異なる結果になる
  ことも確認し、合成ロジックが正しく効いていることを実証）
- **長文生成時の一貫性**: `no_repeat_ngram`は完全に同一のn-gramの再出現は防げるが、
  「意味的に似た言い回しのバリエーション」を使った反復（例:「東京の政治の中心地」→
  「東京の経済の中心地」のように、構文パターンは同じで単語だけ変わる反復）までは
  防げない。これは小型モデル（Qwen2.5-0.5B等）でよく見られる既知の限界で、実機検証
  でも長い生成（100トークン超）では構文テンプレートへの回帰が観測されている。より
  強力な反復抑制（複数n-gramサイズの同時適用、文単位の類似度判定等）や、より大きな
  モデルでの検証が今後の課題

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

