"""Phase 1: 自前PyTorch実装によるLlama系モデルの推論エンジン。

依存はtorchのみ(transformersライブラリは使わない)。
トークナイザーだけは例外的にHugging Face製 `tokenizers` パッケージを使用する。
"""
