# Python開発用Dev Container環境

このリポジトリはVS Code Dev Containersを使用したPython開発環境を提供します。Docker上で一貫した開発環境を構築し、環境構築の手間を省きます。

## 環境の概要

- **Python 3.13**: 最新のPython環境を提供
- **開発ツール**: Git, Zsh, Oh My Zshなどの便利なツールが事前インストール済み
- **VS Code拡張機能**: Python開発に必要な拡張機能が自動的に設定

## フォルダ構成

```
.
├── .devcontainer/         # Dev Container設定
│   └── devcontainer.json  # VS Code Dev Container設定ファイル
├── Dockerfile             # 開発環境のDockerfile
├── module/                # 共有モジュール
│   └── requirements.txt   # Pythonパッケージの依存関係
└── workspace/             # 作業ディレクトリ
    └── code.py            # Pythonコード
```

## 主な特徴

### 1. コンテナ化された開発環境

- Python 3.13がプリインストールされたDebian Bookworm環境
- 専用のvscodeユーザーで実行（権限問題を回避）
- 必要なパッケージは自動的にインストール

### 2. ボリュームマウント

- `module`: プロジェクトのモジュールがコンテナ内の`/module`にマウント
- `workspace`: 開発用コードをコンテナ内の`/workspace`にマウント

### 3. VS Code統合

以下の拡張機能が自動的にインストールされます：
- `ms-python.python`: Python言語サポート
- `ms-python.vscode-pylance`: 高度なPython言語機能
- `charliermarsh.ruff`: Pythonリンターとフォーマッター
- `ms-toolsai.jupyter`: Jupyter Notebook統合

### 4. 開発ツール

- Zsh & Oh My Zsh: 強力なシェル環境
- Git: バージョン管理
- その他基本的なユーティリティ

## 使用方法

1. VS Codeを開き、「Dev Containersで開く」を選択
2. コンテナのビルドと起動を待つ
3. 自動的に必要なパッケージがインストールされる
4. `/workspace`ディレクトリでPythonコードを編集・実行

## 依存パッケージ

現在、以下のPythonパッケージが自動的にインストールされます：
- Flask 3.0.0: Webアプリケーションフレームワーク
- Requests 2.31.0: HTTPリクエスト用ライブラリ

新しいパッケージを追加する場合は、`module/requirements.txt`を更新し、コンテナを再ビルドしてください。

## 注意事項

- 初回起動時には、Dockerイメージのビルドに時間がかかる場合があります
- コンテナ内での変更は`module`と`workspace`ディレクトリのみホストと同期されます