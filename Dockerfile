# .devcontainer/Dockerfile
FROM python:3.13-slim-bookworm

# ユーザーを作成し、コンテナ内で実行されるユーザーとして設定 (推奨)
# devcontainer.jsonのremoteUserと合わせる
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get -y install --no-install-recommends git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ワークスペースディレクトリを作成し、ユーザーに所有権を付与
RUN mkdir -p /workspaces && chown $USERNAME:$USERNAME /workspaces

# ユーザーをデフォルトに設定 (Dockerfileの後半で他のユーザーに戻る必要がない場合)
USER $USERNAME

# ワークスペースディレクトリを設定
WORKDIR /workspaces/${CONTAINER_NAME}

# 必要に応じて、追加のパッケージや設定をここに追加
# 例: Node.js, Go, その他のツール

# 必要に応じて、pipのキャッシュディレクトリを設定 (高速化のため)
ENV PIP_CACHE_DIR=/tmp/pip_cache

# .bashrc などの設定をコピーする場合はここに記述
# COPY .bashrc /home/$USERNAME/.bashrc

# コンテナが開始されたときに実行されるデフォルトコマンド
# Dev Containerでは通常、VS Codeが適切なシェルを起動するため不要ですが、
# コンテナ単体で実行する場合に役立つ
CMD ["bash"]