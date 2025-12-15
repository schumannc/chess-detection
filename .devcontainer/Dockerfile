# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    openssh-client \
    build-essential \
    sudo \
    passwd \
    bash \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

 # Create a non-root user that VS Code devcontainers expect
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}

# Install micromamba (for Python 3.11 on Ubuntu 22.04)
# This avoids Ubuntu's default Python 3.10 and keeps things reproducible.
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/opt/micromamba/bin:$PATH

RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Create a Python 3.11 environment + Jupyter tooling
# Install pip + ipykernel + jupyterlab inside the environment.
RUN micromamba create -y -n py311 -c conda-forge \
      python=3.11 \
      pip \
      ipython \
      ipykernel \
      jupyterlab \
 && micromamba clean -a -y

# Activate env by default for all shells/commands
ENV CONDA_DEFAULT_ENV=py311
ENV PATH=/opt/micromamba/envs/py311/bin:$PATH

ENV PYTHONNOUSERSITE=1 \
    PIP_USER=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install SAM3
ARG SAM3_GIT_URL=https://github.com/facebookresearch/sam3.git
RUN git clone ${SAM3_GIT_URL} /tmp/sam3 \
 && pip install --no-cache-dir -e "/tmp/sam3[notebooks]"

RUN python -m ipykernel install --sys-prefix --name py311 --display-name "Python 3.11 (CUDA Devcontainer)"

RUN which python && python -V && which pip && pip -V

RUN chown -R ${USERNAME}:${USERNAME} /opt/micromamba

USER ${USERNAME}
