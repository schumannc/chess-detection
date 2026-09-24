# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    bash \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Non-root user so files written into the bind-mounted /app stay owned by the host user.
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# Install micromamba (for Python 3.11 on Ubuntu 22.04)
# This avoids Ubuntu's default Python 3.10 and keeps things reproducible.
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/opt/micromamba/bin:$PATH

RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Own the env prefix up front and build everything below as the non-root user.
# A chown -R after the fact rewrites every file into a new overlay layer,
# which duplicated the whole ~7.4GB Python env in the image.
RUN mkdir -p ${MAMBA_ROOT_PREFIX} /opt/sam3 \
 && chown ${USER_UID}:${USER_GID} ${MAMBA_ROOT_PREFIX} /opt/sam3

# Model weights land here; docker-compose mounts a named volume on it so a
# rebuild does not re-download SAM3.
RUN mkdir -p /cache/huggingface && chown -R ${USER_UID}:${USER_GID} /cache
ENV HF_HOME=/cache/huggingface

USER ${USERNAME}

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

# Install SAM3. The clone lives in /opt, not /tmp: the directory is named `sam3`,
# so with it under /tmp any script run from /tmp imports the clone root as a
# namespace package and shadows the real one (sam3.__file__ becomes None).
ARG SAM3_GIT_URL=https://github.com/facebookresearch/sam3.git
RUN git clone --depth 1 ${SAM3_GIT_URL} /opt/sam3 \
 && pip install --no-cache-dir -e "/opt/sam3[notebooks]" \
 && rm -rf /opt/sam3/.git

# App deps baked in, so the bind-mounted source needs no install step at runtime.
COPY --chown=${USER_UID}:${USER_GID} requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# sam3/model_builder.py importa pkg_resources, removido no setuptools 81.
# Precisa vir depois de tudo que possa puxar um setuptools mais novo.
RUN pip install --no-cache-dir "setuptools<81"

RUN python -m ipykernel install --sys-prefix --name py311 --display-name "Python 3.11 (CUDA)"

# src layout: PYTHONPATH replaces the editable install, so nothing writes
# *.egg-info back into the mounted working tree.
ENV PYTHONPATH=/app/src

CMD ["python", "run.py"]
