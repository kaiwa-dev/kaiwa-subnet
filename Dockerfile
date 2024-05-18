FROM ollama/ollama:0.1.38

# override entrypoint
ENTRYPOINT []
CMD ""

ENV PYTHONFAULTHANDLER=1 \
PYTHONUNBUFFERED=1 \
PYTHONHASHSEED=random \
PIP_NO_CACHE_DIR=off \
PIP_DISABLE_PIP_VERSION_CHECK=on \
PIP_DEFAULT_TIMEOUT=100 \
POETRY_NO_INTERACTION=1 \
POETRY_VIRTUALENVS_CREATE=false \
POETRY_CACHE_DIR='/var/cache/pypoetry' \
POETRY_HOME='/usr/local' \
POETRY_VERSION=1.8.2

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 605C66F00D6C9793 \
0E98404D386FA1D9 648ACFD622F3D138 871920D1991BC93C
RUN apt-get update && \
    apt-get install -y git curl python3-pip python3-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /code
COPY poetry.lock pyproject.toml /code/

RUN poetry install --only=main --no-interaction --no-ansi --no-root

COPY . /code
