#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="langchain"
ENV_FILE="$PROJECT_DIR/environment.yml"

usage() {
  cat <<'EOF'
Kanana Schedule Agent runner

Usage:
  ./run.sh                 Run the Week 1 Gradio app
  ./run.sh --week1         Run the Week 1 Gradio app
  ./run.sh --week2         Run the Week 2 Gradio app
  ./run.sh --install       Run uv sync, then run the Week 1 Gradio app
  ./run.sh --conda [ARGS]  Use the legacy conda environment.yml runner
  ./run.sh --help          Show this help

First-time setup:
  ./run.sh --install
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

cd "$PROJECT_DIR"

run_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv를 찾을 수 없습니다. https://docs.astral.sh/uv/ 에서 uv를 먼저 설치해주세요." >&2
    exit 1
  fi

  local active_week="1"
  if [[ "${1:-}" =~ ^--week([12])$ ]]; then
    active_week="${BASH_REMATCH[1]}"
    shift
  elif [[ "${1:-}" =~ ^--week[0-9]+$ ]]; then
    echo "main 브랜치는 Week 1-2만 포함합니다. Week 3-6은 week_1_to_6f 브랜치를 사용하세요." >&2
    exit 1
  fi
  export KANANA_ACTIVE_WEEK="$active_week"
  export PYTHONNOUSERSITE=1

  case "${1:-}" in
    "")
      uv run python app.py
      ;;
    --install)
      uv sync
      uv run python app.py
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "알 수 없는 옵션입니다: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

run_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda를 찾을 수 없습니다. Miniconda 또는 Anaconda를 먼저 설치해주세요." >&2
    exit 1
  fi

  local active_week="1"
  if [[ "${1:-}" =~ ^--week([12])$ ]]; then
    active_week="${BASH_REMATCH[1]}"
    shift
  elif [[ "${1:-}" =~ ^--week[0-9]+$ ]]; then
    echo "main 브랜치는 Week 1-2만 포함합니다. Week 3-6은 week_1_to_6f 브랜치를 사용하세요." >&2
    exit 1
  fi
  export KANANA_ACTIVE_WEEK="$active_week"

  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"

  if [[ "${1:-}" == "--install" ]]; then
    if conda_env_exists; then
      echo "Updating conda env: $ENV_NAME"
      conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    else
      echo "Creating conda env: $ENV_NAME"
      conda env create -f "$ENV_FILE"
    fi
  elif ! conda_env_exists; then
    echo "conda env '$ENV_NAME'가 없어 environment.yml로 새로 만듭니다."
    conda env create -f "$ENV_FILE"
  fi

  conda activate "$ENV_NAME"
  export PYTHONNOUSERSITE=1

  case "${1:-}" in
    "")
      python app.py
      ;;
    --install)
      python app.py
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "알 수 없는 conda 옵션입니다: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

if [[ "${1:-}" == "--conda" ]]; then
  shift
  run_conda "$@"
else
  run_uv "$@"
fi
