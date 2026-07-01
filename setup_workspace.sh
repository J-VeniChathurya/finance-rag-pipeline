#!/usr/bin/env bash
###############################################################################
# setup_workspace.sh
#
# Purpose : Fully automate local project environment setup for a 7-day
#           high-performance MLOps sprint inside WSL2 Ubuntu.
#
# Usage   : bash setup_workspace.sh
#
# Notes   : - Script is idempotent: safe to re-run without breaking things.
#           - Uses 'set -euo pipefail' for fail-fast, safe execution.
###############################################################################

set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# 0. Helper functions for consistent, readable logging
# ---------------------------------------------------------------------------
log_info()  { echo -e "\e[34m[INFO]\e[0m  $1"; }
log_ok()    { echo -e "\e[32m[ OK ]\e[0m  $1"; }
log_warn()  { echo -e "\e[33m[WARN]\e[0m  $1"; }
log_err()   { echo -e "\e[31m[FAIL]\e[0m  $1"; }

# Root of the project = directory this script is executed from
PROJECT_ROOT="$(pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

log_info "Starting workspace bootstrap in: ${PROJECT_ROOT}"

###############################################################################
# 1. SYSTEM PACKAGE AUDITING
#    Check for python3-venv, pip, curl, git — install via apt if missing.
###############################################################################
log_info "Step 1/5: Auditing required system packages..."

REQUIRED_APT_PACKAGES=(
    "python3-venv"
    "python3-pip"
    "curl"
    "git"
)

# Detect which packages are actually missing before touching apt at all,
# so we don't run 'sudo apt update' unnecessarily on every run.
MISSING_PACKAGES=()

is_apt_package_installed() {
    dpkg -s "$1" >/dev/null 2>&1
}

for pkg in "${REQUIRED_APT_PACKAGES[@]}"; do
    if is_apt_package_installed "$pkg"; then
        log_ok "Package present: $pkg"
    else
        log_warn "Package missing: $pkg"
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ "${#MISSING_PACKAGES[@]}" -gt 0 ]; then
    log_info "Installing missing packages: ${MISSING_PACKAGES[*]}"
    sudo apt-get update -y
    sudo apt-get install -y "${MISSING_PACKAGES[@]}"
    log_ok "All missing system packages installed."
else
    log_ok "All required system packages already present. Skipping apt."
fi

###############################################################################
# 2. PRODUCTION FOLDER GENERATION
#    Build the exact directory hierarchy needed for the project.
###############################################################################
log_info "Step 2/5: Generating project folder hierarchy..."

REQUIRED_DIRS=(
    "data/raw"
    "src/core"
    "src/utils"
    "tests"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "${PROJECT_ROOT}/${dir}" ]; then
        log_ok "Directory already exists: ${dir}"
    else
        mkdir -p "${PROJECT_ROOT}/${dir}"
        log_ok "Created directory: ${dir}"
    fi
done

# Add .gitkeep placeholders so empty dirs are tracked by git
for dir in "${REQUIRED_DIRS[@]}"; do
    touch "${PROJECT_ROOT}/${dir}/.gitkeep"
done

###############################################################################
# 3. VIRTUAL ENVIRONMENT ORCHESTRATION
#    Create isolated .venv, upgrade pip, and generate requirements.txt.
###############################################################################
log_info "Step 3/5: Setting up Python virtual environment..."

if [ -d "${VENV_DIR}" ]; then
    log_warn ".venv already exists at ${VENV_DIR}. Reusing existing environment."
else
    python3 -m venv "${VENV_DIR}"
    log_ok "Virtual environment created at: ${VENV_DIR}"
fi

# Activate the venv for the remainder of this script's execution.
# (Only affects this script's subshell — does not alter the caller's shell.)
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log_ok "Virtual environment activated for script context."

log_info "Upgrading pip to the latest release inside .venv..."
pip install --upgrade pip
log_ok "pip upgraded to: $(pip --version)"

log_info "Writing requirements.txt with pinned production dependencies..."
cat > "${PROJECT_ROOT}/requirements.txt" << 'EOF'
# --- Core Framework & Pipelines ---
langchain>=1.2.0
langchain-community>=0.4.0
langchain-chroma>=0.1.0
langchain-text-splitters>=0.2.0
# --- Vector & Embeddings processing ---
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
tiktoken>=0.7.0
# --- Advanced Evaluation & Observability ---
ragas>=0.4.0
great-expectations>=1.11.0
langfuse>=2.0.0
# --- Document Handling ---
pdfplumber>=0.11.0
pypdf>=4.0.0
# --- Core App Plumbing ---
fastapi>=0.110.0
streamlit>=1.35.0
uvicorn>=0.30.0
python-dotenv>=1.0.1
pydantic>=2.7.0
EOF
log_ok "requirements.txt generated."

###############################################################################
# 4. DEPENDENCY PROVISIONING
#    Install all packages safely inside the activated virtual environment.
###############################################################################
log_info "Step 4/5: Installing dependencies from requirements.txt..."
log_warn "This may take several minutes (sentence-transformers, ragas, etc.)"

pip install --no-cache-dir -r "${PROJECT_ROOT}/requirements.txt"

log_ok "All dependencies installed successfully inside .venv."

###############################################################################
# 5. REPOSITORY BASELINE CREATION
#    Create .gitignore and .env.example boilerplate files.
###############################################################################
log_info "Step 5/5: Creating repository baseline files..."

# --- .gitignore ---
GITIGNORE_PATH="${PROJECT_ROOT}/.gitignore"

if [ -f "${GITIGNORE_PATH}" ]; then
    log_warn ".gitignore already exists. Leaving it untouched."
else
    cat > "${GITIGNORE_PATH}" << 'EOF'
# --- Environment & secrets ---
.env

# --- Python virtual environment ---
.venv/

# --- Python bytecode / cache ---
__pycache__/
*.pyc
*.pyo

# --- Vector store / local DB artifacts ---
data/chroma_db/

# --- Raw source PDFs (large / sensitive) ---
data/raw/*.pdf

# --- OS / editor cruft ---
.DS_Store
.vscode/
.idea/
EOF
    log_ok ".gitignore created."
fi

# --- .env.example ---
ENV_EXAMPLE_PATH="${PROJECT_ROOT}/.env.example"

if [ -f "${ENV_EXAMPLE_PATH}" ]; then
    log_warn ".env.example already exists. Leaving it untouched."
else
    cat > "${ENV_EXAMPLE_PATH}" << 'EOF'
# ---------------------------------------------------------------------------
# Environment variable template.
# Copy this file to ".env" and populate with real values.
# NEVER commit the actual ".env" file.
# ---------------------------------------------------------------------------

# --- LLM Providers ---
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# --- Observability (Langfuse) ---
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# --- Vector Store ---
CHROMA_DB_PATH=data/chroma_db

# --- App Config ---
APP_ENV=development
LOG_LEVEL=INFO
EOF
    log_ok ".env.example created."
fi

###############################################################################
# DONE
###############################################################################
deactivate 2>/dev/null || true

echo ""
log_ok "Workspace bootstrap complete."
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  source .venv/bin/activate"
echo "  2. Copy env template:         cp .env.example .env"
echo "  3. Populate .env with your real API keys/secrets."
echo ""
