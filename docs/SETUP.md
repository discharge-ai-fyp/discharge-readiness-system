# Developer Setup Guide

Status: DRAFT v0.1 | Last updated: 2026-10-11
Covers: Windows 10/11, macOS, Linux (Debian/Ubuntu-type).
The Linux steps were tested. The Windows and macOS steps were NOT tested yet. If something fails,
fix this file in a PR so the next person does not hit the same problem.

## 0. Pick your level
| Level | You can | Time |
|---|---|---|
| 1. Quick | run all tests (ML tests + criteria tests) | about 15 min |
| 2. Full ML | also download data, train models, run evaluation and SHAP | about 45 min |

Not needed yet: Docker, Node.js. The backend and frontend are not built (planned).

## 1. Get access to the repository
1. Have a GitHub account.
2. Ask Sufiyan to invite you to the repo with the **Write** role (not Admin). Accept the email invite.
3. Repo: https://github.com/discharge-ai-fyp/discharge-readiness-system

## 2. Install the tools

### Windows (PowerShell, not cmd)
Open PowerShell and run:
```powershell
winget install --id Git.Git -e
winget install --id GitHub.cli -e
winget install --id Microsoft.VisualStudioCode -e
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Close and reopen PowerShell, then check:
```powershell
git --version
gh --version
uv --version
```
Set line endings (Windows only):
```powershell
git config --global core.autocrlf true
```
If winget is missing, install Git for Windows, GitHub CLI and VS Code from their websites.

### macOS (Terminal)
```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git gh uv
brew install --cask visual-studio-code
git config --global core.autocrlf input
```
Skip the xcode and Homebrew lines if they are already installed. Check with `git --version`,
`gh --version`, `uv --version`.

### Linux
```bash
sudo apt update && sudo apt install -y git unzip curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Restart the terminal. Install GitHub CLI from https://cli.github.com if you want `gh`.

### All systems: tell git who you are
```bash
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

## 3. Log in to GitHub
Easiest, all systems:
```bash
gh auth login
```
Choose GitHub.com, HTTPS, "Login with a web browser", and finish in the browser.
On Windows, Git for Windows also opens a browser login the first time you push.

Rules:
- Never paste a token or password into chat, a file in the repo, or a commit.
- Do not use `git config credential.helper store` (it saves your token in plain text).
- If a push is rejected with a message about the `workflow` scope, you changed a file in
  `.github/workflows/`. Run `gh auth refresh -s workflow`. (Not verified by us; check the gh docs if it fails.)

## 4. Clone the repo and switch to dev
Choose a folder without spaces in the path, for example `C:\Users\you\fyp` or `~/fyp`.
```bash
git clone https://github.com/discharge-ai-fyp/discharge-readiness-system.git
cd discharge-readiness-system
git checkout dev
git pull
```
All work starts from `dev`, never from `main`.

## 5. Create the Python 3.12 environment
Python 3.12 is pinned everywhere (local, CI, Docker). Saved models can fail to load on another
Python version, so do not use another version.

Windows (PowerShell, from the repo root):
```powershell
uv venv --python 3.12 ml\venv312
.\ml\venv312\Scripts\Activate.ps1
python --version
```
If PowerShell blocks the script, run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
and activate again. In cmd.exe use `ml\venv312\Scripts\activate.bat` instead.

macOS / Linux (from the repo root):
```bash
uv venv --python 3.12 ml/venv312
source ml/venv312/bin/activate
python --version
```
`python --version` must say 3.12.x, and your prompt should show the venv name.
You must activate the environment in every new terminal.

## 6. Level 1: install test tools and run the tests
With the environment active, from the repo root:
```bash
uv pip install -r ml/requirements-ci.txt -r criteria/requirements-ci.txt
ruff check ml/src ml/tests criteria --select F,E9
python -m pytest criteria -q
cd ml
python -m pytest -q
cd ..
```
Expected: ruff says "All checks passed!", criteria tests report **29 passed**, ML tests report
**3 passed**. If your numbers differ, tell the team before changing code.

## 7. Level 2: full ML setup

### 7.1 Install all ML packages
From the repo root, environment active:
```bash
uv pip install -r ml/requirements.txt
uv pip install pytest ruff
```
This installs about 128 packages (pandas, scikit-learn, xgboost, shap, jupyter, and others). It can
take several minutes. `requirements.txt` was created on Python 3.14 and installs cleanly on 3.12 on
Linux; Windows and macOS are untested.

macOS: if xgboost fails with an error that mentions `libomp`, run `brew install libomp` and try again.

### 7.2 Download the data
Data is never stored in git. Each person downloads it.
1. Open https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. Click Download. The file is named `diabetes+130-us+hospitals+for+years+1999-2008.zip`
   (your browser may use another name, such as `archive.zip`).
3. Extract it so that these two files sit in `ml/data/raw/uci/`:
   - `diabetic_data.csv` (about 18 MB)
   - `IDS_mapping.csv`

Windows (PowerShell, from the repo root; edit the zip name if different):
```powershell
New-Item -ItemType Directory -Force ml\data\raw\uci
Expand-Archive -Path "$HOME\Downloads\diabetes+130-us+hospitals+for+years+1999-2008.zip" -DestinationPath ml\data\raw\uci
dir ml\data\raw\uci
```
macOS / Linux:
```bash
mkdir -p ml/data/raw/uci
unzip ~/Downloads/"diabetes+130-us+hospitals+for+years+1999-2008.zip" -d ml/data/raw/uci
ls -lh ml/data/raw/uci
```
macOS Safari may unzip files automatically; then just move the two CSV files into the folder.
If the zip creates a sub-folder, move the CSV files up into `ml/data/raw/uci/`.
Then run `git status`. `ml/data` must NOT appear. If it does, stop and ask.

Optional: the Kaggle synthetic dataset (demo only, not needed for the main work):
https://www.kaggle.com/datasets/sergionefedov/patient-records-100k-patients-15-conditions
Its page states that the data is synthetic.

### 7.3 Train, evaluate and explain
Always run these from the `ml` folder, never from `ml/src`:
```bash
cd ml
python -m src.train facility xgb
python -m src.train facility logreg
python -m src.train readmit xgb
python -m src.train readmit logreg
python -m src.evaluate facility xgb
python -m src.explain facility xgb
```
Each training run takes a few minutes. Reference values from Sufiyan's run (Linux):

| Run | Expected |
|---|---|
| facility xgb | test_auc about 0.799, test_brier about 0.132 |
| facility logreg | test_auc about 0.792 |
| readmit xgb | test_auc about 0.648 |
| explain facility xgb | line `CONSISTENCY` with a difference around 1e-07 and mean p about 0.219 |

Small differences on another operating system are possible (we have not tested). A big difference,
such as AUC below 0.75 for facility xgb, means something is wrong: tell the team.
Trained models are saved in `ml/models/` and are gitignored. Reports are written to `ml/reports/`.

### 7.4 Notebooks (optional)
```bash
cd ml
jupyter lab
```
Run it from the `ml` folder, because the notebooks read `../data/raw/...`. Before committing a notebook:
```bash
jupyter nbconvert --clear-output --inplace notebooks/02_uci_audit.ipynb
```
The CI check rejects notebooks that still contain outputs.

## 8. Daily workflow
1. `git branch --show-current` before you create or edit any file. If it says `dev` or `main`, create a branch first.
2. `git checkout dev && git pull && git checkout -b feature/short-name`
3. Branch types: `feature/`, `fix/`, `docs/`, `ci/`, `chore/`.
4. Work. Check what changed: `git status --short -uall`
5. Stage only the files you mean to commit: `git add path/to/file`. Never `git add .`
6. Commit with a prefix: `feat:`, `fix:`, `docs:`, `chore:`, `ci:`
   Example: `git commit -m "docs: add setup guide"`
7. `git push -u origin your-branch`
8. Open a pull request on GitHub with base **dev**. Wait for the green checks (Actions tab). Then merge.
9. `git checkout dev && git pull` and delete the merged branch: `git branch -d your-branch`

Merging `dev` into `main` is a milestone step done by Sufiyan.

## 9. Never commit
- Data files (`.csv`), trained models (`.joblib`, `.pkl`), virtual environments, `.env` files,
  keys (`.pem`, `.key`), tokens, passwords.
- The ML CI check fails the PR if a `.csv`, `.joblib`, `.pkl`, `.pem`, `.key` or `.env` file is tracked.
- If a secret is ever committed, tell Sufiyan at once. The secret must be revoked and replaced;
  deleting the line is not enough because it stays in git history.
- Workflow files must sit directly in `.github/workflows/`. A copy in a nested folder is silently ignored.

## 10. Troubleshooting
| Problem | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | You are not in the `ml` folder. `cd ml` first. |
| `ModuleNotFoundError: No module named 'criteria'` | Run the criteria tests from the repo root: `python -m pytest criteria -q` |
| `python` is not 3.12 | Environment not active. Activate it again (section 5). |
| PowerShell: running scripts is disabled | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `uv` or `git` not found after install | Close and reopen the terminal. |
| `mkdir -p` or `source` fails on Windows | Those are macOS/Linux commands. Use the PowerShell versions above. |
| `FileNotFoundError` for `diabetic_data.csv` | The CSV is not in `ml/data/raw/uci/`. Check section 7.2 and the file name. |
| xgboost error mentioning `libomp` (macOS) | `brew install libomp` |
| Push rejected: workflow scope | `gh auth refresh -s workflow` |
| "externally-managed-environment" (Linux) | You used the system pip. Activate the venv first. |
| Windows path errors or very long paths | Clone into a short path such as `C:\fyp`. |
| Tests pass locally but fail on GitHub | CI uses Python 3.12 and a fresh install. Compare your environment with section 5 and 6. |

## 11. Quick reference
| What | Command |
|---|---|
| Activate (Windows PowerShell) | `.\ml\venv312\Scripts\Activate.ps1` |
| Activate (macOS/Linux) | `source ml/venv312/bin/activate` |
| Run criteria tests | `python -m pytest criteria -q` (repo root) |
| Run ML tests | `cd ml` then `python -m pytest -q` |
| Lint | `ruff check ml/src ml/tests criteria --select F,E9` |
| Train | `cd ml` then `python -m src.train facility xgb` |
| Update dev | `git checkout dev && git pull` |

## 12. Ask for help with this information
Your operating system, the exact command, and the full error text (copy it, do not retype it).
Never include tokens or passwords.
