#!/usr/bin/env bash
# Watches /tmp/push_cases.log for completion of the large data push,
# then commits and pushes the code changes made during this session.

LOG=/tmp/push_cases.log
REPO="$(cd "$(dirname "$0")" && pwd)"
SENTINEL="All cases pushed successfully!"

echo "[$(date '+%H:%M:%S')] Waiting for data push to finish (watching $LOG)..."

while true; do
    if grep -qF "$SENTINEL" "$LOG" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] Data push complete. Committing code changes..."
        break
    fi
    # Also stop if push_cases.py process is no longer running and log exists
    if ! pgrep -f "push_cases.py" > /dev/null 2>&1; then
        if [ -f "$LOG" ]; then
            echo "[$(date '+%H:%M:%S')] push_cases.py process ended. Proceeding..."
            break
        fi
    fi
    sleep 60
done

cd "$REPO" || exit 1

# Stage only code/config changes — not data files
git add \
    oso_airfoils/postprocessing/oso_legacy.py \
    oso_airfoils/postprocessing/oso_polar.py \
    oso_airfoils/postprocessing/oso_bl.py \
    oso_airfoils/core/neuralfoil_wrapper.py \
    pyproject.toml

echo "[$(date '+%H:%M:%S')] Staged files:"
git diff --cached --name-only

git commit -m "postprocessing: add oso-legacy CLI, s20/s40 aliases, xxxlarge NF default

- oso_legacy.py: new combined legacy postprocessing CLI for cases 64-69
  (.txt population files). Outputs: objective/variable history plots,
  zoomed objective plot, Cp comparison (clean+rough), polar compare,
  airfoil evolution GIF. MPI-parallel frame rendering via mpirun.
- oso_polar.py / oso_bl.py: replace 's' alias with 's20' and 's40'
  sub-family tuples; _tau_match() gains allowed_stems param
- neuralfoil_wrapper.py: default model changed xxlarge -> xxxlarge
- pyproject.toml: add oso-legacy entry point"

echo "[$(date '+%H:%M:%S')] Pushing to origin/repackaging..."
git push origin repackaging-clean:repackaging

echo "[$(date '+%H:%M:%S')] Code push complete."
