#!/usr/bin/env bash
# Waits for push_code_changes.sh to finish, then configures sparse checkout
# to keep only:
#   - all code (everything outside oso_airfoils/data/cases_*)
#   - oso_airfoils/data/cases_101_to_110/
#   - oso_airfoils/data/cases_111_to_120/
#   - oso_airfoils/data/*/performance_data/ (reference airfoil JSON polars)
#
# After running, the old case directories (cases_0 through cases_91_to_99, cases_100)
# will be absent from the working tree but remain in git history on GitHub.

CODE_LOG=/tmp/push_code.log
REPO="$(cd "$(dirname "$0")" && pwd)"
SENTINEL="Code push complete."

echo "[$(date '+%H:%M:%S')] Waiting for code push to finish (watching $CODE_LOG)..."

while true; do
    if grep -qF "$SENTINEL" "$CODE_LOG" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] Code push done. Setting up sparse checkout..."
        break
    fi
    if ! pgrep -f "push_code_changes.sh" > /dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] push_code_changes.sh process ended. Proceeding..."
        break
    fi
    sleep 30
done

cd "$REPO" || exit 1

# Switch to no-cone mode (supports wildcard patterns)
git sparse-checkout init --no-cone

# Write the sparse checkout rules:
#   /*                        — include everything by default
#   !/oso_airfoils/data/...   — exclude old case directories
#   /oso_airfoils/data/cases_101_to_110/  — re-include 101-110
#   /oso_airfoils/data/cases_111_to_120/  — re-include 111-120
# The family dirs (du/, ffa/, riso_*, etc.) are not cases_* so they
# remain included automatically — performance_data/ JSONs are kept.

git sparse-checkout set --no-cone \
    '/*' \
    '!/oso_airfoils/data/cases_0/' \
    '!/oso_airfoils/data/cases_01_to_10/' \
    '!/oso_airfoils/data/cases_100/' \
    '!/oso_airfoils/data/cases_11_to_20/' \
    '!/oso_airfoils/data/cases_21_to_30/' \
    '!/oso_airfoils/data/cases_31_to_40/' \
    '!/oso_airfoils/data/cases_41_to_50/' \
    '!/oso_airfoils/data/cases_51_to_60/' \
    '!/oso_airfoils/data/cases_61_to_70/' \
    '!/oso_airfoils/data/cases_71_to_80/' \
    '!/oso_airfoils/data/cases_81_to_90/' \
    '!/oso_airfoils/data/cases_91_to_99/' \
    '/oso_airfoils/data/cases_101_to_110/' \
    '/oso_airfoils/data/cases_111_to_120/'

echo "[$(date '+%H:%M:%S')] Sparse checkout rules written. Applying..."

git sparse-checkout reapply

echo "[$(date '+%H:%M:%S')] Done. Checking removed directories..."

# Report what's gone
for d in cases_0 cases_01_to_10 cases_100 cases_11_to_20 cases_21_to_30 \
          cases_31_to_40 cases_41_to_50 cases_51_to_60 cases_61_to_70 \
          cases_71_to_80 cases_81_to_90 cases_91_to_99; do
    if [ ! -d "oso_airfoils/data/$d" ]; then
        echo "  Removed locally: oso_airfoils/data/$d"
    else
        echo "  WARNING: still present: oso_airfoils/data/$d"
    fi
done

echo ""
echo "[$(date '+%H:%M:%S')] Sparse checkout complete."
echo "  Kept locally: oso_airfoils/data/cases_101_to_110/"
echo "  Kept locally: oso_airfoils/data/cases_111_to_120/"
echo "  Kept locally: all performance_data/ JSON dirs (du, ffa, riso_*, s, oso_*)"
echo "  All removed dirs still exist in git on GitHub."
echo ""
echo "  To restore a removed dir: git sparse-checkout add oso_airfoils/data/<dir>/"
