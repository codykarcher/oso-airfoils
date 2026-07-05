#!/usr/bin/env zsh
# push_datfiles.sh - Push airfoils/ datfiles + supporting_data family by family
# Each commit+push stays under the 2GB GitHub pack limit.
# Run from repo root.

set -e
cd /Users/codykarcher/Dropbox/research/oso-airfoils

REMOTE_REF="repackaging-clean:repackaging"

push_if_staged() {
    local label="$1"
    if git diff --cached --quiet; then
        echo "[SKIP] Nothing staged for: $label"
        return
    fi
    echo "[COMMIT] $label"
    git commit -m "Add datfiles/supporting_data: $label"
    echo "[PUSH]   $label"
    git push origin "$REMOTE_REF"
    echo "[DONE]   $label"
}

# ── Small families (all fit in one commit) ──────────────────────────────────
echo "=== Small families (du, ffa, s, riso_a, riso_b, riso_p, mhkf1, oso_2025_wt1) ==="
for fam in du ffa s riso_a riso_b riso_p mhkf1 oso_2025_wt1; do
    dir="oso_airfoils/airfoils/$fam"
    git add "$dir/Family_"*.png "$dir/datfiles/" 2>/dev/null || true
done
push_if_staged "small families - datfiles + family plots"

# ── oso_2026_wt3 (99MB - datfiles + supporting_data fits in one commit) ──────
echo "=== oso_2026_wt3 ==="
git add oso_airfoils/airfoils/oso_2026_wt3/datfiles/ \
        oso_airfoils/airfoils/oso_2026_wt3/supporting_data/ \
        oso_airfoils/airfoils/oso_2026_wt3/Family_*.png 2>/dev/null || true
push_if_staged "oso_2026_wt3 - datfiles + supporting_data"

# ── oso_2026_wt2s (1.2GB supporting_data - commit as one) ─────────────────
echo "=== oso_2026_wt2s ==="
git add oso_airfoils/airfoils/oso_2026_wt2s/datfiles/ \
        oso_airfoils/airfoils/oso_2026_wt2s/supporting_data/ \
        oso_airfoils/airfoils/oso_2026_wt2s/Family_*.png 2>/dev/null || true
push_if_staged "oso_2026_wt2s - datfiles + supporting_data"

# ── oso_2025_wt2 datfiles (small) ──────────────────────────────────────────
echo "=== oso_2025_wt2 datfiles ==="
git add oso_airfoils/airfoils/oso_2025_wt2/datfiles/ \
        oso_airfoils/airfoils/oso_2025_wt2/Family_*.png 2>/dev/null || true
push_if_staged "oso_2025_wt2 - datfiles + family plot"

# ── oso_2025_wt2/supporting_data (4.6GB - push each top-level subdir separately)
echo "=== oso_2025_wt2 supporting_data (batched by subdir) ==="
SUPP="oso_airfoils/airfoils/oso_2025_wt2/supporting_data"
for subdir in "$SUPP"/*/; do
    subname=$(basename "$subdir")
    subsize=$(du -sk "$subdir" | cut -f1)
    echo "  subdir: $subname  size: ${subsize}K"

    if (( subsize > 1500000 )); then
        # Still too big - push individual sub-subdirs
        echo "  -> splitting further..."
        for subsubdir in "$subdir"*/; do
            subsubname=$(basename "$subsubdir")
            git add "$subsubdir" 2>/dev/null || true
            push_if_staged "oso_2025_wt2/supporting_data/$subname/$subsubname"
        done
        # Any remaining files at subdir level
        git add "$subdir" 2>/dev/null || true
        push_if_staged "oso_2025_wt2/supporting_data/$subname (remaining files)"
    else
        git add "$subdir" 2>/dev/null || true
        push_if_staged "oso_2025_wt2/supporting_data/$subname"
    fi
done

echo ""
echo "All datfiles and supporting_data pushed!"
git log --oneline -5
