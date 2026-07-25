#!/usr/bin/env bash
# Run the TPC-H queries q3 q5 q8 q9 q18 across all six scaled table groups
# (all_scaled x {60K,120K,240K}  and  lineitem_scaled x {60K,120K,240K}).
#
# Each group is a separate `vpjoin_bench` process pointed at that group's tables
# via VPJOIN_TABLES (which swaps only the .tbl files, so proof params stay under
# the crate), tagged with VPJOIN_LABEL so its rows are identifiable, and written
# to its own CSV under results/scaling/. A combined CSV is stitched at the end.
#
# Usage:
#   ./run_sweep.sh                 # mode=full (prove + commitment layer)
#   ./run_sweep.sh baseline        # prove/verify only, no commitment layer
#   ./run_sweep.sh commit          # commitment-layer timings
#   VPJOIN_PLAN_ONLY=1 ./run_sweep.sh    # dry run: print planned degrees only
#
# Q5 uses VPJOIN_PRIVACY=rjs (zero padding, deterministic) so the curve reflects
# data size alone; the other four queries materialize nothing and ignore it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

MODE="${1:-full}"                       # baseline | full | commit
QUERIES=(q3 q5 q8 q9 q18)
OUT="${VPJOIN_OUT_DIR:-$REPO/results/scaling}"
mkdir -p "$OUT"

# VPJOIN_PROFILE=debug uses the unoptimized profile (same as `cargo test`, much
# slower to prove); default is the optimized release profile.
PROFILE="${VPJOIN_PROFILE:-release}"
if [ "$PROFILE" = "debug" ]; then RELFLAG=(); TDIR=debug; else RELFLAG=(--release); TDIR=release; fi

# label : data root (relative to repo)
GROUPS=(
  "all-60K:src/new_data/all_scaled/60K"
  "all-120K:src/new_data/all_scaled/120K"
  "all-240K:src/new_data/all_scaled/240K"
  "lineitem-60K:src/new_data/lineitem_scaled/60K"
  "lineitem-120K:src/new_data/lineitem_scaled/120K"
  "lineitem-240K:src/new_data/lineitem_scaled/240K"
)

echo "building vpjoin_bench ($PROFILE) ..."
cargo build "${RELFLAG[@]}" --bin vpjoin_bench
BIN="$REPO/target/$TDIR/vpjoin_bench"

for g in "${GROUPS[@]}"; do
  label="${g%%:*}"; root="$REPO/${g#*:}"
  csv="$OUT/scaling_${label}.csv"
  echo ""
  echo "############################################################"
  echo "# $label   data=$root"
  echo "############################################################"
  VPJOIN_TABLES="$root/data" VPJOIN_LABEL="$label" VPJOIN_PRIVACY=rjs \
    "$BIN" "$MODE" "$csv" "${QUERIES[@]}"
done

# stitch a combined CSV (header once, then every group's rows)
COMBINED="$OUT/scaling_ALL.csv"
first="$OUT/scaling_all-60K.csv"
if [ -f "$first" ]; then
  { head -1 "$first"; for f in "$OUT"/scaling_*.csv; do [ "$f" = "$COMBINED" ] && continue; tail -n +2 "$f"; done; } > "$COMBINED"
  echo ""
  echo "combined results -> $COMBINED"
fi
