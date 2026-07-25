#!/usr/bin/env bash
# Referential-integrity + row-count checker for the scaled TPC-H datasets.
#
# Scans every  <family>/<scale>/data  directory under this folder
# (all_scaled/*, lineitem_scaled/*) and checks that every foreign key resolves
# inside its own directory. Exits non-zero if any key dangles.
#
# Usage:  ./verify_integrity.sh                      # every data dir found
#         ./verify_integrity.sh lineitem_scaled/240K/data   # explicit dir(s)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0

# fk_check <child.tbl> <child_field(s), 1-based, comma-joined> <parent.tbl> <parent_field(s)> <label>
fk_check() {
  local child="$1" cf="$2" parent="$3" pf="$4" label="$5"
  local miss
  miss=$(awk -F'|' -v cf="$cf" -v pf="$pf" '
    BEGIN { ncf=split(cf,ca,","); npf=split(pf,pa,",") }
    NR==FNR { k=""; for(i=1;i<=npf;i++) k=k SUBSEP $(pa[i]); p[k]=1; next }
    { k=""; for(i=1;i<=ncf;i++) k=k SUBSEP $(ca[i]); if(!(k in p)) c++ }
    END { print c+0 }
  ' "$parent" "$child")
  if [ "$miss" -eq 0 ]; then
    printf "    OK   %-52s (all keys resolve)\n" "$label"
  else
    printf "    FAIL %-52s (%d dangling)\n" "$label" "$miss"; fail=1
  fi
}

if [ "$#" -gt 0 ]; then
  DIRS=("$@")
else
  mapfile -t DIRS < <(find "$HERE" -type d -name data | sort)
fi

for d in "${DIRS[@]}"; do
  rel="${d#"$HERE"/}"
  echo "=== $rel ==="
  [ -d "$d" ] || { echo "    missing dir"; fail=1; continue; }
  echo "  row counts:"
  for t in region.cvs nation.tbl supplier.tbl customer.tbl part.tbl partsupp.tbl orders.tbl lineitem.tbl; do
    [ -f "$d/$t" ] && printf "    %-14s %8d\n" "$t" "$(wc -l < "$d/$t")"
  done
  echo "  foreign keys:"
  fk_check "$d/lineitem.tbl" 1   "$d/orders.tbl"   1   "lineitem.orderkey -> orders"
  fk_check "$d/lineitem.tbl" 2   "$d/part.tbl"     1   "lineitem.partkey  -> part"
  fk_check "$d/lineitem.tbl" 3   "$d/supplier.tbl" 1   "lineitem.suppkey  -> supplier"
  fk_check "$d/lineitem.tbl" 2,3 "$d/partsupp.tbl" 1,2 "lineitem.(part,supp) -> partsupp"
  fk_check "$d/orders.tbl"   2   "$d/customer.tbl" 1   "orders.custkey    -> customer"
  fk_check "$d/customer.tbl" 4   "$d/nation.tbl"   1   "customer.nationkey-> nation"
  fk_check "$d/supplier.tbl" 4   "$d/nation.tbl"   1   "supplier.nationkey-> nation"
  fk_check "$d/partsupp.tbl" 1   "$d/part.tbl"     1   "partsupp.partkey  -> part"
  fk_check "$d/partsupp.tbl" 2   "$d/supplier.tbl" 1   "partsupp.suppkey  -> supplier"
  fk_check "$d/nation.tbl"   3   "$d/region.cvs"   1   "nation.regionkey  -> region"
  echo
done

if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit $fail
