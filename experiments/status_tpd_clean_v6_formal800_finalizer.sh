#!/usr/bin/env bash
set -euo pipefail

v6_repo="/home/ly/SCTransNet_main"
v6_python="/home/ly/BasicIRSTD/infrarenet/bin/python"
v6_unit="sctransnet-tpd-clean-v6-formal800-finalizer.service"
v6_comparison="$v6_repo/experiments/results/tpd_clean_v6_formal800_2x5090_v1/NUDT-SIRST/comparison"

systemctl --user show "$v6_unit" \
    --property=ActiveState,SubState,Result,NRestarts \
    --no-pager 2>/dev/null || true
"$v6_python" "$v6_repo/experiments/summarize_tpd_clean_v6_formal800.py" --preflight
for v6_name in \
    tpd_clean_v6_formal800_comparison.json \
    tpd_clean_v6_formal800_comparison.md \
    completion_inputs.json \
    COMPLETE.sha256; do
    v6_path="$v6_comparison/$v6_name"
    if [[ -f "$v6_path" && ! -L "$v6_path" ]]; then
        echo "TPDCLEANV6_FINALIZER_ARTIFACT name=$v6_name sha256=$(sha256sum "$v6_path" | awk '{print $1}')"
    else
        echo "TPDCLEANV6_FINALIZER_ARTIFACT name=$v6_name status=missing"
    fi
done
