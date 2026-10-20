#!/bin/bash
# DISCO-TAB watchdog: run periodically (cron). Detects when experiment jobs
# finish / crash / go idle and sends a notification (notify.sh) on state change.
# Stateful: compares to .watchdog_state so it only alerts on transitions.
cd /home/ec2-user/P-TA-MulFee/disco || exit 0
STATE=.watchdog_state
NOTIFY=./notify.sh
touch "$STATE"
prev() { grep -E "^$1=" "$STATE" 2>/dev/null | tail -1 | cut -d= -f2; }

# --- monitored jobs: key|pattern|label ---
JOBS=(
  "disco|run_all.py --mode main|DISCO-TAB training"
  "sdv|baselines.py --datasets|SDV baselines"
  "tabddpm|baselines_diffusion.py|TabDDPM baseline"
  "llm|baselines_llm.py|LLM baselines (GReaT/RealTabFormer)"
  "score|evaluate_baselines.py|baseline scoring"
  "ext|evaluate_ext.py|enriched-metrics (ext_)"
)

# --- current metrics ---
dv=$(python3 - <<'PY'
import json,glob,os
n=0
for p in glob.glob("results/*__disco__seed*.json"):
    try:
        if json.load(open(p)).get("n_syn",0)>=10: n+=1
    except: pass
print(n)
PY
)
extn=$(grep -l ext_c2st results/*.json 2>/dev/null | wc -l)
validn=$(python3 -c "import json,glob;print(sum(1 for p in glob.glob('results/*.json') if json.load(open(p)).get('n_syn',0)>=10))" 2>/dev/null)
anyrun=0
msgs=""
newstate=""

for entry in "${JOBS[@]}"; do
  key="${entry%%|*}"; rest="${entry#*|}"; pat="${rest%%|*}"; label="${rest##*|}"
  if pgrep -f "$pat" >/dev/null 2>&1; then alive=1; anyrun=1; else alive=0; fi
  was=$(prev "$key"); [ -z "$was" ] && was=0
  if [ "$was" = "1" ] && [ "$alive" = "0" ]; then
    msgs="${msgs}• ${label} STOPPED (finished or crashed) — DISCO ${dv}/72 valid, ext ${extn}.\n"
  fi
  newstate="${newstate}${key}=${alive}\n"
done

# milestone: DISCO reaches full coverage
pdv=$(prev discoN); [ -z "$pdv" ] && pdv=0
if [ "$dv" -ge 72 ] && [ "$pdv" -lt 72 ]; then
  msgs="${msgs}• MILESTONE: DISCO-TAB reached ${dv}/72 valid runs (all 24 datasets x 3 seeds).\n"
fi

# all idle (everything done)
pidle=$(prev allidle); [ -z "$pidle" ] && pidle=0
if [ "$anyrun" = "0" ]; then
  if [ "$pidle" = "0" ]; then
    msgs="${msgs}• ALL EXPERIMENT JOBS ARE IDLE — the run appears COMPLETE. DISCO ${dv}/72, enriched ${extn}/${validn}. Please review results and tell Claude the next tasks.\n"
  fi
  newstate="${newstate}allidle=1\n"
else
  newstate="${newstate}allidle=0\n"
fi

printf "%b" "$newstate" > "$STATE"
echo "discoN=${dv}" >> "$STATE"

if [ -n "$msgs" ]; then
  printf "%b" "$msgs" | sed 's/^/  /'
  $NOTIFY "DISCO-TAB experiments update" "$(printf "%b" "$msgs")$(date '+%Y-%m-%d %H:%M')" >/dev/null 2>&1
fi
