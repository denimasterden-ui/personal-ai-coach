#!/usr/bin/env bash
# Safe live deploy for AICOACH.
#   sync code → wait for a quiet window (no in-flight model request) → restart → verify.
# Config (host/key/units) lives in deploy/deploy.env (untracked). See deploy.env.example.
#
#   ./deploy/deploy.sh            # gated: aborts if requests are in flight
#   ./deploy/deploy.sh --force    # skip the wait (graceful drain still finishes them)
set -euo pipefail

cd "$(dirname "$0")/.."                      # repo root (aicoach-service/)
CFG="deploy/deploy.env"
[ -f "$CFG" ] || { echo "нет $CFG — скопируй deploy/deploy.env.example, заполни, повтори"; exit 1; }
# shellcheck disable=SC1090
source "$CFG"
: "${HOST:?} ${SSH_USER:?} ${SSH_KEY:?} ${REMOTE_DIR:?} ${HEALTH_PORT:?} ${UNIT_SERVICE:?} ${UNIT_BOT:?}"
MAX_TRIES="${MAX_TRIES:-20}"

FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
SSHKEY_E="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"
run() { $SSHKEY_E "$SSH_USER@$HOST" "$@"; }

echo "▶ 1/4 sync кода → $HOST:$REMOTE_DIR"
rsync -az \
  --exclude='.env' --exclude='deploy/deploy.env' --exclude='tenants' \
  --exclude='*.pkl' --exclude='*.enc' --exclude='.venv' --exclude='abtest' \
  --exclude='*.db' --exclude='*.db-*' --exclude='.git' --exclude='__pycache__' \
  -e "$SSHKEY_E" ./ "$SSH_USER@$HOST:$REMOTE_DIR/"

echo "▶ 2/4 жду тихое окно (active=0)…"
tries=0
while :; do
  active=$(run "curl -s --max-time 5 http://127.0.0.1:$HEALTH_PORT/health" \
    | python3 -c 'import sys,json;print(json.load(sys.stdin).get("active","?"))' 2>/dev/null || echo "?")
  [ "$active" = "0" ] && { echo "  active=0 — окно чистое"; break; }
  tries=$((tries+1))
  if [ "$tries" -ge "$MAX_TRIES" ]; then
    [ "$FORCE" = 1 ] && { echo "  таймаут, --force → продолжаю (graceful drain доиграет)"; break; }
    echo "  active=$active уже $MAX_TRIES раз — идут запросы к модели. Прерываю. Повтори позже или --force."; exit 2
  fi
  echo "  active=$active, жду 3с… ($tries/$MAX_TRIES)"; sleep 3
done

echo "▶ 3/4 рестарт: $UNIT_SERVICE → $UNIT_BOT"
run "systemctl restart $UNIT_SERVICE && sleep 4 && systemctl restart $UNIT_BOT && sleep 3"

echo "▶ 4/4 проверка"
run "
  echo -n '  service: '; systemctl is-active $UNIT_SERVICE
  echo -n '  bot:     '; systemctl is-active $UNIT_BOT
  echo -n '  health:  '; curl -s http://127.0.0.1:$HEALTH_PORT/health; echo
  journalctl -u $UNIT_SERVICE -n 15 --no-pager 2>/dev/null | grep -a 'sessions' | tail -2 | sed 's/^/  /'
  journalctl -u $UNIT_BOT -n 8 --no-pager 2>/dev/null | grep -aE 'bot up|shutdown' | tail -2 | sed 's/^/  /'
"
echo "✔ деплой готов. Откат: откати код в git и запусти ./deploy/deploy.sh снова."
