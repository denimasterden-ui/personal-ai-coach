# Деплой AICOACH 24/7 (systemd)

Разворачивание на любом Linux-сервере как два systemd-сервиса (мозг + бот) с автозапуском.
Изолированный периметр: `/opt/aicoach`, права `700` на данные.

## Предпосылки
- Linux-сервер с исходящим интернетом (бот работает по long-polling, публичный URL НЕ нужен).
- Python 3.11+.
- OpenAI-совместимый LLM-endpoint (OpenRouter / свой LiteLLM / локальный Ollama).

## Шаги

1. **Код на сервер** (без .venv/.env/tenants):
   ```
   rsync -av --exclude .venv --exclude .env --exclude tenants --exclude __pycache__ \
     aicoach-service/ SERVER:/opt/aicoach/aicoach-service/
   ```

2. **Окружение на сервере**:
   ```
   cd /opt/aicoach/aicoach-service
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   cp deploy/env.example .env && chmod 600 .env    # заполнить ключи
   mkdir -p tenants && chmod 700 tenants
   ```

3. **Проверить модель** (пример для OpenAI-совместимого endpoint):
   ```
   set -a; . ./.env; set +a
   curl -s "$LLM_BASE_URL/models" -H "Authorization: Bearer $LLM_API_KEY" | head
   # убедиться, что LLM_MODEL есть в списке
   ```

4. **systemd** — юниты предполагают путь `/opt/aicoach/aicoach-service`
   (поправьте `WorkingDirectory`/пути, если разворачиваете в другом месте):
   ```
   cp deploy/aicoach-service.service deploy/aicoach-bot.service /etc/systemd/system/
   systemctl daemon-reload
   systemctl enable --now aicoach-service aicoach-bot
   systemctl status aicoach-service aicoach-bot --no-pager
   ```

5. **Проверка** — напишите/наговорите боту. Логи:
   ```
   journalctl -u aicoach-bot -f
   ```

## Откат
```
systemctl disable --now aicoach-bot aicoach-service
```
Профиль в `tenants/<id>/*.md` при этом остаётся.

## Заметка по изоляции (hardening)
По умолчанию — под тем пользователем, что запускает systemd. Для строгой изоляции личных
данных заведите отдельного системного пользователя `aicoach` с владением только своим каталогом.
