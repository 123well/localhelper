# KDE Agent — автономный агент для Arch Linux + Plasma 6 (Stage-1, Python)

Прототип агента: SSM-модель (7B INT4 на NPU в продакшене, MockBackend для отладки),
система инициативы с адаптивными порогами и дофаминовой наградой, внешняя память/RAG,
событийный мониторинг и обучение LoRA-адаптеров во время сна. Фактические знания
(документация, синтаксис, команды) не хранятся в весах — только через RAG.

## Требования
- Arch Linux, Python ≥ 3.10, `numpy`
- Опционально: `dbus-next` (D-Bus-уведомления; иначе fallback на qdbus6/kdialog/лог),
  `socat` (shell-hook подсказок), `onnxruntime` + драйвер amdxdna (NPU-инференс)

```bash
pip install numpy dbus-next          # или pacman -S python-numpy && pip install dbus-next
```

Без опциональных зависимостей агент полностью работоспособен headless
(MockBackend, уведомления деградируют в лог/stdin).

## Быстрый старт
```bash
python scripts/doctor.py                      # диагностика окружения + сквозной самотест
python -m agent.cli demo                      # scripted end-to-end: опечатки, veto, safety, RAG, сон
python -m unittest discover -s tests -v       # юнит-тесты (17 шт.)
python -m agent.cli run                       # интерактивный режим (терминал как UI)
```

Ожидаемый результат: `DEMO OK`, все тесты `OK`, doctor без FAIL
(WARN про LLM backend / NPU — это нормально до подключения реальных весов).

## Slash-команды в режиме `run`
| Команда | Действие |
|---|---|
| `/feedback <принял\|отклонил\|намеренно\|раздражён>` | явная реакция на последнюю подсказку (награда/штраф) |
| `/missed` | «агент пропустил очевидную ошибку» → отрицательный сигнал, порог снижается |
| `/sleep` | ручной цикл обучения адаптеров + гейты качества |
| `/index <папка>` | импорт документации (*.md/*.txt/*.rst) в RAG |
| `/tools` | список доступных инструментов модели |
| `/state` | контекст, пороги вмешательства, статистика полезности |
| `/undo` | откат последнего действия агента |
| `/quit` | корректное завершение |

Любая другая строка трактуется как пользовательский запрос; строка, начинающаяся
с `$` или `!`, — как введённая пользователем команда терминала (проходит через
детекторы опечаток и safety-gate).

## Изучение документации (RAG)
```bash
python -m agent.cli index ~/docs        # ~/.local/share/kde-agent/memory.db
```

## Хук терминала (подсказки об опечатках в реальном времени)
```bash
python scripts/install_shell_hook.py --bash   # или --zsh ; удаление: --remove
```

## Конфигурация — `~/.config/kde-agent/config.json`
- `model.npu_model_path` — путь к `mamba2-7b-int4.onnx` (XDNA EP);
- `model.cpu_model_path` — GGUF для отладки на CPU/GPU;
- `initiative.typo_use_model=false` — форсировать детерминированный mock-анализатор
  (по умолчанию evidence-packet → LLM-анализатор с fallback);
- `sleep.enabled=false` — отключить ночное обучение;
- `safety.blacklist` / `safety.whitelist` — жёсткие блокировки/разрешения;
- `energy.*` — пороги простоя/батареи для фонового обучения.

## Ночная тренировка адаптеров и Stage-2 (C++)
Автостарт при idle/simple (night scheduler) или вручную `/sleep`.
Экспорт обученного адаптера в INT4 ONNX-граф для C++ рантайма:
```bash
python scripts/export_adapters.py \
  --adapter ~/.local/share/kde-agent/adapters/adapter_XXX.npz \
  --model model.onnx --out model-a1.onnx
```

## Каталоги (XDG)
- конфиг: `~/.config/kde-agent/`
- данные/RAG/адаптеры: `~/.local/share/kde-agent/`
- логи/состояние: `~/.local/state/kde-agent/agent.log`

## Как работает детекция опечаток («программа предлагает — ИИ решает»)
1. **Proposer** (дешёвый, программный) собирает *пакеты доказательств*: неизвестный
   токен сверяется с PATH/seed-словарём, соседями по файловой системе (только
   реально существующие цели, близкие по взвешенному расстоянию AND jaccard ≥ 0.5),
   выученным словам пользователя и раскладкой клавиатуры (layout decode).
2. **Analyzer** (роль SSM-модели) взвешивает доказательства и выносит вердикт
   с confidence и объяснением; ниже порога abstain — молчание вместо назойливости.
   В продакшене пакет уходит в LLM-промпт (`LLMModelAnalyzer`), решение принимает модель.
3. Персональная история **никогда** не легитимизирует неизвестную команду
   (защита от самообучения на собственных опечатках); единственное молчание —
   явное вето пользователя «так задумано», которое хранится в prefs-памяти.
4. Реакции пользователя (принял/отказ/раздражение/пропуск ошибки) двигают
   персональный порог вмешательства каждого типа подсказки (RLReward-петля).

## Структура проекта
```
agent/            # ядро Stage-1
  cli.py main.py config.py demo.py
  core/           # initiative.py (петля инициатив), typo_model.py (proposer+analyzer), events, context
  memory/rag.py   # sqlite-векторное хранилище, collections docs/episodes/prefs, decay/prune
  tools/safety.py # risk-gate: blacklist/confirm/whitelist + undo journal
  training/sleep.py # LoRA-адаптеры + гейты (регрессия навыков / nagging / safety)
  kde/            # минимальная интеграция (уведомления/диалоги) — полная обёртка будет в C++
scripts/          # doctor.py, export_adapters.py, install_shell_hook.py
tests/test_agent.py
cpp/ plasmoid/    # каркасы Stage-2 (C++ Qt6-оболочка, KDE-виджет) — в разработке
```

## Безопасность
Опасные операции (rm системных путей, mkfs, sudo, mass-операции) требуют
подтверждения через диалог или блокируются чёрным списком; все действия пишутся
в аудит-лог; `/undo` откатывает последний файловый шаг; привилегий выше
пользовательских агент не запрашивает; секреты в базу знаний не индексируются.
