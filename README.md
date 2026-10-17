# kde-agent (localhelper)

Автономный локальный агент для Arch Linux + KDE Plasma 6. Железо-цель:
Ryzen AI 350 (NPU ~50 TOPS, amdxdna), RTX 5070 8GB, 32GB DDR5.
Приоритет — энергоэффективность (фон 20–35 ток/с допустим).

Рабочая модель: 7B SSM (Mamba-2/RWKV-6), INT4, на NPU. Факты НЕ в весах —
только RAG и живые внешние источники (ФС, PATH, история пользователя).

## Быстрый старт (Stage-1, чистый Python, без внешних зависимостей)

```bash
python -m agent.cli demo        # end-to-end прогон всех подсистем
python -m unittest discover -s tests   # 20 юнит-тестов, ожидаем OK
python scripts/doctor.py        # диагностика окружения (FAIL=0 = ок)
python -m agent.cli             # интерактивный режим (headless-safe)
```

Требования: Python 3.11+. numpy опционален (mock-энкодер умеет деградировать).
D-Bus/Plasma не обязательны: без сессии уведомления деградируют в лог/stdin.

## Архитектура (главный инвариант: «программа предлагает — ИИ решает»)

```
agent/
  main.py            Agent: событийная проводка (EventBus), подписки, аудит-лог
  cli.py             точка входа: demo | shell | run
  config.py          XDG-пути, пороги, бюджеты; всё отключаемо конфигом
  core/
    initiative.py    KNOWN_CMDS (seed-словарь), ProposalEngine->Analyzer->Verdict,
                     адаптивные пороги по типам, cooldown, почасовой бюджет подсказок
    typo_model.py    детектор опечаток: расстояния как EVIDENCE-признаки, layout
                     decode (йцукен<->qwerty), Lexicon, Mock/LLM ModelAnalyzer
    model.py         обёртка бэкенда LLM (в прототипе — mock-ответы той же формы)
    context.py       скользящий контекст + суммаризация старых окон (token-бюджет)
  memory/rag.py      sqlite+numpy: collections docs/episodes/prefs, decay/prune
  events/monitor.py  событийный мониторинг (idle, повторы); БЕЗ постоянных скриншотов
  tools/registry.py  исполняемые действия; safety.py: blacklist + confirm-gate
  training/sleep.py  обучение во сне: LoRA rank-4 (dA/dB) + гейты качества,
                     экспорт .npz -> W=B@A -> ONNX (для Stage-2 C++)
scripts/
  doctor.py          self-check окружения (NPU/weights/KDE — ожидаемые WARN)
  export_adapters.py дамп выученных адаптеров в ONNX
  install_shell_hook.py интеграция с zsh/bash preexec (передача строк агенту)
tests/test_agent.py  регрессии: опечатки, safety, награды, RAG, сон, false-positive пути
```

Поток одной строки: `user_command` -> Lexicon.observe() **после** анализа
(запрет самолегитимации: история пользователя никогда не делает неизвестную
команду «известной») -> ProposalEngine собирает evidence-packet (PATH/seed,
ФС-соседи, layout decode, user_words) -> ModelAnalyzer выносит вердикт +
confidence; ниже ABSTAIN_BELOW=0.45 — молчание. Молчание также по явному
вето «так задумано» (prefs-память, `intentional-token:<токен>`).

## Правила принятия кандидатов (анти false-positive)

* Команда известна, если она seed/builtin или реальный бинарь в PATH
  (`shutil.which`). Личная история НЕ легитимизирует токен.
* Путь флагуетcя только по верифицированной ФС: кандидат принимается, если
  `weighted_distance <= max(2, len*0.55)` **И** `char_jaccard >= 0.5`,
  **И** цель реально существует на диске прямо сейчас.
* learned-dict кандидаты для путей требуют существования близкого имени под
  проверенным префиксом или в HOME (живой ленивый скан home, раз в 5 с).
  Новый `~/projects` != опечатка `~/Documents`.
* layout-decode сверяется с реальными именами (home/соседи), а не со словарём.
* Новые файлы в существующих каталогах (`touch foo.txt`) — не опечатки.

## Система наград (инициатива учится на реакциях)

| реакция пользователя | reward | эффект |
|---|---|---|
| accept/thanks | +1.0 | порог типа вниз (чаще помогать там, где полезно) |
| refuse | −0.2 | слабое торможение |
| intentional («так задумано») | −0.1 | паттерн в prefs-вето, больше не предлагать |
| annoyed/dismiss | −0.5 | порог вверх (меньше навязчивости) |
| missed (мы проглядели очевидное) | −0.4 | порог вниз |

Плюс: почасовой бюджет подсказок, cooldown на тип, EMA-полезность по типам.

## Безопасность

* `tools/safety.py`: levels ok/confirm/block; blacklist (`rm -rf /`, mkfs, dd
  on dev, fork-bombs...), whitelist никогда не блокируется; подтверждение —
  через событие, без эскалации привилегий (sudo не автоматизируется).
* Аудит-лог всех действий в `$XDG_STATE_HOME/kde-agent/audit.log`.
* `/undo` откатывает последнее действие реестра инструментов.
* Все данные — только в XDG-каталогах пользователя; никаких скриншотов
  «на всякий случай»: события и данные берутся по запросу.

## Обучение во сне (модель заморожена!)

Только LoRA-адаптеры (rank 4: dB:(d,r), dA:(r,d)) + гейты перед сохранением:
регрессия навыков / назойливость / safety-наборы. Адаптер, не прошедший гейты,
не сохраняется. Экспорт: `.npz` -> `W = B@A` -> ONNX — Stage-2 C++ грузит
 exactly эти веса через ONNX Runtime (XDNA EP).

## Stage-2 (план, не реализовано)

C++ Qt6: инференс ONNX Runtime/XDNA, D-Bus (org.kde.Notifications, KWin),
плазмоид, KRunner-плагин, autostart .desktop, user unit-файл. Обучение
остаётся на Python. Расширение ТЗ (карточки знаний, сегментация AST/tree-sitter,
самопроверка, приоритетная очередь изучения + энергогейты батарея/idle/нагрузка,
ассистент кода с проверкой линтерами/pytest/mypy/ruff/shellcheck, профили
проектов, privacy-маскирование секретов, quality-control базы, sandbox примеров,
CLI управления базой) — следующие шаги поверх текущей Stage-1.

## Тесты и регрессии

`tests/test_agent.py` (20 тестов) покрывает: транспозиции команд, запрет
самолегитимации, veto-intentional, адаптивные пороги, бюджет подсказок,
safety-классификатор, RAG roundtrip/collections/preference(value=),
формы градиентов и гейты sleep-тренажёра, идемпотентность логгера, и
**TestPathFalsePositives** — три регрессии на связку Documents/projects
в fake-home (KDE_AGENT_HOME env override для тестов).
