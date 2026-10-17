
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
=======
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