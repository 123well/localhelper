"""End-to-end scripted demo of the whole cognitive cycle.

Scenario:
 1. user asks to list files           -> model -> tool -> answer
 2. shell hook reports `gti status`   -> typo detected -> notification -> accept (+R)
 3. shell hook reports `cd /hme/x`    -> path typo    -> 'intentional' (-R, memorized)
 4. same /hme again                   -> vetoed by memory (no nagging!)
 5. rm -rf /home/user                 -> blocked by safety gate
 6. repetitive sequence               -> automation proposal
 7. RAG: question answered from indexed docs, not weights
 8. /sleep                            -> adapter trained, gates pass, saved
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agent.config import AgentConfig
from agent.main import Agent, index_docs


async def run_demo() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="kde-agent-demo-"))
    import os
    os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
    os.environ["XDG_DATA_HOME"] = str(tmp / "data")
    os.environ["XDG_STATE_HOME"] = str(tmp / "state")

    cfg = AgentConfig()
    ag = Agent(cfg)
    await ag.bridge.connect()
    ag.subscribe()

    print("=== 1. прямой запрос ===")
    ag.bus.emit("user_request", text="list files in /workspace")
    await asyncio.sleep(0.2); await drain(ag)

    print("\n=== 2. опечатка в команде: gti status ===")
    ag.bus.emit("user_command", cmd="gti status")
    await asyncio.sleep(0.2); await drain(ag)
    keys = list(ag.initiative.pending)
    assert keys, "ожидалась подсказка про gti"
    r = ag.initiative.feedback(keys[0], "да, спасибо!")
    print(f"награда за принятую подсказку: {r:+.2f}; utility={ag.initiative.habits.utility}")

    print("\n=== 3. путь /hme — пользователь говорит: так задумано ===")
    ag.bus.emit("user_command", cmd="cd /hme/projects")
    await asyncio.sleep(0.2); await drain(ag)
    keys = [k for k in ag.initiative.pending if k.startswith("typo_path")]
    if keys:
        r = ag.initiative.feedback(keys[0], "это намеренно, у меня такой bind-mount")
        print(f"реакция 'намеренно': {r:+.2f}, паттерн сохранён в prefs")

    print("\n=== 4. повтор /hme — агент НЕ вмешивается (не назойлив) ===")
    ag.initiative.habits.last_seen.clear()          # снимаем cooldown для теста
    before = len(ag.initiative.pending)
    ag.bus.emit("user_command", cmd="cd /hme/projects && ls")
    await asyncio.sleep(0.2); await drain(ag)
    new = [k for k in ag.initiative.pending if k.startswith("typo_path")]
    print("вмешательство повторно:", bool(new), "(ожидается False)")

    print("\n=== 5. опасная команда из запроса модели ===")
    out = await ag.tools.call("exec_command", {"command": "rm -rf /"})
    print("результат:", out[:80])

    print("\n=== 6. повторяющиеся команды -> предложение автоматизации ===")
    ag.initiative.reps.hist.clear()
    for c in ["make", "make test", "git add -A", "make", "make test", "git add -A",
              "make", "make test", "git add -A"]:
        ivs = ag.initiative.on_user_line(c)
    print("детектор повторов:", [i.message for i in ivs] or "нет (порог не достигнут)")

    print("\n=== 7. RAG: знания извне ===")
    docs = tmp / "docs"; docs.mkdir()
    (docs / "pacman.man.txt").write_text(
        "pacman -Qdt показывает сиротские пакеты (orphaned). "
        "Для удаления: pacman -Rns $(pacman -Qdtq). Синтаксис взят из man-страницы.")
    n = index_docs(ag.memory, docs)
    ans = await ag.tools.call("rag_search", {"query": "как найти сиротские пакеты pacman"})
    print(f"проиндексировано {n}; поиск вернул фрагментов: {len(ans.splitlines())}")

    print("\n=== 8. цикл сна: обучение адаптера + гейты ===")
    rep = ag.trainer.sleep_cycle()
    print("отчёт сна:", rep)
    print("базовые навыки после тренировки:", ag.trainer.eval_base_skills())

    print("\n=== состояние контекста ===")
    print(f"ctx={ag.ctx.total()} tok, summaries={len(ag.ctx.summaries)}, "
          f"thresholds={dict(ag.initiative.habits.threshold)}")
    print("DEMO OK")


async def drain(ag: Agent) -> None:
    """Process queued events deterministically for the demo."""
    while not ag.bus.queue.empty():
        ev = ag.bus.queue.get_nowait()
        for h in ag.bus.handlers.get(ev.kind, []):
            r = h(ev)
            if asyncio.iscoroutine(r):
                await r


if __name__ == "__main__":
    asyncio.run(run_demo())
