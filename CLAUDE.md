# Tilawah (FYP) — Claude Code instructions

## Project
AI Quran reading tutor for Noorani Qaida. Learner recites, Whisper (faster-whisper,
OdyAsh/faster-whisper-base-ar-quran) transcribes, rapidfuzz compares with expected
Arabic, verdict pass/fail. See README.md for full architecture.

## Stack and commands
- Backend: FastAPI, Python 3.12 only. Run from inside backend/:
  `.\.venv\Scripts\Activate.ps1` then `uvicorn app.main:app --reload --port 8000`
- Frontend: React 19 + TS + Vite, `npm run dev`, `npm run lint`, `npm run build`
- Tests: `cd backend; python -m pytest tests -q` (33 tests must stay green)
- Windows + PowerShell.

## FYP scope (do not expand)
- Qaida pages p01-p12 only.
- New contribution: tajweed detection = Madd duration (rule-based) + Makharij
  confusion (ML: س/ص, د/ض, ك/ق, ه/ح).
- Evaluation is a core deliverable: per-letter accuracy, FAR, FRR, confusion matrix,
  baselines, threshold ablation. Speaker-wise train/test split, never mix.

## Rules
- Do NOT touch v2/ unless I ask.
- Do NOT modify matching.py normalization without telling me why; add tests first.
- Do NOT commit backend/.env, content/eval_log/, .venv, node_modules.
- Never invent evaluation numbers. All metrics must come from a script run on real data.
- Never add copyrighted content outside content/; repo stays private.
- Plan first for any change touching more than 2 files. Run tests after every change.
- New ML/eval code goes in ml/ and eval/, not inside backend/app/.
- Explain code simply: I must defend it in front of advisors.

## Key files
- backend/app/services/matching.py (scoring), asr.py (transcription), config.py
- frontend/src/hooks/useReciteMachine.ts (fragile: runId/skipId cancellation, be careful)

## Testing rules
- New logic (scoring, Madd, metrics, split utility): write failing unit tests first, then code.
- Add integration tests only for: /api/evaluate flow, Madd in API, eval script, content validation.
- ML classifier: no TDD; validated by held-out evaluation instead.
- After finishing a feature, I will run a separate "adversarial tester" session. In that session, list problems only, no fixes.
- Keep testing effort proportional; do not write tests for UI layout or v2/.