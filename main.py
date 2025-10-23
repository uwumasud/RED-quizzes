# Telegram Quiz Bot — patched for Railway stability & robustness
# Changes:
# 1) Keeps Railway "web service" happy by binding to $PORT with a tiny HTTP server (or do nothing if not set).
# 2) Switches to HTML parse mode + escapes dynamic text to prevent Telegram Markdown parse crashes.
# 3) Reduces API churn: no more per-answer message edits (tick still updates once per second).
# 4) Memory safety: keep per-question answers only for the current round; maintain running totals & free old data.
# 5) Extra safeguards & logging.

from __future__ import annotations
import os, json, time, random, math, html, logging, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("quizbot")

# ---------- CONFIG ----------
QUESTION_TIME = 10            # seconds to answer
DELAY_NEXT    = 5             # gap between questions (live countdown)
POINTS_MAX    = 100           # max points per correct (faster = more)
QUESTIONS_FILE = "questions.json"

ALLOWED_SESSION_SIZES = (10, 20, 30, 40, 50)
MODES = ("beginner", "standard", "expert")

DM_CONFIRM = False            # set True to DM users "You chose ..." in groups
# ----------------------------

def esc(s: str) -> str:
    """Escape dynamic text for HTML parse mode."""
    return html.escape(str(s), quote=True)

@dataclass
class QItem:
    text: str
    options: List[str]
    correct: int
    mode: str

@dataclass
class AnswerRec:
    choice: int
    is_correct: bool
    elapsed: float
    points: int

@dataclass
class GameState:
    chat_id: int
    started_by: int
    questions: List[QItem]
    limit: int
    mode: str

    q_index: int = 0
    q_msg_id: Optional[int] = None
    q_start_ts: Optional[float] = None
    locked: bool = False

    # For the *current* question only
    per_q_answers: Dict[int, Dict[int, AnswerRec]] = field(default_factory=dict)   # qidx -> {user_id: AnswerRec}
    answered_now: Dict[int, Set[int]] = field(default_factory=dict)                 # qidx -> set(user_id) to prevent race double taps

    # Persistent aggregates (keep memory bounded)
    players: Dict[int, str] = field(default_factory=dict)                           # user_id -> display name
    totals: Dict[int, int] = field(default_factory=dict)                            # user_id -> total points
    corrects: Dict[int, int] = field(default_factory=dict)                          # user_id -> total correct

# Active game per chat
GAMES: Dict[int, GameState] = {}
# Last finished snapshot per chat (for /answer and /leaderboard)
LAST: Dict[int, dict] = {}
# Per-chat config (Mode/Length) stored outside chat_data so it never disappears
SETTINGS: Dict[int, Dict[str, int | str]] = {}

# ---------- Helpers ----------
def points(elapsed: float) -> int:
    if elapsed >= QUESTION_TIME:
        return 0
    return int(round((max(0.0, QUESTION_TIME - elapsed) / QUESTION_TIME) * POINTS_MAX))

def answer_kb(q: QItem, qnum: int) -> InlineKeyboardMarkup:
    # Buttons are just the option text (no A/B/C/D); callback data carries indices
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(opt, callback_data=f"ans:{qnum}:{i}")]
        for i, opt in enumerate(q.options)
    ])

def fmt_question(qnum: int, total: int, q: QItem,
                 left: Optional[int] = None, locked_count: Optional[int] = None) -> str:
    head = f"❓ <b>Question {qnum}/{total}</b>\n{esc(q.text)}"
    if left is not None:
        head += f"\n\n⏱ <b>{int(left)}s left</b>"
    if locked_count is not None:
        head += f"\n🗳 <b>Answers locked:</b> {locked_count}"
    return head

def load_questions() -> List[QItem]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for i, q in enumerate(data, start=1):
        if not all(k in q for k in ("text", "options", "correct", "mode")):
            raise ValueError(f"Q{i} missing fields")
        if q["mode"] not in MODES:
            raise ValueError(f"Q{i} invalid mode {q['mode']}")
        if len(q["options"]) != 4:
            raise ValueError(f"Q{i} must have 4 options")
        c = int(q["correct"])
        if not 0 <= c < 4:
            raise ValueError(f"Q{i} invalid correct index {c}")
        out.append(QItem(text=q["text"], options=list(q["options"]), correct=c, mode=q["mode"]))
    return out

def filter_by_mode(all_qs: List[QItem], mode: str) -> List[QItem]:
    return [q for q in all_qs if q.mode == mode]

def shuffle_qs(qs: List[QItem]) -> List[QItem]:
    qs = qs.copy()
    random.shuffle(qs)
    out: List[QItem] = []
    for q in qs:
        pairs = list(enumerate(q.options))
        random.shuffle(pairs)
        new_opts = [t for _, t in pairs]
        new_correct = next(i for i, (orig, _) in enumerate(pairs) if orig == q.correct)
        out.append(QItem(text=q.text, options=new_opts, correct=new_correct, mode=q.mode))
    return out

# Totals are now tracked incrementally; this is here for compatibility
# with calls that expect a function.
def compute_totals(st: GameState) -> Tuple[Dict[int, int], Dict[int, int]]:
    return st.totals, st.corrects

def cancel_jobs_for_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Cancel any tick/close/gap jobs for this chat to avoid stray edits."""
    try:
        for job in context.job_queue.jobs():
            name = getattr(job, "name", "") or ""
            if name.startswith(f"tick:{chat_id}:") or name.startswith(f"close:{chat_id}:") or name.startswith(f"gap:{chat_id}:"):
                job.schedule_removal()
    except Exception as e:
        log.warning("job cleanup error: %s", e)

# ---------- Jobs / Flow ----------
async def tick_edit(context: ContextTypes.DEFAULT_TYPE):
    """Edit the question each second with the live numeric countdown + answer count."""
    data = context.job.data
    chat_id = data["chat_id"]; msg_id = data["msg_id"]; end_ts = data["end_ts"]; qidx = data["qidx"]
    st = GAMES.get(chat_id)
    if not st or st.q_index != qidx:
        context.job.schedule_removal(); return
    left = max(0, int(math.ceil(end_ts - time.time())))
    if left <= 0:
        context.job.schedule_removal()
    try:
        count = len(st.per_q_answers.get(st.q_index, {}))
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=msg_id,
            text=fmt_question(st.q_index + 1, st.limit, st.questions[st.q_index], left, count),
            reply_markup=answer_kb(st.questions[st.q_index], st.q_index + 1),
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        # Rate limits or minor BadRequest shouldn't kill the job
        log.debug("tick_edit edit failed: %s", e)

async def gap_tick(context: ContextTypes.DEFAULT_TYPE):
    """Edits the 'Next question…' message every second with a live countdown."""
    data = context.job.data
    chat_id = data["chat_id"]; msg_id = data["msg_id"]; end_ts = data["end_ts"]
    st = GAMES.get(chat_id)
    if not st:
        context.job.schedule_removal(); return
    left = max(0, int(math.ceil(end_ts - time.time())))
    if left <= 0:
        context.job.schedule_removal()
        try:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text="🚀 Starting next question…")
        except Exception as e:
            log.debug("gap final edit failed: %s", e)
        return
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=msg_id,
            text=f"⏭️ Next question is coming in {left}s…"
        )
    except Exception as e:
        log.debug("gap_tick edit failed: %s", e)

async def post_round_recap(context: ContextTypes.DEFAULT_TYPE, st: GameState, qidx: int):
    """After time ends: show correct answer, round scorers, and current leaderboard (Top 5)."""
    q = st.questions[qidx]
    per = st.per_q_answers.get(qidx, {})
    # correct answers only, highest points first
    scorers = [(uid, rec.points) for uid, rec in per.items() if rec.is_correct]
    scorers.sort(key=lambda t: t[1], reverse=True)

    # current totals (already aggregated)
    scores, corrects = compute_totals(st)
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    lines = [f"📘 <b>Q{qidx+1} Result</b>",
             f"✅ Correct answer: <b>{esc(q.options[q.correct])}</b>"]
    if scorers:
        names = []
        for uid, pts in scorers[:5]:
            names.append(f"{esc(st.players.get(uid, str(uid)))} (+{pts})")
        others = max(0, len(scorers) - 5)
        lines.append("🏎️ Fastest correct: " + ", ".join(names) + (f" … +{others} more" if others else ""))
    else:
        lines.append("😶 No correct answers this round.")

    if ranking:
        lines.append("\n🏁 <b>Current Leaderboard</b> (Top 5)")
        for rank, (uid, total) in enumerate(ranking[:5], start=1):
            name = esc(st.players.get(uid, str(uid)))
            corr = corrects.get(uid, 0)
            medal = ("🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}.")
            lines.append(f"{medal} {name} — {corr} correct — {total} pts")

    await context.bot.send_message(chat_id=st.chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML)

async def close_question(context: ContextTypes.DEFAULT_TYPE):
    """Close the question, post recap, and start a 5s live countdown to the next question."""
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    st.locked = True

    # Freeze question to 0s (remove buttons) and show final count
    try:
        count = len(st.per_q_answers.get(st.q_index, {}))
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=st.q_msg_id,
            text=fmt_question(st.q_index + 1, st.limit, st.questions[st.q_index], 0, count),
            reply_markup=None, parse_mode=ParseMode.HTML
        )
    except Exception as e:
        log.debug("final freeze edit failed: %s", e)

    # Per-round recap
    await post_round_recap(context, st, st.q_index)

    # Free memory from the finished question
    try:
        st.per_q_answers.pop(st.q_index, None)
        st.answered_now.pop(st.q_index, None)
    except Exception:
        pass

    # Live "next question" countdown
    gap_end = time.time() + DELAY_NEXT
    gap_msg = await context.bot.send_message(chat_id=chat_id, text=f"⏭️ Next question is coming in {int(math.ceil(DELAY_NEXT))}s…")
    context.job_queue.run_repeating(
        gap_tick, interval=1.0, first=1.0,
        data={"chat_id": chat_id, "msg_id": gap_msg.message_id, "end_ts": gap_end},
        name=f"gap:{chat_id}:{st.q_index}"
    )
    context.job_queue.run_once(next_question, when=DELAY_NEXT, data={"chat_id": chat_id})

async def next_question(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data["chat_id"]
    st = GAMES.get(chat_id)
    if not st: return
    st.q_index += 1
    if st.q_index >= st.limit:
        await finish_quiz(context, st); return
    await ask_question(context, st)

async def ask_question(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    q = st.questions[st.q_index]
    msg = await context.bot.send_message(
        chat_id=st.chat_id,
        text=fmt_question(st.q_index + 1, st.limit, q, QUESTION_TIME, 0),
        reply_markup=answer_kb(q, st.q_index + 1),
        parse_mode=ParseMode.HTML
    )
    st.q_msg_id = msg.message_id
    st.q_start_ts = time.time()
    st.locked = False
    st.answered_now[st.q_index] = set()   # reset per-question answered set
    end_ts = st.q_start_ts + QUESTION_TIME
    context.job_queue.run_repeating(
        tick_edit, interval=1.0, first=1.0,
        data={"chat_id": st.chat_id, "msg_id": st.q_msg_id, "end_ts": end_ts, "qidx": st.q_index},
        name=f"tick:{st.chat_id}:{st.q_index}"
    )
    context.job_queue.run_once(
        close_question, when=QUESTION_TIME, data={"chat_id": st.chat_id},
        name=f"close:{st.chat_id}:{st.q_index}"
    )

# ---------- Admin & scoring ----------
async def is_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    try:
        m = await context.bot.get_chat_member(chat_id, user_id)
        return m.status in ("administrator", "creator")
    except Exception:
        return False

# ---------- Commands / Callbacks ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = SETTINGS.get(update.effective_chat.id, {})
    mode = cfg.get("mode"); length = cfg.get("length")
    await update.message.reply_text(
        "✨ <b>Quiz Bot</b>\n────────────\n"
        "Step 1: /menu — choose <b>Mode</b> (Beginner/Standard/Expert)\n"
        "Step 2: bot prompts you to choose <b>How many questions</b> (10–50)\n"
        "Then: /startquiz — start (admin-only in groups)\n\n"
        "During play: After each round, you’ll see the correct answer + current top-5.\n"
        "Tools: /leaderboard • /answer • /stopquiz • /reset\n\n"
        f"Current: Mode={esc(mode) if mode else '—'} • Length={esc(length) if length else '—'}",
        parse_mode=ParseMode.HTML
    )

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = [[InlineKeyboardButton("Beginner", callback_data="cfg:mode:beginner"),
             InlineKeyboardButton("Standard", callback_data="cfg:mode:standard"),
             InlineKeyboardButton("Expert",   callback_data="cfg:mode:expert")]]
    await update.message.reply_text(
        "<b>Step 1/2</b>: Choose a <b>Mode</b>.",
        reply_markup=InlineKeyboardMarkup(rows), parse_mode=ParseMode.HTML
    )

async def _send_length_menu(update_or_query, context):
    rows = [
        [InlineKeyboardButton("10", callback_data="cfg:len:10"),
         InlineKeyboardButton("20", callback_data="cfg:len:20"),
         InlineKeyboardButton("30", callback_data="cfg:len:30")],
        [InlineKeyboardButton("40", callback_data="cfg:len:40"),
         InlineKeyboardButton("50", callback_data="cfg:len:50")],
    ]
    chat_id = update_or_query.effective_chat.id if isinstance(update_or_query, Update) else update_or_query.message.chat.id
    await context.bot.send_message(
        chat_id=chat_id,
        text="<b>Step 2/2</b>: How many questions will be held?",
        reply_markup=InlineKeyboardMarkup(rows), parse_mode=ParseMode.HTML
    )

async def cfg_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    if not data.startswith("cfg:"): return
    _, kind, val = data.split(":")
    chat_id = q.message.chat.id

    # Changing settings wipes old session/logs
    GAMES.pop(chat_id, None)
    LAST.pop(chat_id, None)

    if kind == "mode":
        if val not in MODES:
            await q.edit_message_text("Invalid mode."); return
        SETTINGS.setdefault(chat_id, {})["mode"] = val
        try:
            await q.edit_message_text(f"Mode set to <b>{esc(val.title())}</b> ✅", parse_mode=ParseMode.HTML)
        except Exception:
            pass
        await _send_length_menu(update, context)
        return

    if kind == "len":
        try:
            length = int(val)
            if length not in ALLOWED_SESSION_SIZES: raise ValueError()
        except Exception:
            await q.edit_message_text("Invalid length."); return
        SETTINGS.setdefault(chat_id, {})["length"] = length
        await q.edit_message_text(f"Length set to <b>{length}</b> ✅\nUse /startquiz to begin.", parse_mode=ParseMode.HTML)
        return

async def cmd_startquiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user

    # admin-only in groups
    if update.effective_chat.type != "private":
        if not await is_admin(context, chat_id, user.id):
            await update.message.reply_text("Only group admins can start a quiz.")
            return

    cfg = SETTINGS.get(chat_id, {})
    mode = cfg.get("mode"); length = cfg.get("length")
    if mode not in MODES or length not in ALLOWED_SESSION_SIZES:
        await update.message.reply_text("Please run /menu and complete <b>both steps</b> (Mode and Length) first.", parse_mode=ParseMode.HTML)
        return

    # start new session—clear previous logs/snapshots
    GAMES.pop(chat_id, None)
    LAST.pop(chat_id, None)

    all_qs = load_questions()
    pool = filter_by_mode(all_qs, str(mode))
    if len(pool) < int(length):
        await update.message.reply_text(f"Not enough questions in <b>{esc(mode)}</b> (need {length}).", parse_mode=ParseMode.HTML)
        return

    qs = shuffle_qs(pool)[: int(length)]
    st = GameState(chat_id=chat_id, started_by=user.id, questions=qs, limit=int(length), mode=str(mode))
    GAMES[chat_id] = st

    await update.message.reply_text(
        f"🎯 <b>{esc(str(mode).title())}</b> mode • {length} questions\n"
        f"⏱ {QUESTION_TIME}s per question • Next question in {int(math.ceil(DELAY_NEXT))}s after each.",
        parse_mode=ParseMode.HTML
    )
    await ask_question(context, st)

async def on_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    user = q.from_user
    chat_id = q.message.chat.id
    try:
        _, qnum_str, opt_str = q.data.split(":")
        qnum = int(qnum_str); opt = int(opt_str)
    except Exception:
        return

    st = GAMES.get(chat_id)
    if not st: 
        try: await q.answer("No active quiz here.", show_alert=False)
        except Exception: pass
        return
    if st.q_index + 1 != qnum:
        try: await q.answer("This question is already closed.", show_alert=False)
        except Exception: pass
        return
    if st.locked or st.q_start_ts is None:
        try: await q.answer("Time is up!", show_alert=False)
        except Exception: pass
        return

    # ---- One answer per user (race-proof) ----
    st.answered_now.setdefault(st.q_index, set())
    if user.id in st.answered_now[st.q_index]:
        try: await q.answer("Only your first answer counts.", show_alert=False)
        except Exception: pass
        return
    # Mark immediately to prevent two rapid taps racing
    st.answered_now[st.q_index].add(user.id)

    # Player name cache (escaped on render)
    st.players[user.id] = (user.full_name or user.username or str(user.id))[:64]

    # Prevent duplicates in this round
    st.per_q_answers.setdefault(st.q_index, {})
    if user.id in st.per_q_answers[st.q_index]:
        try: await q.answer("Only your first answer counts.", show_alert=False)
        except Exception: pass
        return

    elapsed = max(0.0, time.time() - st.q_start_ts)
    is_correct = (opt == st.questions[st.q_index].correct)
    pts = points(elapsed) if is_correct else 0
    st.per_q_answers[st.q_index][user.id] = AnswerRec(choice=opt, is_correct=is_correct, elapsed=elapsed, points=pts)

    # Aggregate totals immediately (bounded memory)
    if is_correct:
        st.corrects[user.id] = st.corrects.get(user.id, 0) + 1
    st.totals[user.id] = st.totals.get(user.id, 0) + pts

    chosen_txt = st.questions[st.q_index].options[opt]
    # Big visible confirmation (modal)
    try:
        await q.answer(
            f"You chose:\n\n{chosen_txt}\n\n"
            f"{'✅ Correct' if is_correct else '❌ Locked in'} • {pts} pts",
            show_alert=True
        )
    except Exception:
        pass

    # NOTE: We no longer edit the main message on every answer to avoid rate-limit churn.
    # The per-second tick will reflect the updated locked count.

async def cmd_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = GAMES.get(chat_id)

    if st:
        scores, corrects = compute_totals(st)
        source = f"Current session — {st.q_index+1}/{st.limit} asked"
        name_of = st.players
    else:
        snap = LAST.get(chat_id)
        if not snap:
            await update.message.reply_text("No session found here yet."); return
        scores = snap["scores"]; corrects = snap["corrects"]
        source = f"Last finished — {snap['limit']} questions"
        name_of = snap["players"]

    if not scores:
        await update.message.reply_text("No participants yet."); return

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    lines = [f"🏁 <b>Leaderboard</b> ({esc(source)})"]
    for rank, (uid, pts_) in enumerate(ranking[:10], start=1):
        name = esc(name_of.get(uid, str(uid)))
        corr = corrects.get(uid, 0)
        medal = ("🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}.")
        lines.append(f"{medal} {name} — {corr} correct — {pts_} pts")
    lines.append("\nGG! Thanks for participating 🎉")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

async def cmd_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    snap = LAST.get(chat_id)
    if not snap:
        await update.message.reply_text("No finished quiz found here yet."); return
    qs: List[QItem] = snap["questions"]
    lines = ["📘 <b>All Correct Answers</b>"]
    for i, q in enumerate(qs, start=1):
        lines.append(f"Q{i}: <b>{esc(q.options[q.correct])}</b>")
        if len("\n".join(lines)) > 3500:  # Telegram limit guard
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            lines = []
    if lines:
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

async def finish_quiz(context: ContextTypes.DEFAULT_TYPE, st: GameState):
    cancel_jobs_for_chat(context, st.chat_id)  # stop any pending edits/countdowns

    scores, corrects = compute_totals(st)
    if not scores:
        await context.bot.send_message(chat_id=st.chat_id, text="Quiz ended. No participants 😅")
        GAMES.pop(st.chat_id, None); return

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Winner shoutout
    top_uid, top_pts = ranking[0]
    top_name = esc(st.players.get(top_uid, str(top_uid)))
    await context.bot.send_message(
        chat_id=st.chat_id,
        text=f"🎉 Congrats, <b>{top_name}</b>! You topped the quiz with <b>{top_pts} pts</b>!",
        parse_mode=ParseMode.HTML
    )

    # Final results
    lines = ["🏁 <b>Final Results</b> — Top 10"]
    for rank, (uid, pts_) in enumerate(ranking[:10], start=1):
        name = esc(st.players.get(uid, str(uid)))
        corr = corrects.get(uid, 0)
        medal = ("🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}.")
        lines.append(f"{medal} {name} — {corr}/{st.limit} correct — {pts_} pts")
    lines.append("\nUse /leaderboard anytime. Use /answer to reveal all correct answers.\nGG! Thanks for participating 🎉")
    await context.bot.send_message(chat_id=st.chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML)

    # Snapshot for /answer and /leaderboard
    LAST[st.chat_id] = {
        "questions": st.questions,
        "limit": st.limit,
        "scores": scores,
        "corrects": corrects,
        "players": st.players,
        "mode": st.mode,
    }

    # Clear active session
    GAMES.pop(st.chat_id, None)

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    here_id = update.effective_chat.id

    st = GAMES.get(here_id)

    # If no game in this chat, try to find one this user can legitimately stop
    if not st:
        for g in list(GAMES.values()):
            # Allow the starter to stop from anywhere
            if g.started_by == user.id:
                st = g
                break
            # Or allow group admins of that chat to stop it
            try:
                member = await context.bot.get_chat_member(g.chat_id, user.id)
                if member.status in ("administrator", "creator"):
                    st = g
                    break
            except Exception:
                continue

        if not st:
            await update.message.reply_text("No active quiz to stop.")
            return

    # Finish and clear
    st.limit = st.q_index + 1  # truncate to “now”
    await update.message.reply_text("Stopping quiz…")
    cancel_jobs_for_chat(context, st.chat_id)
    await finish_quiz(context, st)

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = GAMES.pop(chat_id, None)
    LAST.pop(chat_id, None)
    SETTINGS.pop(chat_id, None)
    cancel_jobs_for_chat(context, chat_id)
    await update.message.reply_text("✅ Reset complete. Use /menu to choose Mode, then Length, then /startquiz.")

# ---------- Railway keep-alive (only if service type is WEB) ----------
def start_keepalive_server():
    """If Railway assigns a $PORT (web service), bind a tiny HTTP server so the platform sees us as healthy.
    If running as a Worker (recommended), PORT is absent and this does nothing.
    """
    port_env = os.getenv("PORT")
    if not port_env:
        log.info("No $PORT detected — assuming Worker service (recommended).")
        return
    try:
        port = int(port_env)
    except ValueError:
        log.warning("Invalid PORT=%r; skipping keep-alive server", port_env)
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
        def log_message(self, *args, **kwargs):
            return

    def run_server():
        try:
            srv = HTTPServer(("0.0.0.0", port), Handler)
            log.info("Keep-alive HTTP server listening on 0.0.0.0:%d", port)
            srv.serve_forever()
        except Exception as e:
            log.error("Keep-alive server failed: %s", e)

    threading.Thread(target=run_server, daemon=True).start()

# ---------- App build ----------
def build_app() -> Application:
    token = (
        os.getenv("BOT_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM_TOKEN")
    )
    if not token:
        raise RuntimeError("Set BOT_TOKEN (or TELEGRAM_BOT_TOKEN / TELEGRAM_TOKEN) env var.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("menu",        cmd_menu))
    app.add_handler(CallbackQueryHandler(cfg_click, pattern=r"^cfg:"))
    app.add_handler(CommandHandler("startquiz",   cmd_startquiz))
    app.add_handler(CommandHandler("leaderboard", cmd_leaderboard))
    app.add_handler(CommandHandler("answer",      cmd_answer))
    app.add_handler(CommandHandler("stopquiz",    cmd_stop))
    app.add_handler(CommandHandler("reset",       cmd_reset))
    app.add_handler(CallbackQueryHandler(on_answer, pattern=r"^ans:\d+:\d$"))
    return app

if __name__ == "__main__":
    log.info("Starting quiz bot…")
    start_keepalive_server()  # harmless on Worker; required if you made the service type Web
    build_app().run_polling(close_loop=False)
