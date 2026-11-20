# Phase 2 — Personalised Counselling Product (religion & culture aware)

> Status: PLAN (no implementation yet). Builds on v1 serving stack; v1 eval work is
> frozen and ignored per project decision (2026-11-20). Product mode, not research mode.
>
> Scope statement from the user: local login DB, one-time onboarding (religion/culture
> focused), KB expansion with more religion-specific material, religion-driven RAG,
> religion-aware OARS prompt, and a good chat experience on Qwen3-1.7B DPO.

## 0. What v1 gives us (and what changes)

Already done and reused as-is:
- Qwen3-1.7B + DPO LoRA served by vLLM (`make model-server`), reasoning parser strips thinking.
- FastAPI layer (`backend/`), OARS system prompt in `backend/core/chat.py`, RAG injection
  via trailing `<retrieved_techniques>` user message.
- RAG: Chroma embedded + bge-small-en-v1.5, 2526 chunks / 30 sources (South Asian KB).
- Frontend: vendored Local-LLM-Chat v3 single HTML, localStorage-persisted history.

**Memory today (confirmed):** the server is stateless; the frontend holds the full
conversation in `localStorage` (`chatHistory`) and re-sends it as the OpenAI `messages`
array every turn. `build_messages()` replays all turns; `_fit_budget()` trims oldest
turns only when the prompt exceeds budget. So: previous messages → always seen; long
histories → oldest turns dropped; cleared storage / another browser → conversation gone.
Phase 2 replaces client-held memory with server-side persistence (§6).

**Dropped from v1:** all eval work (F009/F017/F019 harnesses stay in the repo, unused).

## 1. Research findings that shape this design

Distilled from the religion/spirituality-in-therapy literature; each item has a design
consequence below.

1. **Religious coping splits into positive and negative (Pargament's RCOPE).**
   Positive: benevolent reappraisal, collaborative coping ("God and I handle this
   together"), seeking spiritual support, forgiveness. Negative ("spiritual struggle"):
   punishment reappraisals ("God is punishing me"), abandonment ("God has left me"),
   guilt over sin/devotion, religiously-framed self-blame. Negative coping is a robust
   predictor of worse outcomes.
   → **Consequence:** onboarding must capture not just *which* faith but *how faith
   lands for this person right now* (comfort vs strain), and the prompt must be taught
   to spot and gently counter spiritual struggle instead of amplifying guilt.

2. **Faith-adapted CBT works about as well as standard CBT — its real win is
   acceptability and engagement for religious clients** (meta-analyses 2014/2015/2021;
   RCTs: Razali, Koenig/RCBT, Zhang, recent 2026 R-CBT RCT). Core rules from the
   technique reviews: use the *client's own* beliefs and texts, never the therapist's;
   encourage positive religious coping, discourage negative; never proselytize.
   → **Consequence:** we are building rapport and cultural fit, not claiming superior
   efficacy. The model offers faith material *as options* in the user's own idiom.

3. **Indian CBT adaptation literature** (>96% of surveyed Indian clinicians say
   adaptation is required): the named adaptation techniques are *culture/religion
   specific metaphors, proverbs, stories and analogies*, local idioms of distress,
   family involvement, caution with confrontational Socratic questioning.
   → **Consequence:** the KB expansion is not just "more therapy text" — it needs an
   **analogies/proverbs bank per tradition** (Gita verses with counselling gloss,
   shabads, sabr/tawakkul framing, parables, dohas).

4. **FICA spiritual history (Puchalski)** — the clinical standard for taking a
   spiritual history in one conversation: **F**aith/belief, **I**mportance/influence,
   **C**ommunity, **A**ddress-in-care ("how would you like me to handle this?").
   Guidance: use as a conversation guide, not a checklist; ask the "what gives your
   life meaning" question even when the person is not religious.
   → **Consequence:** FICA is the skeleton of our onboarding flow (§5), including the
   explicit consent question ("shall I bring in your faith when it fits?").

5. **South Asian community findings:** faith framing (Gurbani/Sikhi on a Sikh
   helpline, for example) measurably increases belonging and willingness to talk;
   explanatory models like nazar/evil-eye, jinn, past-life karma are common and must
   be explored, not pathologised; stigma ("log kya kahenge", derogatory vernacular
   terms for mental illness) suppresses help-seeking; family is the default support
   system and decision-maker, not an obstacle.
   → **Consequence:** explanatory-model exploration already in the v1 prompt stays;
   family/caste/community become first-class profile facts; the "none/spiritual" user
   path must be explicit so we never push religion at someone.

6. **Gita-as-therapy angle** (Indian J Psychiatry etc.): Arjuna presents textbook
   anxiety (trembling, dry mouth, confusion, avoidance) and Krishna runs what reads
   like a CBT session — catastrophe → reappraisal → value-based action (karma yoga,
   Gita 2.47: act without attachment to fruits).
   → **Consequence:** this is our flagship Hindu analogy set; same pattern exists per
   tradition (chardi kala, sabr with the 94:5-6 "with hardship comes ease", the
   Buddha's second-arrow sutta for pain-vs-suffering).

## 2. Non-goals (explicitly skipped)

- Eval harness / judges — frozen. No new eval features.
- Email verification, password reset emails, OAuth — local product; password reset is
  a documented sqlite command until someone needs more.
- Docker / cloud DB / Redis / any new infrastructure. SQLite + Chroma, as-is.
- JWT / external auth libs — stdlib random tokens in a table.
- Regional-language content (Hindi/Bengali/Tamil corpora) — bge-small is English-only;
  Hinglish code-mixing keeps working through the model, but the KB stays English.
  Upgrade path: multilingual embedder swap (re-ingest only). Not now.
- Voice interface, multi-device sync, model retraining. The 1.7B-DPO adapter is frozen.

## 3. Accounts & storage (SQLite, stdlib only)

New module `backend/core/db.py` (thin, explicit SQL — no ORM, matching "explicit over
clever"). DB file `data/therapist.db`, gitignored, created on first run.

```sql
CREATE TABLE users   (id INTEGER PK, username TEXT UNIQUE NOT NULL,
                      pw_salt BLOB, pw_hash BLOB, created_at TEXT);
CREATE TABLE tokens  (token TEXT PK, user_id INT, created_at TEXT);
CREATE TABLE profiles(user_id INT PK UNIQUE, call_name TEXT, age_bracket TEXT,
                      gender TEXT, language_pref TEXT,
                      religion TEXT, family_religion TEXT,
                      faith_importance TEXT,   -- anchor | quite | little | none
                      faith_stance TEXT,       -- comfort | strain | both | unset
                      faith_consent TEXT,      -- yes | only_if_raised | no
                      supports TEXT, concern TEXT, updated_at TEXT);
CREATE TABLE messages(id INTEGER PK, user_id INT, session_id TEXT,
                      role TEXT, content TEXT, created_at TEXT);
```

- Passwords: `hashlib.scrypt` (stdlib) with per-user random salt. No new dependency.
- Sessions tokens: `secrets.token_hex(32)`, stored, sent as `Authorization: Bearer`;
  revocable (logout deletes the row). No expiry for a local product.
- `session_id` is a client-chosen uuid per conversation; a new chat = new uuid. No
  separate sessions table — `messages` grouped by `(user_id, session_id)` is enough.
- Sensitive-data note: religion is special-category data everywhere except a local
  sqlite file. Everything stays on-disk locally; no telemetry, no logs of profile
  values beyond debug level.

API surface (all thin, logic in core/):

| Route | Purpose |
|---|---|
| `POST /api/auth/register` `{username,password,age_18_plus}` | create user + token; 18+ attestation required |
| `POST /api/auth/login` `{username,password}` | token |
| `POST /api/auth/logout` (auth) | delete token |
| `GET  /api/me` (auth) | username + profile |
| `PUT  /api/profile` (auth) | write profile fields (onboarding is just the first writer of this) |
| `GET  /api/sessions` (auth) | conversation list (id, preview, updated_at) |
| `POST /api/chat` (auth) `{session_id?, message, stream?}` | §6 |

`/v1/chat/completions` stays untouched as the raw OpenAI-compatible path (curl/tests).

## 4. Prompt-budget fix (pre-req, 1 line + test)

Measured on the current code: with `max_tokens=1024`, budget = 4096−1024−256 = **2816
est-tokens**, but system prompt (~450) + five 512-token passages (~2560) already
**exceeds it** — meaning with RAG on, the current code is dropping nearly *all* history
every turn. The conservative 4096 constant was set for the 8192 window and predates
the 1024-token DPO completion cap.

Fix: raise the budget base `4096 → 6144` in `_fit_budget` (server window 8192 −
completion 1024 leaves 7168 hard; 6144 keeps the estimator's safety margin). This is
the single highest-leverage line in phase 2: without it, personalised history is moot.
Regression test asserts history survives alongside a full RAG block.

## 5. Onboarding (first-login profile form)

Delivery: the first time a new user logs in, they get **one form screen** with every
field below, fill it in once, save → a single `PUT /api/profile`. Stored in `profiles`.
The same form *is* the Settings → profile editor, so everything stays editable anytime.
No chat wizard, no scripted bubbles — counselling chat starts after the form is saved.

Fields (FICA-informed, one screen, ~1 minute):

| # | Field (form label) | Options | Maps to |
|---|---|---|---|
| 1 | "What should I call you?" | free text (nickname ok) | rapport |
| 2 | "How old are you, roughly?" | <18, 18–24, 25–34, 35–49, 50+ | intake; <18 → no service |
| 3 | "I want to get pronouns and register right — how do you identify?" | woman / man / non-binary / prefer not to say | intake |
| 4 | "Which language are you most comfortable opening up in?" | English / Hinglish / Hindi / other (free) | style |
| 5 | "Do you follow a religion or spiritual path? No wrong answer." | Hindu / Muslim / Sikh / Christian / Buddhist / Jain / spiritual-not-religious / none / other | FICA-F |
| 6 | "And the family you grew up in?" (same options) | same options | mismatch = counselling gold |
| 7 | "How much does faith matter to you *these days*?" | it's my anchor / quite a bit / a little / not at all | FICA-I; **the gating signal** |
| 8 | "When things get hard, does faith comfort you, weigh on you, or both?" | comforts / weighs down / both / prefer not to say | FICA-I + RCOPE screen |
| 9 | "Who's in your corner right now — family, friends, a faith community, anyone?" | multi-select + free text | FICA-C |
| 10 | "Would you like me to bring in ideas from your own faith when they fit, or keep it neutral?" | yes, when it fits / only when I raise it / keep it neutral | FICA-A (explicit consent) |
| 11 | "Anything you'd like to talk about, now or later? One line is plenty." | free text, optional | presenting concern → seeds first-turn retrieval |

Design notes:
- Q6 (family religion) is deliberately separate from Q5: the atheist kid in a devout
  household, the interfaith marriage, the convert — the *gap* is often exactly what
  they'll want to talk about, and the prompt must not assume the two match.
- Q7+Q10 together gate everything downstream: `none`-importance **or** `keep it
  neutral` consent → zero faith framing, secular retrieval only. Never override an
  explicit "no".
- Q8 is a one-question spiritual-struggle screen; `weighs down`/`both` flips the
  prompt into struggle-aware mode (§9).
- Minors (<18): hard no — the app does not serve under-18s (user decision
  2026-11-20). Registration requires an 18+ attestation (§3), and an honest <18
  answer on the form is declined warmly with signposts to Childline 1098 (24/7,
  for children) and Tele MANAS 14416; no chat proceeds. No birthdate/ID
  verification — attestation + honest-answer decline is the standard bar for a
  local product.
- Every field except the name is optional, every select carries a "prefer not to say";
  saving with just a name means v1 behaviour exactly (neutral prompt, general RAG).

## 6. Server-side memory & `/api/chat`

Server-authoritative history:

- `POST /api/chat` receives `{session_id?, message}` (+ token). Backend loads profile +
  history from SQLite, runs the existing `core.chat` pipeline with the profile, then
  persists the user turn and the reply. First message with no `session_id` creates one
  and returns it.
- Memory becomes durable: survives cleared localStorage, browser switches, restarts.
- Frontend stops sending history (thinner client, single source of truth, and the
  stored transcript can't desync from what the model saw).
- Streaming: same SSE passthrough pattern as `chat_stream`, then a post-stream
  persist of the assembled reply.
- Token budget for history grows via §4.

## 7. KB expansion (make it genuinely multi-tradition)

Current corpus is Hindu-heavy (Akhilananda, sattvavajaya, Gita-adjacent, ~160k words)
with good Buddhist coverage, thin Islamic (one brochure + 2 papers), and near-zero
Sikh / Christian / Jain / secular material. Target: roughly double the corpus
(~30 → ~60 files), every file tagged with a `religion` metadata value.

New/expanded sources (public-domain or open-access only; same text-extraction flow
into `data/rag-kb/text/`, same 512-token chunker):

| Tradition | Additions (examples, all open) |
|---|---|
| Hindu | Gita-as-therapy papers (Bhatia 2013 *Indian J Psychiatry*, PMC "Psychotherapy — insights from Bhagavad Gita", Dhillon 2023 *Pastoral Psychology* PMC); **analogies file**: selected Gita verses + counselling gloss (2.47 karma-yoga for outcome anxiety; 2.14 impermanence; 6.5 lift yourself by yourself; equanimity 2.48), Kabir dohas (public domain) |
| Sikh | Bedi 2019 religio-cultural healing paper (full text, open); British Sikh Report 2018 mental-health chapter; Mann & Kaur 2025 helpline study; Sikh healthy-faith toolkit; concepts file: naam simran, sangat, seva, chardi kala, hukam, the five vices — each with a counselling gloss |
| Islamic | Naeem/Rathod culturally-sensitive CBT literature; tazkiyah al-nafs / sabr / tawakkul concept pieces; **scrupulosity/waswas clinical guidance** (needed: repeated-prayer loops are a real risk pattern); Quran framing file: 94:5-6, dua/istighfar as coping, "more merciful than a mother" |
| Christian (Indian) | open IJP/NIMHANS writings on faith & psychiatry in India; Bible analogies file (public-domain WEB/KJV): Ps 55:22, Mt 11:28–30, lament-psalm structure for grief, prodigal son for shame, Gethsemane "take this cup" for acceptance |
| Jain | anekantavada (many-sided view → cognitive flexibility/reappraisal), aparigraha (non-attachment), ksama (forgiveness) — open Jain-studies/IJIP papers + Tattvartha-based glosses |
| Buddhist | (already strong) add Sallatha Sutta (the second arrow — pain vs suffering), metta practice pieces |
| Secular / none | meaning & values work without God-language (secular ACT-style acceptance, self-compassion summaries, logotherapy-adjacent meaning pieces) — the retrieval path for `none` users must not starve |
| Cross-cutting India | log-kya-kahenge stigma, joint-family dynamics, arranged-marriage/marital conflict, exam & competitive pressure, workplace stress, caregiver burden, urban migration loneliness, men's help-seeking stigma, nazar/karma explanatory models |

Licensing rules: prefer public-domain originals (scripture translations: Arnold/WEB;
Guru Granth Sahib and Quran translations are public domain) and CC/BY academic papers;
no full pirated books; keep the existing `_index.txt` inventory convention and record
provenance per file.

Coverage decision (user, 2026-11-20): the six dominant traditions above + secular are
the scope. Ambedkarite/Navayana Buddhist, Adivasi (Sarna, Donyi-Polo), Parsi
(Zoroastrian), Baha'i and Jewish users are consciously **skipped**. They reach the
product through the free-text "other / tell me" religion answer (their own words are
stored and rendered into the `<person>` block), general retrieval, and the
ask-don't-generalise prompt rules. Revisit only if a real user actually needs one.

**Mechanics:** `data/rag-kb/manifest.json` maps each source filename →
`{"religion": "hindu|islam|sikh|christian|buddhist|jain|secular|general"}`.
`ingest.py` reads the manifest and adds `religion` to chunk metadata (missing entry →
`"general"`). Existing 30 files get tags without renaming (Hindu/Buddhist ones are
identifiable from `_index.txt`). `make ingest` unchanged otherwise.

## 8. Religion-aware retrieval

`retrieve(query, n, religion=None)` — extend the existing `where` filter:

- **Hybrid split:** with `RAG_TOP_K=5` and a profiled user of `faith_importance ≥
  "a little"` and consent ≠ neutral: **3 chunks from `where={"religion": user}` + 2
  general** (unfiltered). Never 5/5 faith chunks — the general South-Asian therapy
  material (family systems, OARS-adjacent technique, crisis handling) must always ride
  along, and pure-filter retrieval degrades when the user's message is off-topic for
  their tradition.
- **Gating:** `none`/`not at all` importance **or** `keep it neutral` consent → 5
  general, `religion=None`.
- **Soft override:** if the user's *own message* names their tradition explicitly
  ("I've stopped doing namaaz", "God is punishing me", "my parents' puja…"), include
  their religion split even when the profile gates it off — people talk faith when
  they talk faith. Simplest robust version: keyword scan of the last user message per
  tradition (small curated list, tested), not a model call.
- **Strain flag bonus:** `faith_stance ∈ {strain, both}` boosts retrieval toward the
  compassion/mercy/scrupulosity content of their tradition over practice-prescription
  content (achieved by the split + manifest `topic` tags on the new concept files, not
  a new index).
- `source_filter` (existing) stays; the eval-only precompute scripts are untouched and
  irrelevant now.

## 9. OARS prompt v2 (religion-aware, person-aware)

Keep the v1 bones (role, OARS conversation style, cultural attunement, safety tiers,
example). Add two dynamic sections, rendered only from a non-empty profile:

```
<person>
What they told us at intake — use naturally, don't recite:
- Calls themselves {call_name}; {age_bracket}; {gender}.
- Most comfortable in {language_pref} — match it.
- Follows {religion}; grew up in a {family_religion} family.
- Faith matters {faith_importance} to them right now.
- Supports: {supports}. First shared concern: {concern}.
Treat these as starting points to confirm in conversation, not fixed facts.
</person>

<faith_style>   (rendered per Q7/Q8/Q10 answers)
anchor/quite + consent yes:
  Their faith is a live resource. Mirror their own idiom (dua, paath, simran,
  puja, karma, chanting, mindfulness) in plain words when it fits. When you offer a
  reframe or practice, you may draw from THEIR tradition — one idea at a time,
  framed as an option ("some people find it helps to…"), never a prescription,
  never more than one religious idea per reply. Reference a story or verse only
  when it genuinely matches their situation, in plain words, briefly.
strain/both:
  Be alert for spiritual struggle: feeling punished or abandoned by God, guilt
  about sin or devotion, "my karma is doing this to me", or repeated prayer/ritual
  used to quiet anxiety. Respond gently: every tradition they belong to also
  carries mercy, forgiveness and compassion — help them hear that side. Never
  amplify guilt, never suggest more devotion as a fix. If ritualizing looks
  compulsive, name it kindly and encourage a professional who understands
  scrupulosity.
none/neutral consent:
  No religious framing, no God-language, no scripture. Keep support fully secular
  and human. If THEY bring up faith, follow their lead without endorsing or
  challenging it.
always:
  Never proselytize, never push them toward or away from any faith. Their faith
  community (temple, mosque, gurdwara, church, sangat) may be named as ONE support
  option among friends, family and professionals. If their faith differs from
  their family's, stay neutral and explore the space between. Safety rules beat
  everything: if a crisis is framed religiously ("God wants this to end"), respond
  to the crisis first, with the helpline numbers, before any faith content.
</faith_style>
```

Prompt-budget note: `<person>`+`<faith_style>` add ~120–180 est-tokens; with §4's
budget fix the whole thing fits with room for history. The 3+2 retrieval split keeps
the passage block the same size as today.

Model-capability caution (1.7B): the block is short and imperative for a reason —
long persona prompts make small models parrot the profile back ("As a Hindu
25-year-old…"). The "use naturally, don't recite" line + a regression check on
profile-parroting in tests. If it still leaks, the fallback is trimming `<person>`
to three facts (name, faith+importance, language).

`docs/prompt.md` gets updated to mirror v2 at implementation time, per repo convention.

## 10. Frontend (vendored Local-LLM-Chat v3, same single file)

- **Auth screen** (login/register) → token in localStorage.
- **First-login profile form**: the §5 fields on one screen (selects + short text),
  saved once via `PUT /api/profile`; the Settings profile editor reuses the same form.
- **Chat rewired to `/api/chat`**: sends token + session_id + latest message only;
  history comes from the server. New-chat button mints a session_id. Session list
  from `GET /api/sessions`.
- **Settings**: profile summary rendered as chips (editable → `PUT /api/profile`),
  logout, and "delete my data" (drops rows; cheap SQL, high trust).
- Existing localStorage `chatHistory` becomes a one-time migration path into the
  first server session (nice-to-have; flag as optional).

## 11. Build order & proposed feature ledger

Per repo rules, each becomes a `docs/features.json` entry when work starts (planned
status), tests-first throughout:

| Order | Feature | Content | Tests |
|---|---|---|---|
| P0 | F020 Budget fix | `_fit_budget` 4096→6144 | history+RAG fits regression |
| P1a | F021 KB expansion + manifest | sources per §7, `manifest.json`, ingest `religion` tag, `make ingest` | manifest↔files consistency, ingest metadata |
| P1b | F022 Religion-aware retrieval | `retrieve(religion=)`, 3+2 split, gating, message-scan override | temp-collection filter/split/gating |
| P1c | F023 OARS prompt v2 | `<person>` + `<faith_style>` renderers, profile arg through `core.chat` | render per profile, no-parrot, budget |
| P2a | F024 SQLite auth + profile | `db.py`, auth routes, `PUT /api/profile` | register/login/token/validation |
| P2b | F025 Server-side sessions | `/api/chat`, messages persistence, stream persist | round-trip, isolation between users |
| P3 | F026 Frontend product shell | auth + onboarding + chat rewire + settings | route-level stubs (no GPU) |

P0–P1 ship value without any auth (profile can be injected via env/test for a
preview), P2–P3 make it a product. Nothing here touches model serving.