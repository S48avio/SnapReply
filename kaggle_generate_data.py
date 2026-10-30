# Smart Reply synthetic data generator: one Kaggle cell.
#
# Kaggle setup: Settings -> Accelerator = "GPU T4 x2", Internet = On.
# Paste this whole file into ONE cell and run it. If an earlier run already
# changed packages, do "Run -> Factory reset" first so you start clean.
# 1) First run with N_EMAILS = 50 to check quality and speed.
# 2) Then set N_EMAILS = 5000 and use "Save Version -> Save & Run All" so it
#    keeps running after you close the browser.
# Output: /kaggle/working/smart_reply_data.jsonl (Output panel -> download).
# Re-running the cell resumes: ids already in the file are skipped.

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
# vLLM replaces Kaggle's torch with a new CUDA build but leaves Kaggle's torchaudio on the
# old one. transformers imports torchaudio if it exists and crashes on the mismatch.
# We don't need audio, so remove it.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)

import json, os, random, re, time
from collections import Counter
import torch

# vLLM prints hundreds of harmless warnings (bf16->fp16 casts on T4); show errors only.
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

print(f"torch {torch.__version__} | CUDA build {torch.version.cuda}")
try:
    torch.ones(1).cuda()
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
except Exception as e:
    raise SystemExit(
        f"GPU check failed ({e!r}).\nLikely the Kaggle NVIDIA driver is too old for torch's "
        f"CUDA {torch.version.cuda} build. Paste this message to Claude for the next fix.")

N_EMAILS = 20000
CHUNK = 250          # emails per batch; the file is saved after every chunk
SEED = 42
OUT = "/kaggle/working/smart_reply_data.jsonl"
N_GPUS = torch.cuda.device_count()
# 7B in fp16 is ~15 GB, so it needs two 16 GB GPUs; fall back to 3B on one GPU.
MODEL = "Qwen/Qwen2.5-7B-Instruct" if N_GPUS >= 2 else "Qwen/Qwen2.5-3B-Instruct"

INTENTS = {
    "accept": "agree, say yes, confirm",
    "decline": "say no, politely refuse, can't make it",
    "thanks": "thank the sender",
    "ask_question": "ask a short clarifying question (when, where, which, who)",
    "will_do": "commit to doing the requested task",
    "reschedule": "propose another time or date",
    "acknowledge": "confirm you received or read it, no commitment",
    "need_more_info": "ask for more details or documents before deciding",
    "follow_up": "say you will check and get back to them later",
    "apologize": "say sorry for a delay, mistake or inconvenience",
}
INTENT_LINES = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())

# Random combinations of these give variety, so the model doesn't repeat one template.
# 83 topics x 22 senders x 8 tones x 5 lengths = ~73k combinations, more than N_EMAILS.
TOPICS = [
    # work
    "scheduling a meeting", "moving a meeting to another time", "project deadline",
    "project status update", "invoice or payment", "overdue payment reminder",
    "job interview", "job offer", "team lunch or dinner", "office party",
    "sharing a report for review", "requesting a document", "requesting a signature",
    "travel plans", "business trip booking", "a delayed delivery", "shipping address change",
    "server or IT outage", "password or account access", "new software rollout",
    "budget approval", "expense reimbursement", "vacation or leave request", "a sick day",
    "working from home", "client feedback", "client complaint", "contract signing",
    "contract renewal", "a price quote", "a purchase order", "a sales lead",
    "a partnership proposal", "a product launch", "a bug in the software", "a feature request",
    "a code review", "a presentation draft", "slides for a meeting", "a phone or video call",
    "conference registration", "a webinar invitation", "a training session",
    "performance review", "salary discussion", "promotion news", "hiring a new team member",
    "onboarding a new hire", "a colleague leaving the team", "a team offsite",
    "a survey to fill out", "meeting notes and action items", "an urgent approval",
    "a data request", "a security alert", "broken office equipment",
    "parking or building access", "a volunteering event", "a charity donation",
    # personal
    "weekend plans", "a birthday party invitation", "a wedding invitation",
    "a dinner reservation", "borrowing something", "returning a borrowed item",
    "rent or apartment issue", "a home repair", "a doctor appointment",
    "a dentist appointment", "school or kids event", "a parent-teacher meeting",
    "a sports match", "concert or movie tickets", "a holiday trip with friends",
    "a family visit", "a lost item", "a favor request", "a car repair", "a package pickup",
    "a group gift collection", "a book club meeting", "moving to a new house",
    "a pet sitting request",
]
SENDERS = [
    "your manager", "your manager's boss", "a coworker", "a new coworker", "a client",
    "a potential customer", "a close friend", "an old friend", "a family member",
    "your parent", "a vendor", "a supplier", "the HR department", "the IT helpdesk",
    "a recruiter", "someone on your team", "your landlord", "a neighbor",
    "your child's teacher", "a professor", "a service provider", "an event organizer",
]
TONES = ["formal", "casual", "urgent", "friendly", "apologetic", "excited",
         "brief and direct", "polite but firm"]
LENGTHS = ["one sentence", "2-3 sentences", "4-6 sentences",
           "one short paragraph ending with a question", "two short paragraphs"]


def email_spec(i):
    # Seeded by id, so a resumed run produces the same spec for the same id.
    rng = random.Random(SEED * 100003 + i)
    return {"topic": rng.choice(TOPICS), "sender": rng.choice(SENDERS),
            "tone": rng.choice(TONES), "length": rng.choice(LENGTHS)}


def email_messages(s):
    return [
        {"role": "system", "content": "You write realistic, varied emails for a dataset. Use fictional names and details."},
        {"role": "user", "content": (
            f"Write one email.\nSender: {s['sender']}\nTopic: {s['topic']}\n"
            f"Tone: {s['tone']}\nLength: {s['length']}\n"
            "The email must need a reply from the recipient (a question, request, invitation or update).\n"
            'Return only JSON: {"subject": "...", "body": "..."}')},
    ]


def reply_messages(e):
    # Pass 2: the email already exists, so it was written without knowing the intents.
    return [
        {"role": "system", "content": "You write short Gmail Smart Reply suggestions."},
        {"role": "user", "content": (
            f"Email subject: {e.get('subject', '')}\nEmail body:\n{e['body']}\n\n"
            f"Possible reply intents:\n{INTENT_LINES}\n\n"
            "Pick the 3 different intents that are the most natural replies to this email. "
            "If the email asks for something, include one positive and one negative option. "
            "Write each reply as a short suggestion of 2 to 15 words, in the voice of the recipient.\n"
            'Return only JSON: {"replies": [{"intent": "...", "text": "..."}, '
            '{"intent": "...", "text": "..."}, {"intent": "...", "text": "..."}]}')},
    ]


def parse_json(text):
    # Grab the outermost {...}; this also strips ```json fences the model may add.
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def clean_replies(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("replies"), list):
        return []
    replies, seen = [], set()
    for r in obj["replies"]:
        if not isinstance(r, dict):
            continue
        intent = str(r.get("intent", "")).strip().lower()
        text = str(r.get("text", "")).strip()
        if intent in INTENTS and intent not in seen and 1 <= len(text.split()) <= 25:
            seen.add(intent)
            replies.append({"intent": intent, "text": text})
    return replies[:3]


def load_generator():
    """Returns generate(list_of_chat_messages, temperature, max_tokens) -> list of texts."""
    try:
        from vllm import LLM, SamplingParams
        # T4 has no bfloat16, so force float16. vLLM batches all prompts on the GPU itself.
        llm = LLM(model=MODEL, dtype="half", tensor_parallel_size=max(N_GPUS, 1),
                  max_model_len=2048, gpu_memory_utilization=0.90, seed=SEED)

        def generate(messages_list, temperature, max_tokens):
            params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
            outs = llm.chat(messages_list, params, use_tqdm=False)
            return [o.outputs[0].text for o in outs]
        return generate, "vllm"
    except Exception as e:
        print(f"vLLM unavailable ({e!r}); falling back to transformers (slower).")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto")

    def generate(messages_list, temperature, max_tokens, batch_size=16):
        texts = []
        for b in range(0, len(messages_list), batch_size):
            prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages_list[b:b + batch_size]]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                     max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id)
            texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return texts
    return generate, "transformers"


# Resume: collect ids already saved. A crash can leave a half-written last line; skip it.
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    with open(OUT, "rb+") as f:  # make sure new records start on a fresh line
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
todo = [i for i in range(N_EMAILS) if i not in done]
print(f"GPUs: {N_GPUS} | model: {MODEL} | already done: {len(done)} | to do: {len(todo)}")

if todo:
    t0 = time.time()
    print("Loading model (first run downloads ~15 GB)...", flush=True)
    generate, backend = load_generator()
    print(f"Model ready on {backend} in {(time.time() - t0) / 60:.1f} min. "
          f"The first chunk is slower (GPU warm-up), so the ETA settles after 2-3 chunks.", flush=True)
    n_chunks = (len(todo) + CHUNK - 1) // CHUNK
    start, kept, pairs = time.time(), 0, 0
    for c in range(0, len(todo), CHUNK):
        ids = todo[c:c + CHUNK]
        specs = [email_spec(i) for i in ids]

        # Pass 1: emails.
        emails = [parse_json(t) for t in generate([email_messages(s) for s in specs], 0.9, 400)]
        emails = [e if isinstance(e, dict) and str(e.get("body", "")).strip() else None for e in emails]

        # Pass 2: replies, only for emails that parsed.
        has = [k for k, e in enumerate(emails) if e]
        raw = generate([reply_messages(emails[k]) for k in has], 0.8, 300) if has else []
        replies = {k: clean_replies(parse_json(t)) for k, t in zip(has, raw)}

        # Failed ids are written too (with "error"), so a resumed run doesn't retry them forever.
        with open(OUT, "a") as f:
            for k, (i, s) in enumerate(zip(ids, specs)):
                rec = {"id": i, **s, "model": MODEL, "backend": backend}
                r = replies.get(k, [])
                if emails[k] and len(r) >= 2:
                    rec.update(subject=str(emails[k].get("subject", "")).strip(),
                               body=str(emails[k]["body"]).strip(), replies=r)
                    kept += 1
                    pairs += len(r)
                else:
                    rec["error"] = "bad email json" if not emails[k] else "too few valid replies"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        mins = (time.time() - start) / 60
        n = c + len(ids)
        saved = len(done) + n
        rate = n / max(mins, 1e-9)
        print(f"[{time.strftime('%H:%M:%S')}] chunk {c // CHUNK + 1}/{n_chunks} | "
              f"saved {saved}/{N_EMAILS} ({100 * saved / N_EMAILS:.1f}%) | left {N_EMAILS - saved} | "
              f"good {kept}/{n} ({100 * kept / n:.0f}%) | pairs {pairs} | "
              f"{rate:.0f} emails/min | elapsed {mins:.0f} min | ETA {(N_EMAILS - saved) / rate:.0f} min",
              flush=True)

# Summary over the whole file (this run + earlier runs).
rows = [json.loads(line) for line in open(OUT) if line.strip()]
good = [r for r in rows if "replies" in r]
print(f"\nFile: {OUT}\nemails: {len(rows)} | good: {len(good)} | "
      f"(email, reply) pairs: {sum(len(r['replies']) for r in good)}")
print("errors:", dict(Counter(r["error"] for r in rows if "error" in r)))
print("replies per intent:", dict(Counter(x["intent"] for r in good for x in r["replies"]).most_common()))
if good:
    print("\nExample:\n" + json.dumps(good[0], indent=2, ensure_ascii=False))
