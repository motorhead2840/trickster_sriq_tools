"""Keyed control test of the gate in abhaya_gate(1).ipynb. One Colab cell, one GPU.

Seven arms on 40 two-option questions with textbook keys, greedy decoding, no injected noise:
plain; plain a second time (the noise floor); kernel only; true reference always; false
reference always; full gate with the true reference; full gate with the false reference.
Then the notebook's own three questions with the notebook's own settings.
No token is needed: the model is public.
"""
import json
import math
import os
import re
import time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("GATE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ON_GPU = torch.cuda.is_available()
device = torch.device("cuda" if ON_GPU else "cpu")
GPU = torch.cuda.get_device_name(0) if ON_GPU else "none (CPU run)"
if not ON_GPU:
    print("No CUDA device. In Colab: Runtime > Change runtime type > GPU. Continuing on CPU.")


def gate_torch(logits, threshold=0.86, scale=3.5, gain=0.50, snr_floor=5.8):
    """The v4 kernel of abhaya_gate(1).ipynb, cell 7, written as PyTorch."""
    x = logits.to(torch.float32)
    m = x.max(dim=-1, keepdim=True).values
    mean = x.mean(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    s = e.sum(dim=-1, keepdim=True)
    entropy = m + torch.log(s) - (e * x).sum(dim=-1, keepdim=True) / s
    xi = (1.0 - entropy / math.log(x.shape[-1])).clamp(0.0, 1.0)
    std = ((x - mean) ** 2).mean(dim=-1, keepdim=True).clamp_min(1e-6).sqrt()
    snr = (m - mean) / std
    collision = (e * e).sum(dim=-1, keepdim=True) / (s * s)
    below = xi < threshold
    damping = torch.where(below, torch.exp(-scale * (1.0 - xi)), torch.ones_like(xi)).clamp_min(1e-3)
    gamma = torch.where(below, gain * (1.0 - torch.exp(-scale * (threshold - xi).clamp_min(0.0))), torch.zeros_like(xi))
    flux = x - mean
    out = (mean + torch.where(x < m - 1.0, flux * (1.0 - gamma), flux)) / damping
    return torch.where(snr >= snr_floor, out, x), torch.cat([xi, collision, snr], dim=-1)


KERNEL, KERNEL_REL_DIFF = "pytorch", None
try:
    if not ON_GPU:
        raise RuntimeError("Triton needs a GPU")
    import triton
    import triton.language as tl

    @triton.jit
    def abhaya_gate_v4_kernel(logits_ptr, out_ptr, telemetry_ptr, vocab_size, max_entropy, threshold,
                              dharma_scale, cancel_gain, snr_floor, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        in_row_ptr = logits_ptr + pid * vocab_size
        out_row_ptr = out_ptr + pid * vocab_size
        m_i = -float('inf')
        sum_x = 0.0
        for off in range(0, vocab_size, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < vocab_size
            x = tl.load(in_row_ptr + cols, mask=mask, other=-float('inf'))
            m_i = tl.maximum(m_i, tl.max(x, axis=0))
            sum_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        mean_x = sum_x / vocab_size
        sq_diff_sum = 0.0
        s_i = 0.0
        s2_i = 0.0
        s_x_i = 0.0
        for off in range(0, vocab_size, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < vocab_size
            x = tl.load(in_row_ptr + cols, mask=mask, other=-float('inf'))
            x_safe = tl.where(mask, x, mean_x)
            diff = x_safe - mean_x
            sq_diff_sum += tl.sum(tl.where(mask, diff * diff, 0.0), axis=0)
            exp_vals = tl.where(mask, tl.exp(x - m_i), 0.0)
            s_i += tl.sum(exp_vals, axis=0)
            s2_i += tl.sum(exp_vals * exp_vals, axis=0)
            s_x_i += tl.sum(exp_vals * tl.where(mask, x, 0.0), axis=0)
        std_x = tl.sqrt(tl.maximum(sq_diff_sum / vocab_size, 1e-6))
        snr_z = (m_i - mean_x) / std_x
        collision_c = s2_i / (s_i * s_i)
        entropy = m_i + tl.log(s_i) - (s_x_i / s_i)
        xi = tl.clamp(1.0 - (entropy / max_entropy), 0.0, 1.0)
        tl.store(telemetry_ptr + pid * 3 + 0, xi)
        tl.store(telemetry_ptr + pid * 3 + 1, collision_c)
        tl.store(telemetry_ptr + pid * 3 + 2, snr_z)
        is_clean = snr_z >= snr_floor
        instability = tl.maximum(0.0, threshold - xi)
        damping = tl.exp(-dharma_scale * (1.0 - xi))
        applied_damping = tl.maximum(tl.where(xi < threshold, damping, 1.0), 1e-3)
        gamma = tl.where(xi < threshold, cancel_gain * (1.0 - tl.exp(-dharma_scale * instability)), 0.0)
        for off in range(0, vocab_size, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < vocab_size
            x = tl.load(in_row_ptr + cols, mask=mask, other=-float('inf'))
            noise_flux = x - mean_x
            is_tail = x < (m_i - 1.0)
            cancelled_flux = tl.where(is_tail, noise_flux * (1.0 - gamma), noise_flux)
            crystallized = tl.where(is_clean, (mean_x + cancelled_flux) / applied_damping, x)
            tl.store(out_row_ptr + cols, crystallized, mask=mask)

    def gate_triton(logits, threshold=0.86, scale=3.5, gain=0.50, snr_floor=5.8):
        x = logits.to(torch.float32).contiguous()
        out = torch.empty_like(x)
        rows, vocab = x.shape
        telemetry = torch.empty((rows, 3), device=x.device, dtype=torch.float32)
        abhaya_gate_v4_kernel[(rows,)](x, out, telemetry, vocab, math.log(vocab), threshold, scale, gain,
                                        snr_floor, BLOCK_SIZE=4096)
        return out, telemetry

    torch.manual_seed(0)
    probe = torch.randn((2, 151936), device=device)
    probe[0, 7] += 7.0          # row 0: clean and below the threshold, so the gate rescales it
    probe[1] *= 3.5             # row 1: low SNR, so the kernel passes it through
    a, ta = gate_triton(probe)
    b, tb = gate_torch(probe)
    torch.cuda.synchronize()
    KERNEL_REL_DIFF = float(((a - b).abs().max() / b.abs().max()).item())
    assert KERNEL_REL_DIFF < 1e-3 and float((ta - tb).abs().max()) < 1e-2, (KERNEL_REL_DIFF, ta, tb)
    KERNEL = "triton"
except Exception as exc:  # the PyTorch form is the same arithmetic
    print("Triton kernel not used:", repr(exc)[:300])

run_gate = gate_triton if KERNEL == "triton" else gate_torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
DTYPE = getattr(torch, os.environ.get("GATE_DTYPE", "bfloat16" if ON_GPU else "float32"))
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(device).eval()
print("GPU", GPU, "| torch", torch.__version__, "| kernel", KERNEL,
      "| triton vs pytorch relative difference", KERNEL_REL_DIFF)

OFF = (10**9, 10**9)  # noise window that never opens

# (first option, second option, property, key). Every key is a textbook value.
ITEMS = [
    ("diamond", "copper", "the higher thermal conductivity at room temperature", "diamond"),
    ("oxygen", "nitrogen", "the lower boiling point at 1 atm", "nitrogen"),
    ("Venus", "Mercury", "the shorter orbital period around the Sun", "Mercury"),
    ("Mercury", "Venus", "the higher average surface temperature", "Venus"),
    ("lead", "gold", "the higher density at room temperature", "gold"),
    ("tungsten", "iron", "the higher melting point", "tungsten"),
    ("air", "water", "the higher speed of sound at room temperature", "water"),
    ("potassium", "argon", "the greater standard atomic weight", "argon"),
    ("copper", "silver", "the higher electrical conductivity at room temperature", "silver"),
    ("red light", "blue light", "the longer wavelength", "red light"),
    ("quartz", "topaz", "the greater hardness on the Mohs scale", "topaz"),
    ("helium", "hydrogen", "the lower gas density at the same temperature and pressure", "hydrogen"),
    ("water", "ethanol", "the higher boiling point at 1 atm", "water"),
    ("Titan", "Ganymede", "the larger diameter", "Ganymede"),
    ("tin", "lead", "the lower melting point", "tin"),
    ("proton", "neutron", "the greater mass", "neutron"),
    ("gold", "osmium", "the higher density at room temperature", "osmium"),
    ("Neptune", "Uranus", "the greater average distance from the Sun", "Neptune"),
    ("carbon", "nitrogen", "the larger number of protons in its nucleus", "nitrogen"),
    ("titanium", "aluminium", "the lower density at room temperature", "aluminium"),
    ("Jupiter", "Saturn", "the greater mass", "Jupiter"),
    ("lithium", "sodium", "the lower density at room temperature", "lithium"),
    ("Mars", "Earth", "the longer year", "Mars"),
    ("ultraviolet light", "infrared light", "the higher photon energy", "ultraviolet light"),
    ("platinum", "silver", "the higher melting point", "platinum"),
    ("methane", "propane", "the lower boiling point at 1 atm", "methane"),
    ("calcium", "magnesium", "the greater standard atomic weight", "calcium"),
    ("fluorine", "chlorine", "the higher electronegativity on the Pauling scale", "fluorine"),
    ("zinc", "copper", "the lower melting point", "zinc"),
    ("uranium", "lead", "the larger number of protons in its nucleus", "uranium"),
    ("cobalt", "nickel", "the greater standard atomic weight", "cobalt"),
    ("tellurium", "iodine", "the greater standard atomic weight", "tellurium"),
    ("helium", "neon", "the lower boiling point at 1 atm", "helium"),
    ("Callisto", "Europa", "the larger diameter", "Callisto"),
    ("Jupiter", "Saturn", "the lower mean density", "Saturn"),
    ("Europa", "Io", "the larger diameter", "Io"),
    ("Earth", "Venus", "the longer rotation period on its axis", "Venus"),
    ("Sirius", "Betelgeuse", "the greater distance from Earth", "Betelgeuse"),
    ("deuterium", "tritium", "the larger number of neutrons in its nucleus", "tritium"),
    ("Mercury", "Mars", "the larger diameter", "Mars"),
]


def question_of(a, b, prop):
    return (f"Which has {prop}, {a} or {b}? "
            "Reply with one sentence of the form 'Of the two, the answer is ___.'")


def reference_of(a, b, prop, named):
    return f"Of {a} and {b}, the one with {prop} is {named}."


def prompt_of(question, reference=None):  # the notebook's two prompt formats
    if reference:
        return f"Verified Reference: {reference}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def pattern_of(option):
    word = option.split()[0].lower()
    return r"\balumin(?:i)?um\b" if word.startswith("alumin") else rf"\b{re.escape(word)}s?\b"


def sentence_done(text):
    """True after a full stop that is not a decimal point, or after a line break."""
    body = text.lstrip()
    return "\n" in body or bool(re.search(r"(?<!\d)[.!?]$", body.rstrip(" ")))


def answer_step(tokens, a, b):
    """Index of the generated token that begins the first option the text names. None when it names neither."""
    for k in range(len(tokens)):
        text = tokenizer.decode(tokens[:k + 1], skip_special_tokens=True).lower()
        starts = [m.start() for opt in (a, b) for m in [re.search(pattern_of(opt), text)] if m]
        if starts:
            pos = min(starts)
            for j in range(k + 1):
                if len(tokenizer.decode(tokens[:j + 1], skip_special_tokens=True)) > pos:
                    return j
    return None


def named_in(text, a, b):
    """Which option the first line of the output names. None when it names neither."""
    line = text.strip().split("\n")[0].lower()
    tail = line.split("answer is", 1)[1] if "answer is" in line else line
    for part in (tail, line):
        found = [(m.start(), opt) for opt in (a, b) for m in [re.search(pattern_of(opt), part)] if m]
        if found:
            return min(found)[1]
    return None


@torch.no_grad()
def plain_generate(prompt, use_kernel=False, max_new_tokens=40, stop=True):
    """Greedy decoding. With use_kernel the v4 kernel is applied to every step's logits."""
    cur = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
    past, tokens, active, changed, trace = None, [], 0, 0, []
    for _ in range(max_new_tokens):
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        raw = out.logits[:, -1, :].to(torch.float32)
        top_vals, top_ids = torch.topk(raw, k=2, dim=-1)
        c1, c2 = (tokenizer.decode([top_ids[0, i].item()]) for i in (0, 1))
        collision = float((torch.softmax(raw, dim=-1) ** 2).sum())
        ratio = float(torch.exp(top_vals[0, 0] - top_vals[0, 1]))
        trace.append({"first": c1, "second": c2, "collision": round(collision, 4), "ratio": round(ratio, 3),
                      "condition": bool(collision < 0.52 and ratio < 1.6 and c1.strip().isalpha() and c2.strip().isalpha())})
        pick = raw
        if use_kernel:
            pick, telem = run_gate(raw)
            xi, _, snr = telem[0].tolist()
            active += int(xi < 0.86 and snr >= 5.8)
            changed += int(torch.argmax(pick, dim=-1).item() != torch.argmax(raw, dim=-1).item())
        nxt = torch.argmax(pick, dim=-1, keepdim=True)
        tokens.append(nxt.item())
        cur = nxt
        if nxt.item() == tokenizer.eos_token_id:
            break
        if stop and sentence_done(tokenizer.decode(tokens, skip_special_tokens=True)):
            break
    return {"output": tokenizer.decode(tokens, skip_special_tokens=True).strip(), "token_ids": tokens,
            "steps": len(tokens), "kernel_active_steps": active, "kernel_changed_token": changed, "trace": trace}


@torch.no_grad()
def full_gate(question, reference, inject_noise_steps=OFF, noise_sigma=11.5, max_new_tokens=40, stop=True):
    """abhaya_v4_generate_clean of abhaya_gate(1).ipynb, cell 9, with the trigger returned as data.

    stop=True ends the answer at its first full stop, in every arm alike, so that the trigger
    is not counted for firing on text the model writes after its answer. stop=False is the notebook.
    """
    cur_ids = tokenizer(prompt_of(question), return_tensors="pt").to(device)["input_ids"]
    past_kv, tokens, grounded, trigger, blocked, step = None, [], False, None, 0, 0
    while len(tokens) < max_new_tokens and step < 120:
        prev_len = past_kv.get_seq_length() if past_kv is not None else 0
        out = model(cur_ids, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        raw = out.logits[:, -1, :]
        if inject_noise_steps[0] <= step <= inject_noise_steps[1]:
            torch.manual_seed(200 + step)
            raw = raw + torch.randn_like(raw) * noise_sigma
        gated, telem = run_gate(raw)
        _, col_c, snr_z = telem[0].tolist()
        step += 1
        if snr_z < 5.8:
            blocked += 1
            if prev_len == 0:
                past_kv = None
            else:
                added = past_kv.get_seq_length() - prev_len
                if added > 0:
                    past_kv.crop(-added)
            continue
        top_vals, top_ids = torch.topk(raw.to(torch.float32), k=2, dim=-1)
        ratio = torch.exp(top_vals[0, 0] - top_vals[0, 1]).item()
        c1 = tokenizer.decode([top_ids[0, 0].item()])
        c2 = tokenizer.decode([top_ids[0, 1].item()])
        if (col_c < 0.52 and ratio < 1.6 and c1.strip().isalpha() and c2.strip().isalpha()
                and not grounded and len(tokens) >= 3):
            trigger = {"token_index": len(tokens), "first": c1, "second": c2,
                       "collision": round(col_c, 4), "ratio": round(ratio, 3),
                       "discarded_text": tokenizer.decode(tokens, skip_special_tokens=True)}
            grounded = True
            cur_ids = tokenizer(prompt_of(question, reference), return_tensors="pt").to(device)["input_ids"]
            past_kv, tokens, step = None, [], 0
            continue
        nxt = torch.argmax(gated, dim=-1, keepdim=True)
        tokens.append(nxt.item())
        cur_ids = nxt
        if nxt.item() == tokenizer.eos_token_id:
            break
        if stop and sentence_done(tokenizer.decode(tokens, skip_special_tokens=True)):
            break
    return {"output": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            "fired": trigger is not None, "trigger": trigger, "noise_blocked": blocked}


def at_answer_words(trig, a, b):
    """True when the two candidate tokens at the trigger are the starts of the two options."""
    c1, c2 = trig["first"].strip().lower(), trig["second"].strip().lower()
    w1, w2 = a.split()[0].lower(), b.split()[0].lower()
    return bool(c1 and c2) and ((w1.startswith(c1) and w2.startswith(c2)) or (w1.startswith(c2) and w2.startswith(c1)))


def fisher_one_sided(a, b, c, d):
    """P(first-row count >= a) for the table [[a, b], [c, d]] with fixed margins."""
    row1, col1, n = a + b, a + c, a + b + c + d
    top = sum(math.comb(col1, k) * math.comb(n - col1, row1 - k)
              for k in range(a, min(row1, col1) + 1) if row1 - k <= n - col1)
    return top / math.comb(n, row1)


def sign_test(arm, against):
    """One-sided exact sign test that `arm` is correct on more questions than `against`."""
    up = sum(r["arms"][arm]["mark"] == "correct" and r["arms"][against]["mark"] != "correct" for r in rows)
    down = sum(r["arms"][arm]["mark"] != "correct" and r["arms"][against]["mark"] == "correct" for r in rows)
    n = up + down
    return {"better": up, "worse": down,
            "p": (sum(math.comb(n, k) for k in range(up, n + 1)) / 2 ** n) if n else 1.0}


ARMS = ["plain", "plain_repeat", "kernel_only", "true_ref_always", "false_ref_always", "gate_true_ref", "gate_false_ref"]
t_start = time.time()
rows = []
for a, b, prop, key in ITEMS:
    other = b if key == a else a
    q = question_of(a, b, prop)
    true_ref, false_ref = reference_of(a, b, prop, key), reference_of(a, b, prop, other)
    res = {
        "plain": plain_generate(prompt_of(q)),
        "plain_repeat": plain_generate(prompt_of(q)),
        "kernel_only": plain_generate(prompt_of(q), use_kernel=True),
        "true_ref_always": plain_generate(prompt_of(q, true_ref)),
        "false_ref_always": plain_generate(prompt_of(q, false_ref)),
        "gate_true_ref": full_gate(q, true_ref),
        "gate_false_ref": full_gate(q, false_ref),
    }
    row = {"question": q, "options": [a, b], "key": key, "other": other, "arms": res}
    for arm in ARMS:
        named = named_in(res[arm]["output"], a, b)
        res[arm]["named"] = named
        res[arm]["mark"] = "unscored" if named is None else ("correct" if named == key else "wrong")
    j = answer_step(res["plain"]["token_ids"], a, b)
    row["plain_answer_token"] = None if j is None else dict(res["plain"]["trace"][j], index=j)
    rows.append(row)

def count(arm, mark):
    return sum(r["arms"][arm]["mark"] == mark for r in rows)


totals = {arm: {m: count(arm, m) for m in ("correct", "wrong", "unscored")} for arm in ARMS}

scored = [r for r in rows if r["arms"]["plain"]["mark"] != "unscored"]
wrong_base = [r for r in scored if r["arms"]["plain"]["mark"] == "wrong"]
right_base = [r for r in scored if r["arms"]["plain"]["mark"] == "correct"]
fw = sum(r["arms"]["gate_true_ref"]["fired"] for r in wrong_base)
fr = sum(r["arms"]["gate_true_ref"]["fired"] for r in right_base)
p_detect = fisher_one_sided(fw, len(wrong_base) - fw, fr, len(right_base) - fr) if wrong_base and right_base else None

# The trigger's own condition read at the token where the plain model names its answer, with the
# three-token rule set aside. This is the most favourable reading of the trigger as a detector.
at_wrong = [r["plain_answer_token"] for r in wrong_base if r["plain_answer_token"]]
at_right = [r["plain_answer_token"] for r in right_base if r["plain_answer_token"]]
cw, cr = sum(t["condition"] for t in at_wrong), sum(t["condition"] for t in at_right)
p_answer = fisher_one_sided(cw, len(at_wrong) - cw, cr, len(at_right) - cr) if at_wrong and at_right else None


def auroc_of(wrong_scores, right_scores):
    """Chance that a wrong answer carries a lower score than a correct one. 0.5 is no information."""
    pairs = [(w, r_) for w in wrong_scores for r_ in right_scores]
    return (sum(1.0 if w < r_ else 0.5 if w == r_ else 0.0 for w, r_ in pairs) / len(pairs)) if pairs else None


score_w, score_r = [t["collision"] for t in at_wrong], [t["collision"] for t in at_right]
auroc = auroc_of(score_w, score_r)
auroc_p = None
if auroc is not None:  # one-sided permutation test, 20,000 relabellings, fixed seed
    shuffler = torch.Generator().manual_seed(11)
    pooled, hits = torch.tensor(score_w + score_r), 0
    for _ in range(20000):
        mixed = pooled[torch.randperm(len(pooled), generator=shuffler)].tolist()
        hits += auroc_of(mixed[:len(score_w)], mixed[len(score_w):]) >= auroc
    auroc_p = (hits + 1) / 20001


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None



corrected = sum(r["arms"]["plain"]["mark"] == "wrong" and r["arms"]["gate_true_ref"]["mark"] == "correct" for r in rows)
broken = sum(r["arms"]["plain"]["mark"] == "correct" and r["arms"]["gate_true_ref"]["mark"] != "correct" for r in rows)
missed = len(wrong_base) - fw

repeat_same_text = sum(r["arms"]["plain_repeat"]["output"] == r["arms"]["plain"]["output"] for r in rows)
repeat_same_mark = sum(r["arms"]["plain_repeat"]["mark"] == r["arms"]["plain"]["mark"] for r in rows)
same_text = sum(r["arms"]["kernel_only"]["output"] == r["arms"]["plain"]["output"] for r in rows)
kernel_steps = sum(r["arms"]["kernel_only"]["steps"] for r in rows)
kernel_active = sum(r["arms"]["kernel_only"]["kernel_active_steps"] for r in rows)
kernel_changed = sum(r["arms"]["kernel_only"]["kernel_changed_token"] for r in rows)

fired_false = [r for r in rows if r["arms"]["gate_false_ref"]["fired"]]
followed_false = sum(r["arms"]["gate_false_ref"]["named"] == r["other"] for r in fired_false)
followed_false_always = sum(r["arms"]["false_ref_always"]["named"] == r["other"] for r in rows)
answer_at_trigger = sum(r["arms"]["gate_true_ref"]["fired"]
                        and at_answer_words(r["arms"]["gate_true_ref"]["trigger"], *r["options"]) for r in rows)

vs_plain, vs_ref, kernel_vs_plain = (sign_test("gate_true_ref", "plain"), sign_test("gate_true_ref", "true_ref_always"),
                                     sign_test("kernel_only", "plain"))
tests = {
    "T1_kernel_improves_answers": kernel_changed > 0 and kernel_vs_plain["p"] < 0.05,
    "T2_trigger_separates_wrong_from_right": bool(p_detect is not None and len(wrong_base) and len(right_base)
                                                  and fw / len(wrong_base) > fr / len(right_base) and p_detect < 0.05),
    "T2b_condition_at_answer_token_separates": bool(p_answer is not None and cw / len(at_wrong) > cr / len(at_right)
                                                    and p_answer < 0.05),
    "T3_gate_beats_plain_and_reference_alone": vs_plain["p"] < 0.05 and vs_ref["p"] < 0.05,
    "T4_gate_holds_against_false_reference": totals["gate_false_ref"]["correct"] >= totals["plain"]["correct"],
}

# Part 2. The notebook's own three questions, with its own reference sentences and noise settings.
NOTEBOOK = [
    ("Compare the room-temperature thermal conductivity of pure diamond and copper in one sentence.",
     "Pure diamond has a thermal conductivity of ~2,200 W/(m·K) at room temperature, roughly 5 times higher than copper (~400 W/(m·K)).",
     "Pure diamond has a thermal conductivity of ~80 W/(m·K) at room temperature, roughly 5 times lower than copper (~400 W/(m·K))."),
    ("Is the boiling point of liquid nitrogen higher or lower than that of liquid oxygen at 1 atm? State both temperatures.",
     "At 1 atm, liquid nitrogen boils at 77.36 K (-195.79 °C), which is lower than liquid oxygen's boiling point of 90.19 K (-182.96 °C).",
     "At 1 atm, liquid nitrogen boils at 97.36 K (-175.79 °C), which is higher than liquid oxygen's boiling point of 90.19 K (-182.96 °C)."),
    ("Which planet has a shorter orbital period around the Sun, Mercury or Venus, and by how many Earth days?",
     "Mercury completes an orbit in 88 Earth days compared to Venus's 225 Earth days, making Mercury's orbital period 137 Earth days shorter.",
     "Venus completes an orbit in 88 Earth days compared to Mercury's 225 Earth days, making Venus's orbital period 137 Earth days shorter."),
]
part2 = []
for q, true_fact, false_fact in NOTEBOOK:  # the third sentence of each entry is deliberately false
    part2.append({
        "question": q,
        "plain_no_gate": plain_generate(prompt_of(q), max_new_tokens=48, stop=False)["output"],
        "true_reference_no_gate": plain_generate(prompt_of(q, true_fact), max_new_tokens=48, stop=False)["output"],
        "gate_true_reference": full_gate(q, true_fact, (12, 15), 11.5, 48, stop=False),
        "gate_false_reference": full_gate(q, false_fact, (12, 15), 11.5, 48, stop=False),
    })

record = {
    "schema": "abhaya_gate_control_v1",
    "date_utc": datetime.now(timezone.utc).isoformat(),
    "model": MODEL_ID, "gpu": GPU, "kernel": KERNEL,
    "kernel_triton_vs_pytorch_relative_difference": KERNEL_REL_DIFF,
    "torch": torch.__version__, "dtype": str(next(model.parameters()).dtype), "decode": "greedy", "noise_in_part1": "none",
    "stop_in_part1": "first full stop or line break, all arms",
    "questions": len(rows), "totals": totals,
    "plain_run_twice": {"texts_identical": repeat_same_text, "marks_identical": repeat_same_mark},
    "kernel_only": {"texts_identical_to_plain": same_text, "steps": kernel_steps,
                    "steps_kernel_rescaled_logits": kernel_active, "steps_chosen_token_changed": kernel_changed},
    "trigger": {"plain_wrong": len(wrong_base), "fired_on_plain_wrong": fw,
                "plain_correct": len(right_base), "fired_on_plain_correct": fr,
                "fisher_one_sided_p": p_detect, "fired_at_the_two_answer_words": answer_at_trigger,
                "plain_wrong_corrected": corrected, "plain_wrong_missed": missed, "plain_correct_broken": broken},
    "condition_at_answer_token": {"plain_wrong": len(at_wrong), "condition_true_on_wrong": cw,
                                  "plain_correct": len(at_right), "condition_true_on_correct": cr,
                                  "fisher_one_sided_p": p_answer,
                                  "mean_collision_wrong": mean([t["collision"] for t in at_wrong]),
                                  "mean_collision_correct": mean([t["collision"] for t in at_right]),
                                  "auroc_low_collision_marks_wrong": auroc,
                                  "auroc_permutation_p": auroc_p},
    "false_reference": {"gate_fired": len(fired_false), "output_named_false_answer_after_firing": followed_false,
                        "always_arm_named_false_answer": followed_false_always},
    "paired_sign_tests": {"gate_true_ref_vs_plain": vs_plain, "gate_true_ref_vs_true_ref_always": vs_ref,
                          "kernel_only_vs_plain": kernel_vs_plain},
    "tests": tests, "seconds": round(time.time() - t_start, 1), "rows": rows, "notebook_questions": part2,
}
for r in rows:  # the per-step traces were only needed to find the answer token
    for arm_result in r["arms"].values():
        arm_result.pop("trace", None)
        arm_result.pop("token_ids", None)
path = f"abhaya_gate_control_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}.json"
with open(path, "w") as fh:
    json.dump(record, fh, indent=1, ensure_ascii=False)

short = {"correct": "ok", "wrong": "WRONG", "unscored": "?"}
print(f"=== GATE CONTROL TEST | {MODEL_ID} | {GPU} | kernel {KERNEL} | greedy | {len(rows)} questions ===")
print(f"{'#':>2} {'key':<10} {'plain':<6} {'again':<6} {'kernel':<6} {'trueref':<7} {'falseref':<8} {'gate+true':<9} {'gate+false':<10} trigger")
for i, r in enumerate(rows, 1):
    a_ = r["arms"]
    trig = a_["gate_true_ref"]["trigger"]
    where = f"token {trig['token_index']} {trig['first']!r} vs {trig['second']!r}" if trig else "did not fire"
    print(f"{i:>2} {r['key'][:10]:<10} {short[a_['plain']['mark']]:<6} {short[a_['plain_repeat']['mark']]:<6} "
          f"{short[a_['kernel_only']['mark']]:<6} "
          f"{short[a_['true_ref_always']['mark']]:<7} {short[a_['false_ref_always']['mark']]:<8} "
          f"{short[a_['gate_true_ref']['mark']]:<9} {short[a_['gate_false_ref']['mark']]:<10} {where}")
print("\nTOTALS (correct / wrong / unscored)")
for arm in ARMS:
    t = totals[arm]
    print(f"  {arm:<18} {t['correct']:>2} / {t['wrong']:>2} / {t['unscored']:>2}")
print(f"\nPlain run twice: same text in {repeat_same_text} of {len(rows)}, same mark in {repeat_same_mark}. "
      "Anything short of all is run-to-run noise, and the other arms carry the same noise.")
print(f"Kernel only: text identical to plain in {same_text} of {len(rows)}; kernel rescaled the logits on "
      f"{kernel_active} of {kernel_steps} steps; chosen token changed on {kernel_changed} steps")
print(f"Trigger: fired on {fw} of {len(wrong_base)} questions the plain model got wrong, "
      f"and on {fr} of {len(right_base)} it got right; one-sided Fisher p = {p_detect}")
print(f"Trigger fired exactly at the two answer words in {answer_at_trigger} of "
      f"{sum(r['arms']['gate_true_ref']['fired'] for r in rows)} firings")
print(f"Gate with true reference: corrected {corrected} wrong answers, let {missed} wrong answers through, "
      f"turned {broken} correct answers into wrong or unscored")
print(f"Gate with true reference against plain: better on {vs_plain['better']}, worse on {vs_plain['worse']}, "
      f"p = {vs_plain['p']:.4f}; against the reference with no gate: better on {vs_ref['better']}, "
      f"worse on {vs_ref['worse']}, p = {vs_ref['p']:.4f}")
print(f"Condition at the answer token (three-token rule set aside): true on {cw} of {len(at_wrong)} wrong answers "
      f"and {cr} of {len(at_right)} correct answers; p = {p_answer}")
print(f"Collision at the answer token: mean {mean([t['collision'] for t in at_wrong])} when wrong, "
      f"{mean([t['collision'] for t in at_right])} when correct; AUROC {auroc} (0.5 is no information), "
      f"permutation p = {auroc_p}")
print(f"False reference: gate fired on {len(fired_false)}; output then named the false answer in {followed_false}. "
      f"Without the gate the false answer was named in {followed_false_always} of {len(rows)}")
print("\nTESTS (conditions fixed before the run)")
for name, passed in tests.items():
    print(f"  {name:<44} {'PASS' if passed else 'FAIL'}")
print("\n=== THE NOTEBOOK'S THREE QUESTIONS ===")
for i, p in enumerate(part2, 1):
    print(f"[{i}] {p['question']}")
    print("  plain, no gate          :", p["plain_no_gate"])
    print("  true reference, no gate :", p["true_reference_no_gate"])
    print("  gate + true reference   :", p["gate_true_reference"]["output"], "| fired:", p["gate_true_reference"]["fired"])
    print("  gate + FALSE reference  :", p["gate_false_reference"]["output"], "| fired:", p["gate_false_reference"]["fired"])
print("\nJSON_SUMMARY " + json.dumps({k: record[k] for k in (
    "model", "gpu", "kernel", "questions", "totals", "plain_run_twice", "kernel_only", "trigger", "condition_at_answer_token",
    "false_reference", "paired_sign_tests", "tests")}))
print(f"\nWrote {path} ({record['seconds']} s in total)")
