import re
import time
from fractions import Fraction
from collections import defaultdict
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from util.device import get_model_input_device
from glue_eval.useful_functions import get_model_context_length

MAX_NUMBER_OF_FEW_SHOTS = 100
MIN_GENERATION_TOKENS = 64

SUBJECT_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

def _load_math_split(split):
    parts = []
    for cfg in SUBJECT_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg)[split]
        ds = ds.add_column("subject", [cfg] * len(ds))
        parts.append(ds)
    return concatenate_datasets(parts)


def _stratified_select_examples(dataset, count, stratify_key="subject", seed=37):
    if count is None or count >= len(dataset) or stratify_key not in dataset.column_names:
        return _select_examples(dataset, count)

    buckets = defaultdict(list)
    for idx, example in enumerate(dataset):
        buckets[str(example[stratify_key])].append(idx)

    total = len(dataset)
    target = min(count, total)
    subjects = sorted(buckets.keys())
    allocations = {}
    remainders = []
    taken = 0

    for subject in subjects:
        share = (len(buckets[subject]) * target) / total
        base = min(len(buckets[subject]), int(share))
        allocations[subject] = base
        taken += base
        remainders.append((share - base, subject))

    for _, subject in sorted(remainders, reverse=True):
        if taken >= target:
            break
        if allocations[subject] < len(buckets[subject]):
            allocations[subject] += 1
            taken += 1

    chosen_indices = []
    rng = np.random.RandomState(seed)
    for subject in subjects:
        subject_indices = buckets[subject][:]
        rng.shuffle(subject_indices)
        chosen_indices.extend(subject_indices[:allocations[subject]])

    rng.shuffle(chosen_indices)
    return dataset.select(chosen_indices)

def _select_examples(dataset, count):
    if count is None:
        return dataset
    return dataset.select(range(min(count, len(dataset))))


def _normalize_answer(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace('$', '')
    text = text.replace(' ', '')
    text = text.replace('\\left', '').replace('\\right', '')
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = text.replace('\\%', '%')
    text = text.replace('°', '')
    text = text.rstrip('.').rstrip(',')
    return text


def _strip_wrappers(text):
    text = _normalize_answer(text)

    prefixes = [
        "finalanswer:",
        "answer:"
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]

    return text.strip()


def _as_fraction(text):
    cleaned = _strip_wrappers(text)
    if not cleaned:
        return None

    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = cleaned[1:-1]

    cleaned = cleaned.replace('−', '-').replace('–', '-')
    cleaned = cleaned.replace(' ', '')
    cleaned = cleaned.replace('%', '')

    try:
        if '/' in cleaned and not any(ch in cleaned for ch in ['a', 'b', 'c', 'd', 'x', 'y', 'z']):
            return Fraction(cleaned)
        if re.fullmatch(r'[-+]?[0-9]*\.?[0-9]+', cleaned):
            return Fraction(str(float(cleaned))).limit_denominator(10**6)
    except Exception:
        return None

    return None


def _equivalent_answers(predicted, gold):
    pred = _strip_wrappers(predicted)
    gt = _strip_wrappers(gold)

    if pred == gt:
        return True

    pred_frac = _as_fraction(pred)
    gt_frac = _as_fraction(gt)

    if pred_frac is not None and gt_frac is not None:
        return pred_frac == gt_frac

    try:
        pred_float = float(pred)
        gt_float = float(gt)

        return (
            abs(pred_float - gt_float)
            <= max(
                1e-8,
                1e-6 * max(
                    abs(pred_float),
                    abs(gt_float),
                    1.0
                )
            )
        )
    except Exception:
        pass

    return False


def _extract_boxed_answer(text):
    if not text:
        return ""

    boxed_pos = text.rfind("\\boxed{")

    if boxed_pos != -1:
        start = boxed_pos + len("\\boxed{")
        depth = 1
        i = start

        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            return text[start:i - 1].strip()

    matches = re.findall(r'####\s*([^\n]+)', text)
    if matches:
        return matches[-1].strip()

    lower = text.lower()

    markers = [
        "final answer:",
        "answer:"
    ]

    for marker in markers:
        idx = lower.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return lines[-1] if lines else ""


def _extract_numeric_candidate(text):
    if text is None:
        return ""

    candidates = [
        r'[-+]?\d+\/\d+',
        r'[-+]?\d*\.\d+',
        r'[-+]?\d+',
    ]

    for pattern in candidates:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return ""


class MATHEval():
    def __init__(self, model, tokenizer, number_of_tests=None, number_of_few_shots=0, eval_split='test'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = "EleutherAI/hendrycks_math"
        # Build combined train/test across all subject configs
        full_train = _load_math_split("train")
        full_eval = _load_math_split(eval_split)  # eval_split usually "test"

        self.few_shots = _stratified_select_examples(full_train, number_of_few_shots)
        self.eval_dataset = _stratified_select_examples(full_eval, number_of_tests)
        self._initialize_prompts()

    def _extract_ground_truth(self, example):
        return _normalize_answer(_extract_boxed_answer(example['solution']))

    def _get_labels(self, example):
        gold = self._extract_ground_truth(example)
        return gold

    def _initialize_prompts(self):
        self.prefix_prompt = 'Problem: '
        self.postfix_prompt = '\nSolve carefully. Box your final answer using \\boxed{}.\nFinal Answer: '
        self.few_shot_context = []
        for few_shot in self.few_shots:
            answer = self._extract_ground_truth(few_shot)
            self.few_shot_context.append(
                f"{self.prefix_prompt}{few_shot['problem']}{self.postfix_prompt}{answer}\n"
            )

    def _create_prompt(self, example, gen_len):
        question = f"{self.prefix_prompt}{example['problem']}{self.postfix_prompt}"
        question_token_length = len(self.tokenizer(question)["input_ids"])
        remaining_token_length = get_model_context_length(self.model) - question_token_length - gen_len
        actual_few_shot = ""
        for few_shot in self.few_shot_context:
            few_shot_token_length = len(self.tokenizer(few_shot)["input_ids"])
            remaining_token_length -= few_shot_token_length
            if remaining_token_length < 0:
                break
            actual_few_shot += few_shot
        input_prompt = actual_few_shot + question
        return input_prompt, example['problem'], self._extract_ground_truth(example)

    def _get_answer(self, generated_text):
        answer_text = _extract_boxed_answer(generated_text)
        return _normalize_answer(answer_text)

    def evaluate(self, gen_len=2048, print_logs=False):
        generation_tokens = max(gen_len, MIN_GENERATION_TOKENS)
        correct = 0
        incorrect = 0
        invalid = 0
        predictions = []
        labels = []
        stored_generations = []
        start = time.time()

        for s, example in enumerate(self.eval_dataset):
            input_prompt, problem, label = self._create_prompt(example, generation_tokens)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(get_model_input_device(self.model))
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)
            output = self.model.generate(input_prompt_ids, max_new_tokens=generation_tokens, do_sample=False)
            generated_ids = output[0][input_prompt_ids.shape[1]:]

            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            answer = self._get_answer(generated_text)
            predictions.append(answer)
            labels.append(label)

            is_correct = _equivalent_answers(answer, label)
            if answer == "":
                invalid += 1
            elif is_correct:
                correct += 1
            else:
                incorrect += 1

            stored_generations.append({
                'problem': problem,
                'input_prompt': input_prompt_text,
                'true_answer': label,
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': is_correct,
                'correct_new': is_correct,
            })

            if print_logs:
                print(generated_text)
                print(correct, incorrect, invalid, s + 1)
                print('--' * 50)

        end = time.time()
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s + 1,
            # 'accuracy': correct / max(1, (correct + incorrect)),
            'accuracy': correct / max(1, s + 1),
            'time': end - start,
        }
        return result_dict, stored_generations
