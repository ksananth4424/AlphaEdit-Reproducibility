from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import get_model_context_length
import time
import torch
import numpy as np
from util.device import get_model_input_device

MAX_NUMBER_OF_FEW_SHOTS = 100


def _select_examples(dataset, count):
    if count is None:
        return dataset
    return dataset.select(range(min(count, len(dataset))))


class HellaSwagEval():
    def __init__(self, model, tokenizer, number_of_tests=None, number_of_few_shots=0, eval_split='validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = "Rowan/hellaswag"
        dataset = load_dataset(self.dataset_name)
        self.few_shots = _select_examples(dataset['train'], number_of_few_shots)
        self.eval_dataset = _select_examples(dataset[eval_split], number_of_tests)
        self._initialize_prompts()

    def _initialize_prompts(self):
        self.prefix_prompt = 'Task: '
        self.postfix_prompt = '\nChoose the best continuation. Answer with only A, B, C, or D: '
        self.few_shot_context = []
        for few_shot in self.few_shots:
            endings = few_shot['endings']
            self.few_shot_context.append(
                f"{self.prefix_prompt}{few_shot.get('activity_label', '')}\nContext: {few_shot.get('ctx', '')}\n(A) {endings[0]}\n(B) {endings[1]}\n(C) {endings[2]}\n(D) {endings[3]}\n{self.postfix_prompt}{chr(65 + int(few_shot['label']))}\n"
            )

    def _create_prompt(self, example, gen_len):
        ctx = example.get('ctx', '')
        if not ctx and 'ctx_a' in example:
            ctx = f"{example.get('ctx_a', '')} {example.get('ctx_b', '')}".strip()
        question = (
            f"{self.prefix_prompt}{example.get('activity_label', '')}\n"
            f"Context: {ctx}\n"
            f"(A) {example['endings'][0]}\n(B) {example['endings'][1]}\n(C) {example['endings'][2]}\n(D) {example['endings'][3]}"
            f"{self.postfix_prompt}"
        )
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
        return input_prompt, ctx, example['endings'], int(example['label']), example.get('activity_label', '')

    def _get_answer(self, generated_text):
        answer_text = generated_text.split(self.postfix_prompt)[-1].strip().strip()
        answer_text = answer_text.lower()
        if answer_text.startswith('a'):
            return 0
        if answer_text.startswith('b'):
            return 1
        if answer_text.startswith('c'):
            return 2
        if answer_text.startswith('d'):
            return 3
        return -1

    def evaluate(self, gen_len=32, print_logs=False):
        a_tok, b_tok, c_tok, d_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['A', 'B', 'C', 'D'])

        if 'llama' in self.model.config._name_or_path.lower():
            a_tok = a_tok[1:]
            b_tok = b_tok[1:]
            c_tok = c_tok[1:]
            d_tok = d_tok[1:]

        a_len, b_len, c_len, d_len = (len(n) for n in [a_tok, b_tok, c_tok, d_tok])
        suffixes = {0: ['A', a_tok, a_len], 1: ['B', b_tok, b_len], 2: ['C', c_tok, c_len], 3: ['D', d_tok, d_len]}

        correct = 0
        incorrect = 0
        invalid = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []
        start = time.time()

        for s, example in enumerate(self.eval_dataset):
            input_prompt, ctx, endings, label, activity = self._create_prompt(example, gen_len)
            input_prompt_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(get_model_input_device(self.model))
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)
            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])

            if 'llama' in self.model.config._name_or_path.lower():
                prefix_tok_len = prefix_tok_len - 1

            max_len = input_prompt_ids.shape[1] + gen_len
            output = self.model.generate(input_prompt_ids, max_length=max_len, do_sample=False)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self._get_answer(generated_text)
            predictions.append(answer)
            labels.append(label)

            probs = [0 for _ in suffixes.keys()]
            for i in range(len(suffixes.keys())):
                prompt_tok = self.tokenizer([f"{input_prompt} {suffixes[i][0]}"] , return_tensors="pt").to(get_model_input_device(self.model))
                with torch.no_grad():
                    logits = self.model(**prompt_tok).logits
                if 'llama' in self.model.config._name_or_path.lower():
                    logits = logits[:, 1:, :]
                cur_len = suffixes[i][2]
                for j in range(cur_len):
                    cur_tok = suffixes[i][1][j]
                    probs[i] += -torch.nn.functional.log_softmax(
                        logits[0, prefix_tok_len + j - 1, :], dim=0
                    )[cur_tok].item()
                probs[i] /= cur_len

            answer_new = int(np.argmax([np.exp(-p) for p in probs]))
            predictions_new.append(answer_new)

            if answer == -1:
                invalid += 1
            else:
                if answer == label:
                    correct += 1
                else:
                    incorrect += 1

            stored_generations.append({
                'activity_label': activity,
                'context': ctx,
                'endings': endings,
                'input_prompt': input_prompt_text,
                'true_answer': chr(65 + label),
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': answer == label,
                'highest_probability_answer': chr(65 + answer_new),
                'correct_new': answer_new == label,
            })

            if print_logs:
                mcc = matthews_corrcoef(labels, predictions)
                f1 = f1_score(labels, predictions, average='weighted')
                print(generated_text)
                print(correct, incorrect, invalid, s + 1, '|ACC: ', correct / (correct + incorrect + invalid), '|MCC:', mcc, '|F1:', f1)
                print('--' * 50)

        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        f1_new = f1_score(labels, predictions_new, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s + 1,
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end - start,
        }
        return result_dict, stored_generations
