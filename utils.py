import os
from collections import Counter
from typing import List, Tuple
import random
import numpy as np
from datasets import (Dataset, IterableDataset,)
from tqdm import tqdm
import torch
import math
import evaluate
from transformers import PreTrainedTokenizerBase
import re

def fix_zero_training_loss(model, tokenizer, train_dataset):
    """
    Sometimes the labels get masked by all -100s, causing the loss
    to be 0. We check for this!
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if isinstance(train_dataset, IterableDataset):
        # Skip the check since the code below assumes
        # an indexable dataset
        return
    
    if len(train_dataset) == 0: return


    row = train_dataset[0]
    if type(row) is dict and "labels" in row:

        # Check the first 100 rows
        seen_bad  = 0
        seen_good = 0
        for i, row in enumerate(train_dataset):
            try:    check_tokens = list(set(row["labels"]))
            except: continue
            if len(check_tokens) == 1 and check_tokens[0] == -100: seen_bad += 1
            else: seen_good += 1
            if i >= 100: break
        pass

        # Check ratio
        if seen_bad == 0 and seen_good == 0: return

        elif seen_bad / (seen_bad + seen_good) == 1:
            raise ZeroDivisionError(
                "Unsloth: All labels in your dataset are -100. Training losses will be all 0.\n"\
                "For example, are you sure you used `train_on_responses_only` correctly?\n"\
                "Or did you mask our tokens incorrectly? Maybe this is intended?\n"\
                "Maybe you're using a Llama chat template on a non Llama model for example?"
            )
        elif seen_bad / (seen_bad + seen_good) >= 0.9:
            print(
                "Unsloth: Nearly all labels in your dataset are -100. Training losses will be all 0.\n"\
                "For example, are you sure you used `train_on_responses_only` correctly?\n"\
                "Or did you mask our tokens incorrectly? Maybe this is intended?\n"\
                "Maybe you're using a Llama chat template on a non Llama model for example?"
            )
    pass
pass


def _longest_common_sublist(lists):
    """
    Finds the longest common sublist among multiple lists.

    Parameters:
    lists (List[List[int]]): A list of lists.

    Returns:
    List[int]: The longest common sublist. If multiple sublists have the same maximum length,
               one of them is returned. If there's no common sublist, an empty list is returned.
    """
    if not lists: return []

    # Find the minimum length among all lists
    min_len = min(len(lst) for lst in lists)
    if min_len == 0: return []

    def has_common_sublist(length):
        """
        Checks if there's a common sublist of the given length across all lists.

        Returns:
        (bool, List): Tuple of whether such a sublist exists and the sublist itself.
        """
        common = set()
        first = lists[0]
        # Generate all possible sublists of the given length from the first list
        for i in range(len(first) - length + 1):
            sub = tuple(first[i:i + length])
            common.add(sub)
        pass

        # Iterate over the remaining lists and retain only the common sublists
        for lst in lists[1:]:
            current = set()
            for i in range(len(lst) - length + 1):
                sub = tuple(lst[i:i + length])
                if sub in common:
                    current.add(sub)
            common = current
            if not common:
                return False, []
        pass
        
        # If common is not empty, return one of the common sublists
        return True, list(common.pop())
    pass

    left, right = 1, min_len
    result = []

    while left <= right:
        mid = left + (right - left) // 2
        exists, sublist = has_common_sublist(mid)
        if exists:
            result = sublist  # Update result with the latest found sublist
            left = mid + 1    # Try to find a longer sublist
        else:
            right = mid - 1   # Try with a shorter length
    pass

    return result
pass


def _find_common_token_ids(component, tokenizer, force_match = False):
    """
    \n### User:\n\n
    \n\n### User:\n\n
    etc
    we need to find the middle most repeatted part.
    Tokenizers can tokenize newlines or spaces as 1 token!
    """
    right_text = ""
    if   component.endswith (" "): right_text = " "
    elif component.endswith("\n"): right_text = "\n"
    left_text = ""
    if   component.startswith (" "): left_text = " "
    elif component.startswith("\n"): left_text = "\n"
    stripped = component.strip()
    
    # Add current pieces and also newlines
    all_input_ids = []
    if not force_match:
        for left in range(3):
            for right in range(3):
                x = left*left_text + stripped + right*right_text
                x = tokenizer(x, add_special_tokens = False).input_ids
                all_input_ids.append(x)

                x = left*"\n" + stripped + right*"\n"
                x = tokenizer(x, add_special_tokens = False).input_ids
                all_input_ids.append(x)
            pass
        pass
    else:
        x = tokenizer(component, add_special_tokens = False).input_ids
        all_input_ids.append(x)
    pass

    # Old longest common substring is replaced with actual longest common list of numbers
    # substring = _old_longest_common_substring([str(x + [0]) for x in all_input_ids])
    # substring = substring.split(", ")[:-1]
    # substring = [int(x) for x in substring if x.isdigit()]
    substring = _longest_common_sublist([x + [0] for x in all_input_ids])

    # If substring is simply [0], this might be just the original single token
    # Fixes https://github.com/unslothai/unsloth/issues/1290
    # Mistral [INST] [/INST] singular tokens breaks since we output [0] but we need [3] [4]
    if substring == [0] and len(all_input_ids[0]) == 1:
        single_token = all_input_ids[0][0]
        # Confirm single token in every single possible match
        if all(single_token in x for x in all_input_ids):
            substring = [single_token]
    pass

    # Also if substring is original input_ids + [0], then leave it as the original one
    # This happens when no newlines / spaces are used in chat template
    # Eg Phi-4 does not use newlines or spaces
    if (len(set(str(x) for x in all_input_ids)) == 1) and \
        (len(all_input_ids[0]) + 1 == len(substring)) and \
        (all_input_ids[0] == substring[:-1]):

        # Use original un-changed substring
        substring = all_input_ids[0]
    pass
    
    # Also get rest of tokenized string
    original = tokenizer(component, add_special_tokens = False).input_ids
    # Get optional left and right
    for j in range(len(original)):
        if original[j : j + len(substring)] == substring: break
    optional_left  = original[:j]
    optional_right = original[j+len(substring):]
    return substring, optional_left, optional_right
pass


def train_on_responses_only(
    trainer,
    instruction_part = None,
    response_part    = None,
    force_match      = True,  # Match newlines as well!
    tokenizer        = None,  # Optional
    return_function  = False, # Useful for iterating over lists
    num_proc         = None,
):
    """
    Trains only on responses and not on the instruction by masking out
    the labels with -100 for the instruction part.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if tokenizer is None and trainer is not None:
        tokenizer = trainer.processing_class if hasattr(trainer, "processing_class") else trainer.tokenizer
    # Get non vision tokenizer
    if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if  not hasattr(tokenizer, "_unsloth_input_part") or \
        not hasattr(tokenizer, "_unsloth_output_part"):
        
        if instruction_part is None or response_part is None:
            raise ValueError("Unsloth: instruction_part and response_part must be given!")
        pass
    elif (instruction_part is not None or response_part is not None) and \
        (hasattr(tokenizer, "_unsloth_input_part") or hasattr(tokenizer, "_unsloth_output_part")):

        raise ValueError("Unsloth: Your tokenizer already has instruction and response parts set - do not give custom ones!")
    else:
        instruction_part = tokenizer._unsloth_input_part
        response_part    = tokenizer._unsloth_output_part
    pass

    # Get most common tokens since tokenizers can tokenize stuff differently!
    Q_must, Q_left, Q_right = _find_common_token_ids(instruction_part, tokenizer, force_match)
    A_must, A_left, A_right = _find_common_token_ids(response_part,    tokenizer, force_match)

    # Store some temporary stuff
    A_first = A_must[0]
    len_A_must = len(A_must)
    A_left_reversed = A_left[::-1]
    A_right_forward = A_right

    Q_first = Q_must[0]
    len_Q_must = len(Q_must)
    Q_left_reversed = Q_left[::-1]
    Q_right_forward = Q_right
    torch_Tensor = torch.Tensor
    torch_int64  = torch.int64

    def _train_on_responses_only(examples):
        input_ids_ = examples["input_ids"]
        use_tensors = False
        if type(input_ids_) is torch_Tensor:
            use_tensors = True
            input_ids_ = input_ids_.tolist()
        if "labels" in examples:
            labels_ = examples["labels"].tolist()
            assert(len(labels_) == len(input_ids_))
        else:
            labels_ = [None]*len(input_ids_)

        all_labels = []
        for input_ids, old_labels in zip(input_ids_, labels_):
            n = len(input_ids)
            labels = [-100] * n
            
            use_old_labels = False
            if old_labels is not None:
                use_old_labels = True
                assert(n == len(old_labels))
            n_minus_1 = n - 1
            j = 0
            while j < n:
                # Find <assistant>
                if (input_ids[j] == A_first) and \
                    (input_ids[j : (k := j + len_A_must)] == A_must):

                    # Now backtrack to get previous optional tokens
                    for optional_left in A_left_reversed:
                        if j < 1: break
                        if optional_left == input_ids[j-1]: j -= 1
                        else: break
                    pass
                    # And forwards look as well
                    for optional_right in A_right_forward:
                        if k >= n_minus_1: break
                        if optional_right == input_ids[k+1]: k += 1
                        else: break
                    pass
                    # assistant_j = j
                    assistant_k = k

                    j = assistant_k
                    # Given <assistant>, now find next user
                    while j < n:
                        # Find <user>
                        # Also accept last final item if assistant is the last turn
                        if (j == n_minus_1) or \
                            ((input_ids[j] == Q_first) and \
                             (input_ids[j : (k := j + len_Q_must)] == Q_must)):

                            # Now backtrack to get previous optional tokens
                            for optional_left in Q_left_reversed:
                                if j < 1: break
                                if optional_left == input_ids[j-1]: j -= 1
                                else: break
                            pass
                            # And forwards look as well
                            for optional_right in Q_right_forward:
                                if k >= n_minus_1: break
                                if optional_right == input_ids[k+1]: k += 1
                                else: break
                            pass
                            user_j = j
                            # Account for last item
                            if user_j != n_minus_1:
                                # user_k = k
                                # j = user_k
                                j = k
                            else:
                                user_j = n
                                k = n
                            pass

                            if not use_old_labels:
                                # Now copy input_ids to labels
                                labels[assistant_k : user_j] = input_ids [assistant_k : user_j]
                                # print(assistant_j, assistant_k, user_j, user_k)
                            else:
                                # Copy over from old labels!
                                labels[assistant_k : user_j] = old_labels[assistant_k : user_j]
                            break
                        pass
                        j += 1
                    pass
                pass
                j += 1
            pass
            all_labels.append(labels)
        pass
        return { "labels" : torch.tensor(all_labels, dtype = torch.int64) if use_tensors else all_labels }
    pass
    if return_function:
        return _train_on_responses_only

    from multiprocessing import cpu_count
    if num_proc is None or type(num_proc) is not int: num_proc = cpu_count()

    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        if not hasattr(trainer.train_dataset, "map"):
            raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
        if isinstance(trainer.train_dataset, IterableDataset):
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batch_size = trainer.train_dataset._ex_iterable.batch_size, batched = True)
        else:
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batched = True, num_proc = num_proc)
    pass
    
    if hasattr(trainer, "eval_dataset")  and trainer.eval_dataset  is not None:
        # Eval datasets could be a dict!
        if type(trainer.eval_dataset) is dict:
            for key, value in trainer.eval_dataset.items():
                if not hasattr(value, "map"):
                    raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
                if isinstance(trainer.eval_dataset, IterableDataset):
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batch_size = trainer.eval_dataset._ex_iterable.batch_size, batched = True)
                else:
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batched = True, num_proc = num_proc)
        else:
            if not hasattr(trainer.eval_dataset, "map"):
                raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
            if isinstance(trainer.eval_dataset, IterableDataset):
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batch_size = trainer.eval_dataset._ex_iterable.batch_size, batched = True)
            else:
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batched = True, num_proc = num_proc)
        pass
    pass

    # Edit data collator as well if not DataCollatorForSeq2Seq
    from transformers import DataCollatorForSeq2Seq
    if hasattr(trainer, "data_collator") and \
        not isinstance(trainer.data_collator, DataCollatorForSeq2Seq):
        trainer.data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)

    # Check if all labels randomnly got masked to nothing - maybe wrong chat template?
    fix_zero_training_loss(None, tokenizer, trainer.train_dataset)
    return trainer
pass



def train_test_split(member: str, test_size: float = 0.2, seed: int = 42, data_path: str = '/playpen-ssd/smerrill/dataset') -> Tuple[List[dict], List[dict]]:
    """
    Splits the dataset into training and test sets. Synthetic data is always added to the training set.

    Parameters:
    - member: The name identifier for the board member.
    - test_size: Proportion of the real (non-synthetic) data to include in the test split.
    - seed: Random seed for reproducibility.
    - data_path: Base directory for the dataset files.

    Returns:
    - A tuple (train_data, test_data)
    """
    real_data, synth_data = [], []

    if member == 'kateacuff':
        real_data = np.load(os.path.join(data_path, 'kateacuff_train.npy'))
        synth_data = np.load(os.path.join(data_path, 'synth_kateacuff.npy'))
        test_data = np.load(os.path.join(data_path, 'kateacuff_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'kateacuff_train_completion.npy'), allow_pickle=True)

        
    elif member == 'ellenosborne':
        real_data = np.load(os.path.join(data_path, 'ellenosborne_train.npy'))
        synth_data = np.load(os.path.join(data_path, 'synth_ellenosborne.npy'))
        test_data = np.load(os.path.join(data_path, 'ellenosborne_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'ellenosborne_train_completion.npy'), allow_pickle=True)
        
    elif member == 'grahampaige':
        real_data = np.load(os.path.join(data_path, 'grahampaige_train.npy'))
        synth_data = np.load(os.path.join(data_path, 'synth_grahampaige.npy'))
        test_data = np.load(os.path.join(data_path, 'grahampaige_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'grahampaige_train_completion.npy'), allow_pickle=True)                             
        
    elif member == 'judyle':
        real_data = np.load(os.path.join(data_path, 'judyle_train.npy'))
        synth_data = np.load(os.path.join(data_path, 'synth_judyle.npy'))
        test_data = np.load(os.path.join(data_path, 'judyle_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'judyle_train_completion.npy'), allow_pickle=True)
        
    elif member == 'katrinacallsen':
        real_data = np.load(os.path.join(data_path, 'katrinacallsen_train.npy'))
        test_data = np.load(os.path.join(data_path, 'katrinacallsen_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'katrinacallsen_train_completion.npy'), allow_pickle=True)
        
    elif member == 'davidoberg':
        real_data = np.load(os.path.join(data_path, 'davidoberg_train.npy'))
        test_data = np.load(os.path.join(data_path, 'davidoberg_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'davidoberg_train_completion.npy'), allow_pickle=True)
        
    elif member == 'jonnoalcaro':
        real_data = np.load(os.path.join(data_path, 'jonnoalcaro_train.npy'))
        test_data = np.load(os.path.join(data_path, 'jonnoalcaro_test.npy'), allow_pickle=True)
        train_completion_data = np.load(os.path.join(data_path, 'jonnoalcaro_train_completion.npy'), allow_pickle=True)
        
    else:
        raise ValueError(f"Unknown member: {member}")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float between 0 and 1.")

    train_data = list(real_data) + list(synth_data)
    return train_data, test_data, train_completion_data




def compute_perplexity_on_dataset_accelerate(model, tokenizer, dataset, accelerator, max_length=1024, batch_size=1):
    import math
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset

    class PromptCompletionDataset(TorchDataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            return item['prompt'] + item['completion']

    eval_dataset = PromptCompletionDataset(dataset)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            # Move to the device of the input embedding layer
            embed_device = model.model.embed_tokens.weight.device
            input_ids = encodings.input_ids.to(embed_device)
            attention_mask = encodings.attention_mask.to(embed_device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            losses.append(accelerator.gather(loss.detach()).cpu())
    all_losses = torch.cat(losses)
    mean_loss = all_losses.mean().item()
    perplexity = math.exp(mean_loss)
    return perplexity


def compute_metrics(generated_texts, reference_texts, bleu, rouge, bertscore):
    bleu_score = bleu.compute(predictions=generated_texts, references=[[r] for r in reference_texts])
    rouge_score = rouge.compute(predictions=generated_texts, references=reference_texts)
    bertscore_result = bertscore.compute(predictions=generated_texts, references=reference_texts, lang="en")

    avg_bertscore_f1 = sum(bertscore_result['f1']) / len(bertscore_result['f1'])
    return bleu_score, rouge_score, bertscore_result, avg_bertscore_f1


def get_speaker_special_tokens(train_data):
    def extract_speakers(text):
        return re.findall(r"^(?:speaker \d+|[a-zA-Z0-9_]+):", text, flags=re.MULTILINE)

    speaker_counter = Counter()
    for sample in train_data:
        speakers = extract_speakers(sample["text"])
        speaker_counter.update(speakers)

    speaker_tokens = list(speaker_counter.keys())
    # Add special tokens
    special_tokens = {
    "additional_special_tokens": speaker_tokens + ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    }
    return special_tokens


def debug_tokenization(example_text: str, tokenizer: PreTrainedTokenizerBase):
    tokens = tokenizer(example_text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"][0]
    decoded = [tokenizer.decode([tid]) for tid in input_ids]

    print("=== Tokenized Input ===")
    for i, (tid, tok) in enumerate(zip(input_ids, decoded)):
        print(f"{i:03}: {tid.item():>5}  ->  {repr(tok)}")


def preprocess_test_data(test_data):
    # test_data: list of {'prompt': ..., 'completion': ...}
    # Concatenate prompt and completion for evaluation
    return Dataset.from_list([
        {"text": item["prompt"] + item["completion"]} for item in test_data
    ])


def compute_perplexity_metrics(eval_pred):
    import math
    logits, labels = eval_pred
    # Mask out padding tokens
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    shift_logits = torch.tensor(logits[..., :-1, :])
    shift_labels = torch.tensor(labels[..., 1:])
    loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)
    )
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity}

def wrap_prompt(prompt, agent_name):
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nunknownspeaker:{prompt}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n{agent_name}:" 

def pad_test_data(test_data, train_completion_data, target_size=25):
    test_data = list(test_data)
    train_completion_data = list(train_completion_data)

    if len(test_data) < target_size:
        needed = target_size - len(test_data)
        extra_data = train_completion_data[:needed]
        test_data.extend(extra_data)

    return np.array(test_data)

def add_system_message(prompt, system_message):
    system_message = f"<|begin_of_text|><|system|>\n\n{system_message}<|eot_id|>\n\n"
    return system_message + prompt
