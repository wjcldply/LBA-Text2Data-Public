import sys
import torch
import numpy as np
import json
from LLaMA3_1.llama3_1_Tester_v6 import Tester_v6
from .compare import compareNplot


def compute_acc(gold_list, extracted_list):
    '''입력 리스트들은 list of dictionaries형임'''
    correct_extractions = [kg for kg in extracted_list if kg in gold_list]

    total_extracted_len = len(extracted_list)
    total_gold_len = len(gold_list)
    correct_extracted_len = len(correct_extractions)

    precision = correct_extracted_len / total_extracted_len if total_extracted_len != 0 else 0
    recall = correct_extracted_len / total_gold_len if total_gold_len != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1
    

def compute_score_for_sentence(gold_triples, extracted_triples):
    """입력파라메터는 모두 list of dicts, 각각은 동일 문장에 대해 추출한 gold 및 extracted 결과인 딕셔너리들이 담긴 리스트들임"""
    num_gold = len(gold_triples)
    num_extracted = len(extracted_triples)

    print(f'''Gold_KGs(#{num_gold}) -> {[f"{g.get('object1', '')} - {g.get('relation', '')} - {g.get('object2', '')}" if len(g)==3 else f"{g.get('object', '')} - {g.get('attribute', '')}" for g in gold_triples]}'''); sys.stdout.flush()
    print(f'''Extracted_KGs(#{num_extracted}) -> {[f"{e.get('object1', '')} - {e.get('relation', '')} - {e.get('object2', '')}" if len(e)==3 else f"{e.get('object', '')} - {e.get('attribute', '')}" for e in extracted_triples]}'''); sys.stdout.flush()

    if num_gold == 0 or num_extracted == 0:
        print(f'Precision:{0.0} | Recall: {0.0} | F1: {0.0}'); sys.stdout.flush()
        return 0.0, 0.0, 0.0

    # Exact-Matching 기반으로 성능평가 수행
    precision, recall, f1 = compute_acc(gold_triples, extracted_triples)
    print(f'Precision:{precision} | Recall: {recall} | F1: {f1}'); sys.stdout.flush()
    return precision, recall, f1

def classify_errors(error_counts, gold_list, extracted_list, schemas):
    # Initialize the error counts
    # error_counts = [0] * 8  # [Empty Gold, Empty Extracted, Wrong Type, Wrong Relation/Attribute in Schema, Wrong Relation/Attribute Outside Schema, Wrong Object in Schema, Wrong Object Outside Schema, Incomplete Extraction]

    # Case 1: Empty Gold
    if not gold_list:
        error_counts[0] += 1
        return error_counts

    # Case 2: Empty Extracted
    if not extracted_list:
        error_counts[1] += 1
        return error_counts

    # Helper function to check if an item is a triple or a dual
    def is_triple(item):
        return len(item) == 3

    def is_dual(item):
        return len(item) == 2

    # Iterate through each extracted triple/dual
    for extracted in extracted_list:
        if extracted in gold_list:
            # Correct match, no need to classify error
            continue

        # Case 8: Incomplete Extraction
        if len(extracted) == 1:
            error_counts[7] += 1
            continue  # Move to the next extracted item

        # Case 3: Wrong Triple/Dual Type
        if (is_triple(extracted) and all(is_dual(g) for g in gold_list)) or (is_dual(extracted) and all(is_triple(g) for g in gold_list)):
            error_counts[2] += 1
            continue  # Move to the next extracted item

        # Case 4/5: Wrong Relation/Attribute Within/Outside Schema (for triples and duals)
        if is_triple(extracted):
            extracted_object1, extracted_relation, extracted_object2 = extracted
            gold_relations = [g[1] for g in gold_list if is_triple(g)]
            
            if extracted_relation not in schemas['relations']:
                error_counts[4] += 1  # Wrong Relation Outside Schema
            elif extracted_relation not in gold_relations:
                error_counts[3] += 1  # Wrong Relation Within Schema

        elif is_dual(extracted):
            extracted_object, extracted_attribute = extracted
            gold_attributes = [g[1] for g in gold_list if is_dual(g)]

            if extracted_attribute not in schemas['attributes']:
                error_counts[4] += 1  # Wrong Attribute Outside Schema
            elif extracted_attribute not in gold_attributes:
                error_counts[3] += 1  # Wrong Attribute Within Schema

        # Case 6/7: Wrong Object Within/Outside Schema
        extracted_objects = [extracted[0]] if is_dual(extracted) else [extracted[0], extracted[2]]
        if len(extracted_objects)==1:
            if extracted_objects[0] not in schemas['objects']:
                error_counts[6] += 1  # Wrong Object Outside Schema
            else:
                error_counts[5] += 1  # Wrong Object Within Schema
        else:
            if extracted_objects[0] not in schemas['objects'] or extracted_objects[1] not in schemas['objects']:
                error_counts[6] += 1  # Wrong Object Outside Schema
            else:
                error_counts[5] += 1  # Wrong Object Within Schema
    return error_counts

def eval_KG(gold_dataset, extracted_dataset, schemas):
    """입력파라메터 모두 list of list of dicts"""
    all_precisions = []  # sentences(cases) 개의 실수값이 들어감
    all_recalls = []  # sentences(cases) 개의 실수값이 들어감
    all_f1s = []  # sentences(cases) 개의 실수값이 들어감
    error_counts = [0] * 8

    for idx, (gold_triples, extracted_triples) in enumerate(zip(gold_dataset, extracted_dataset)):
        # gold_triples, extracted_triples 모두 list of dicts -> for문으로 iter 돌고 있는 건 각 문장에 대한 것들임
        print(f'Sentence #{idx+1}'); sys.stdout.flush()
        precision, recall, f1 = compute_score_for_sentence(gold_triples, extracted_triples)  # 각 문장에 대한 추출결과(list of dicts)들을 파라메터로 전달
        print('----------'*20); sys.stdout.flush()
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        error_counts = classify_errors(error_counts=error_counts, gold_list=[list(item.values()) for item in gold_triples], extracted_list=[list(item.values()) for item in extracted_triples], schemas=schemas)

    avg_precision = np.mean(all_precisions)   # sentences(cases) 개의 실수값의 평균
    avg_recall = np.mean(all_recalls)   # sentences(cases) 개의 실수값의 평균
    avg_f1 = np.mean(all_f1s)   # sentences(cases) 개의 실수값의 평균
    error_cases = {"Empty Gold":error_counts[0],"Empty Extracted":error_counts[1], "Triple/Dual Type Miss":error_counts[2], "Rel/Attr Miss Within Schema":error_counts[3], "Rel/Attr Miss Outside Schema":error_counts[4], "Obj Miss Within Schema":error_counts[5], "Obj Miss Outside Schema":error_counts[6], "Incomplete Extraction":error_counts[7]}

    return avg_precision, avg_recall, avg_f1, all_precisions, all_recalls, all_f1s, error_cases

def test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=None, isMultiTurn=False, label = 'None', model=None, openai_api_key=None, batch_size=1, lora_adapter_path=None):
    """for accuracy-based (exact matching) testing"""
    """입력파라메터 sample_sentences와 sample_gold_KGs는 둘 다 list of list of dicts임"""
    tester = Tester_v6(list_of_sentences=sample_sentences, demonstrations=few_shot_demos, schemas=schemas, multiTurnQA=isMultiTurn, openai_api_key=openai_api_key, batch_size=batch_size, lora_adapter_path=lora_adapter_path)
    sample_extracted_KGs = tester.batch_inference()

    print(f'Counts || Sentences: {len(sample_sentences)} | Gold: {len(sample_gold_KGs)} | Extracted: {len(sample_extracted_KGs)}'); sys.stdout.flush()
    
    avg_precision, avg_recall, avg_f1, all_precisions, all_recalls, all_f1s, error_cases = eval_KG(sample_gold_KGs, sample_extracted_KGs, schemas)
    print(f'{label} || {model}')
    print(f"Average Scores (Exact-Matching Accuracy based) ||  Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}"); sys.stdout.flush()
    print(f"All Precisions: {all_precisions}"); sys.stdout.flush()
    print(f"All Recalls: {all_recalls}"); sys.stdout.flush()
    print(f"All F1 Scores: {all_f1s}"); sys.stdout.flush()
    
    del tester
    torch.cuda.empty_cache()

    return all_precisions, all_recalls, all_f1s, error_cases

sample_KGs = './Data/DramaQA_KG_Processed/KG_GOLD_TEST_500.json'
sample_sentences = [] # list of list of dicts
sample_gold_KGs = []  # list of list of dicts
with open(sample_KGs, 'r') as file:
    data = json.load(file)
data = data[:100]  # Test Purpose

for item in data:
    sentence = item['sentence'] # list of dicts
    gold_KG = item['gold']  # list of dicts

    # Append to respective lists
    sample_sentences.append(sentence)
    sample_gold_KGs.append(gold_KG)

schema_file = './Data/DramaQA_KG_Processed/KG_SCHEMA_TEST.json'
with open(schema_file, 'r') as file:
    schemas = json.load(file)

# Few-Shot Demonstrations - Single Turn
with open('./Data/FewShotDemo/single_turn_demo.json', 'r') as file:
    few_shot_demos_singleTurn = json.load(file)

# Few-Shot Demonstrations - Multi Turn QA
with open('./Data/FewShotDemo/multi_turn_demo.json', 'r') as file:
    few_shot_demos_multiTurnQA = json.load(file)

print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()


labels = [
            'Single-Turn : Zero-Shot + Schemas', 
            'Multi-Turn : Zero-Shot + Schemas', 
            'Single-Turn : One-Shot (1) + Schemas', 
            'Multi-Turn : One-Shot (1) + Schemas',
            'Single-Turn : Few-Shot (10) + Schemas',
            'Multi-Turn : Few-Shot (10) + Schemas'
        ]



""" LLaMA 3.1 8B [LoRA] - EPOCH 1 """
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'LLaMA3.1-8B-LoRA-ZeroShot(Epoch1)'
lora_adapter_path = "./LLaMA3_1/fine_tuning/fine_tuned_models/BackUps/lora3.1_8B_rank32_alpha16_batch1_accumstep4_epoch5_ZeroShot/checkpoint-epoch1"

# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()


########################################################################################################################################################################
print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()

""" LLaMA 3.1 8B [LoRA] - EPOCH 3 """
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'LLaMA3.1-8B-LoRA-ZeroShot(Epoch3)'
lora_adapter_path = "./LLaMA3_1/fine_tuning/fine_tuned_models/BackUps/lora3.1_8B_rank32_alpha16_batch1_accumstep4_epoch5_ZeroShot/checkpoint-epoch3"

# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()


########################################################################################################################################################################
print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()

""" LLaMA 3.1 8B [LoRA] - EPOCH 5 """
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'LLaMA3.1-8B-LoRA-ZeroShot(Epoch5)'
lora_adapter_path = "./LLaMA3_1/fine_tuning/fine_tuned_models/BackUps/lora3.1_8B_rank32_alpha16_batch1_accumstep4_epoch5_ZeroShot/checkpoint-epoch5"

# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=lora_adapter_path)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()


########################################################################################################################################################################
print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()


""" LLaMA 3.1 8B BASE """
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'LLaMA3.1-8B [BASE]'

# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=None, batch_size=2, lora_adapter_path=None)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()


########################################################################################################################################################################
print('=========='*20); sys.stdout.flush()
print('=========='*20); sys.stdout.flush()

""" GPT4o - Mini"""
# Test Cases
precisions = []
recalls = []
f1s = []
all_error_cases = {}

model = 'GPT4o-Mini'

with open('./Test/api_key.txt', 'r') as key_file:
    api_key = key_file.readline().strip()


# 1) Single-Turn : Zero-Shot + Schemas
print(f'# 1) Single-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, label=labels[0], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[0]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 2) Multi-Turn : Zero-Shot + Schemas
print(f'# 2) Multi-Turn : Zero-Shot + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=None, schemas=schemas, isMultiTurn=True, label=labels[1], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[1]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 3) Single-Turn : One-Shot (1) + Schemas
print(f'# 3) Single-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:1], schemas=schemas, label=labels[2], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[2]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 4) Multi-Turn : One-Shot (1) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:1]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : One-Shot (1) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[3], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[3]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 5) Single-Turn : Few-Shot (10) + Schemas'
print(f'# 5) Single-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_singleTurn[:10], schemas=schemas, label=labels[4], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[4]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# 6) Multi-Turn : Few-Shot (10) + Schemas
few_shot_demos_multiTurnQA = few_shot_demos_multiTurnQA[:10]  # Test용
# print(f'Multi-Turn Few-Shot Demos: {few_shot_demos_multiTurnQA}')  # Test용
print(f'# 4) Multi-Turn : Few-Shot (10) + Schemas || {model}'); sys.stdout.flush()
p, r, f, error_cases = test_case(sample_sentences, sample_gold_KGs, few_shot_demos=few_shot_demos_multiTurnQA, schemas=schemas, isMultiTurn=True, label=labels[5], model=model, openai_api_key=api_key, batch_size=5)
precisions.append(p); recalls.append(r); f1s.append(f); all_error_cases[labels[5]]=error_cases
print(error_cases)
print('=========='*20); sys.stdout.flush()

# PLOT
print('Saving Plot...'); sys.stdout.flush()
compareNplot(precisions, recalls, f1s, all_error_cases=all_error_cases, labels=labels, model=model)
print('Test Complete'); sys.stdout.flush()