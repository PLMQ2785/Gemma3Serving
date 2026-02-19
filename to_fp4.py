import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
MODEL_ID = "google/gemma-3-27b-it"
OUTPUT_DIR = "./gemma-3-27b-it-NVFP4"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQ_LENGTH = 4096

# ──────────────────────────────────────────────
# 1. 모델 & 프로세서 로드
# ──────────────────────────────────────────────
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer

# ──────────────────────────────────────────────
# 2. 캘리브레이션 데이터셋 직접 구성
#    (oneshot 내부 파이프라인을 우회)
# ──────────────────────────────────────────────
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def tokenize_fn(example):
    """ultrachat 메시지를 Gemma 3 chat template으로 변환 후 토큰화"""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        return_tensors=None,  # list 형태로 반환
    )
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }

ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

# ──────────────────────────────────────────────
# 3. NVFP4 양자화 레시피
#    - FP4 가중치 양자화 (NVIDIA FP4 형식)
#    - lm_head는 정밀도 유지를 위해 제외
# ──────────────────────────────────────────────
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "re:vision_model.*",
        "re:connector.*"
        ],
)

# ──────────────────────────────────────────────
# 4. 양자화 실행
#    dataset에 전처리 완료된 HF Dataset 직접 전달
# ──────────────────────────────────────────────
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir=OUTPUT_DIR,
)

# ──────────────────────────────────────────────
# 5. 모델 & 프로세서 저장
# ──────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"✅ NVFP4 양자화 완료 → {OUTPUT_DIR}")
