import os

import torch
import shutil
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

# ──────────────────────────────────────────────
# 설정 (H200 및 범용 최적화)
# ──────────────────────────────────────────────
MODEL_ID = "google/gemma-3-27b-it"
OUTPUT_DIR = "./gemma-3-27b-it-NVFP4"
NUM_CALIBRATION_SAMPLES = 512 #512
MAX_SEQUENCE_LENGTH = 1024 #1024

# 1. 출력 디렉토리 초기화
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# 2. 모델 및 프로세서 로드
print(f"Loading model: {MODEL_ID}...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    device_map="cpu", 
    torch_dtype="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = processor.tokenizer

# 3. 데이터셋 전처리 (Gemma 3 멀티모달 템플릿 적용)
print("Preprocessing dataset...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess_fn(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    }

ds = ds.map(preprocess_fn, remove_columns=ds.column_names)

# 4. W4A16 레시피 설정 (비전 타워 보호 필히 포함)
# H200에서 안정적인 GPTQ 스타일의 W4A16을 적용합니다.
# recipe = QuantizationModifier(
#     targets="Linear",
#     scheme="W4A16", 
#     ignore=[
#         "lm_head",
#         r"re:.*vision_model.*",       # 비전 타워 레이어 제외 (KeyError 방지)
#         r"re:.*multi_modal_projector.*", # 멀티모달 커넥터 제외
#         r"re:.*connector.*"
#     ],
# )

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        r"re:.*vision_model.*",       # 비전 타워 레이어 제외 (KeyError 방지)
        r"re:.*multi_modal_projector.*", # 멀티모달 커넥터 제외
        r"re:.*connector.*"
    ],
)


# 5. 양자화 실행
print("Starting W4A16 quantization...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    batch_size=4,
    # output_dir=OUTPUT_DIR,
)

# 6. 압축 저장
print("Saving compressed model...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
processor.save_pretrained(OUTPUT_DIR)

print(f"✅ W4A16 양자화 완료: {OUTPUT_DIR}")
