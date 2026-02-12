import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot


MODEL_ID = "google/gemma-3-27b-it"

# 1. 모델 및 토크나이저 로드
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    device_map="cpu", 
    dtype=torch.bfloat16, 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. NVFP4 레시피 설정
# Gemma 3의 Linear 레이어들을 대상으로 하며, 출력 헤드는 제외합니다.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head"]
)

# 3. 양자화 실행 (Calibration 데이터셋 필요)
# NVFP4는 글로벌 활성화 스케일 보정을 위해 샘플 데이터가 필요합니다.
oneshot(
    model=model,
    recipe=recipe,
    dataset="ultrachat-200k",
    num_calibration_samples=512,
    max_seq_length=4096,
    output_dir="./gemma-3-27b-it-NVFP4",
)