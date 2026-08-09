import torch
import os
import logging
from typing import Optional
from transformers import pipeline as hf_pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyannote.audio import Pipeline
from langchain_ollama import ChatOllama
from app.config import HF_TOKEN, DEVICE, MODEL_PATHS, AGENT_CONFIG

logger = logging.getLogger("scam_detector.models")


class AIModels:
    _instance: Optional["AIModels"] = None

    def __new__(cls) -> "AIModels":
        if cls._instance is None:
            cls._instance = super(AIModels, cls).__new__(cls)
            cls._instance._is_initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_is_initialized", False):
            return
        self.init_models()
        self._is_initialized = True

    def init_models(self) -> None:
        print("Initializing AI Models...")
        self._load_diarization()
        self._load_asr()
        self._load_caller_identifier()
        self._load_scam_detector()
        self._load_explainer()
        print("All AI Models loaded successfully")

    def _load_diarization(self) -> None:
        print("Loading Pyannote Speaker Diarization...")
        try:
            token = HF_TOKEN if HF_TOKEN else None
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token
            ).to(torch.device(DEVICE))
        except Exception as e:
            print(f"Pyannote loading skipped or failed: {e}")
            self.diarization = None

    def _load_asr(self) -> None:
        device_id = 0 if DEVICE == "cuda" else -1
        self.asr = hf_pipeline(
            "automatic-speech-recognition",
            model="biodatlab/distill-whisper-th-small",
            device=device_id
        )

    def _load_caller_identifier(self) -> None:
        path = MODEL_PATHS["CALLER_IDENTIFIER"]
        weights_exist = os.path.exists(os.path.join(path, "pytorch_model.bin")) or os.path.exists(os.path.join(path, "model.safetensors"))
        if not weights_exist:
            path = "airesearch/wangchanberta-base-att-spm-uncased"
        try:
            self.caller_tokenizer = AutoTokenizer.from_pretrained(path)
            self.caller_model = AutoModelForSequenceClassification.from_pretrained(path)
            self.caller_model.to(DEVICE)
            self.caller_model.eval()
        except Exception as e:
            print(f"Failed to load Caller ID model from {path}: {e}")
            self.caller_tokenizer = None
            self.caller_model = None

    def _load_scam_detector(self) -> None:
        path = MODEL_PATHS["SCAM_DETECTOR"]
        weights_exist = os.path.exists(os.path.join(path, "pytorch_model.bin")) or os.path.exists(os.path.join(path, "model.safetensors"))
        if not weights_exist:
            path = "airesearch/wangchanberta-base-att-spm-uncased"
        try:
            sd_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
            self.scam_classifier = hf_pipeline(
                "text-classification",
                model=path,
                tokenizer=sd_tokenizer,
                device=0 if DEVICE == "cuda" else -1
            )
        except Exception as e:
            print(f"Failed to load Scam Detector model from {path}: {e}")
            self.scam_classifier = None

    def _load_explainer(self) -> None:
        self.explainer_slm = ChatOllama(
            model=AGENT_CONFIG["OLLAMA_MODEL"],
            temperature=0.3,
            base_url=AGENT_CONFIG["OLLAMA_BASE_URL"]
        )


_models_instance = None

def get_models() -> AIModels:
    global _models_instance
    if _models_instance is None:
        _models_instance = AIModels()
    return _models_instance


