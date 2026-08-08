import torch
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
        logger.info("Initializing AI Models...")
        self._load_diarization()
        self._load_asr()
        self._load_caller_identifier()
        self._load_scam_detector()
        self._load_explainer()
        logger.info("All AI Models loaded")

    def _load_diarization(self) -> None:
        try:
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN
            ).to(torch.device(DEVICE))
        except Exception as e:
            logger.warning(f"Failed to load Pyannote: {e}")
            self.diarization = None

    def _load_asr(self) -> None:
        device_id = 0 if DEVICE == "cuda" else -1
        self.asr = hf_pipeline(
            "automatic-speech-recognition",
            model="biodatlab/distill-whisper-th-small",
            device=device_id
        )
        self.asr.model.config.forced_decoder_ids = self.asr.tokenizer.get_decoder_prompt_ids(
            language="th", task="transcribe"
        )

    def _load_caller_identifier(self) -> None:
        path = MODEL_PATHS["CALLER_IDENTIFIER"]
        try:
            self.caller_tokenizer = AutoTokenizer.from_pretrained(path)
            self.caller_model = AutoModelForSequenceClassification.from_pretrained(path)
            self.caller_model.to(DEVICE)
            self.caller_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load Caller ID model from {path}: {e}")
            self.caller_tokenizer = None
            self.caller_model = None

    def _load_scam_detector(self) -> None:
        path = MODEL_PATHS["SCAM_DETECTOR"]
        sd_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.scam_classifier = hf_pipeline(
            "text-classification",
            model=path,
            tokenizer=sd_tokenizer,
            device=DEVICE
        )

    def _load_explainer(self) -> None:
        self.explainer_slm = ChatOllama(
            model=AGENT_CONFIG["OLLAMA_MODEL"],
            temperature=0.3,
            base_url=AGENT_CONFIG["OLLAMA_BASE_URL"]
        )


def get_models() -> AIModels:
    return AIModels()


