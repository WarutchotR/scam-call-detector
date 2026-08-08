import os
from dotenv import load_dotenv

load_dotenv()

# Basic App Configuration
DEVICE = os.getenv("DEVICE", "cuda")
PITCH_ONLY = os.getenv("PITCH_ONLY", "false").lower() == "true"
SAMPLE_RATE = 16000
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Model Paths
_c_caller = r"C:\2Year\KBTGHack\wangchan_finetuned_model2_freeze2\wangchan_finetuned_model2_freeze2\checkpoint-350"
_c_scam = r"C:\2Year\KBTGHack\scam_detector_model\checkpoint-1930_best"

DEFAULT_CALLER_PATH = _c_caller if os.path.exists(_c_caller) else r"D:\KBTG_cybersec\checkpoint-350"
DEFAULT_SCAM_PATH = _c_scam if os.path.exists(_c_scam) else r"D:\KBTG_cybersec\model"

MODEL_PATHS = {
    "CALLER_IDENTIFIER": os.getenv("CALLER_IDENTIFIER_PATH", DEFAULT_CALLER_PATH),
    "SCAM_DETECTOR": os.getenv("SCAM_DETECTOR_PATH", DEFAULT_SCAM_PATH),
}

# Agent & SLM Settings
AGENT_CONFIG = {
    "SLIDING_WINDOW_SIZE": 5,
    "SUSPICIOUS_THRESHOLD": 0.5,
    "MAX_SUSPICIOUS_KEEP": 5,
    "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
    "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
}

