import torch
import librosa
import numpy as np
import time
import os
import logging
from langchain_core.prompts import ChatPromptTemplate
from app.config import SAMPLE_RATE
from app.models import get_models

logger = logging.getLogger("scam_detector.pipeline")

class HybridPipeline:
    def __init__(self) -> None:
        self.models = get_models()

        self.diarization = self.models.diarization
        self.asr = self.models.asr
        self.scam_classifier = self.models.scam_classifier
        self.explainer_slm = self.models.explainer_slm

        self.reset_state()
        self.diarization_cache = {}
        
        # Prompt for warning and advice (when SCAM detected 3 times)
        self.warning_prompt = ChatPromptTemplate.from_messages([
            ("system", """คุณคือผู้ช่วย AI ที่ช่วยปกป้องผู้ใช้จากมิจฉาชีพทางโทรศัพท์
            หน้าที่ของคุณคือเตือนผู้ใช้อย่างจริงจังและให้คำแนะนำที่ปฏิบัติได้จริง
            ตอบเป็นภาษาไทย ใช้ภาษาที่เข้าใจง่าย"""),
                        ("user", """⚠️ ตรวจพบพฤติกรรมหลอกลวงหลายครั้ง!

            ข้อความที่น่าสงสัย:
            {scam_messages}

            กรุณา:
            1. เตือนผู้ใช้ว่านี่คือสายมิจฉาชีพ (1-2 ประโยค)
            2. ระบุเทคนิคหลอกลวงที่ใช้ (bullet points สั้นๆ)
            3. ให้คำแนะนำว่าควรทำอย่างไร (3-4 ข้อ)

            ตอบ:""")
        ])
    
    def reset_state(self):
        """Reset state for new session"""
        self.recent_memory = []
        self.suspicious_memory = []
        self.segment_count = 0
        self.scam_count = 0
        self.scam_messages = []
        self.warning_sent = False
    
    def precompute_diarization(self, audio_path):
        """
        Pre-compute Diarization for audio file (Run on startup)
        Return: list of segments [(start, end, speaker), ...]
        """
        # Check cache first
        cache_key = os.path.basename(audio_path)
        if cache_key in self.diarization_cache:
            print(f"   Using cached diarization for {cache_key}")
            return self.diarization_cache[cache_key]
        
        print(f"   Pre-computing diarization for {cache_key}...")
        start_time = time.time()
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        audio_input = {"waveform": torch.from_numpy(y).float(), "sample_rate": sr}
        
        # Run Diarization
        diarization_output = self.diarization(audio_input, num_speakers=2)
        
        # Extract annotation
        annotation = None
        if hasattr(diarization_output, "itertracks"):
            annotation = diarization_output
        elif hasattr(diarization_output, "annotation"):
            annotation = diarization_output.annotation
        else:
            for attr in dir(diarization_output):
                if attr.startswith("_"):
                    continue
                val = getattr(diarization_output, attr)
                if hasattr(val, "itertracks"):
                    annotation = val
                    break
        
        if not annotation:
            print("   Error: Could not find annotation")
            return []
        
        # Create segments list
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        # Cache it
        self.diarization_cache[cache_key] = segments
        
        elapsed = time.time() - start_time
        print(f"   Diarization complete: {len(segments)} segments in {elapsed:.1f}s")
        
        return segments
    
    def transcribe(self, audio_chunk):
        """Transcribe audio chunk (REALTIME) - use numpy array directly"""
        if len(audio_chunk) < SAMPLE_RATE * 0.3:
            return None
        
        # Convert to dict format supported by HuggingFace (avoid torchcodec)
        audio_input = {
            "raw": audio_chunk.astype(np.float32),
            "sampling_rate": SAMPLE_RATE
        }
            
        generate_kwargs = {
            "max_new_tokens": 128,
            "no_repeat_ngram_size": 3,
            "condition_on_prev_tokens": False,
            "temperature": 0.0
        }
        
        result = self.asr(audio_input, return_timestamps=False, generate_kwargs=generate_kwargs)
        text = result["text"].strip()
        
        return text if len(text) > 2 else None
    
    def detect_scam(self, text):
        """Detect scam with context (REALTIME - BERT)"""
        full_context = ""
        if self.suspicious_memory:
            full_context += "[สัญญาณก่อนหน้า] " + " | ".join(self.suspicious_memory) + " "
        if self.recent_memory:
            full_context += "[บทสนทนาล่าสุด] " + " ".join(self.recent_memory[-3:]) + " "
        full_context += text
        
        result = self.scam_classifier(full_context)[0]
        score = result['score']
        label = result['label']
        
        pred_class = "SCAM" if label in ["SCAM", "LABEL_1"] else "SAFE"
        
        if score < 0.6:
            status = "SAFE"
        elif score < 0.75:
            status = "WAIT"
        else:
            status = pred_class
        
        return status, score, full_context
    
    def explain_scam(self, context):
        """Explain why it is a scam (REALTIME - SLM)"""
        try:
            chain = self.explain_prompt | self.explainer_slm
            response = chain.invoke({"context": context})
            return response.content.strip()
        except Exception as e:
            print(f"   SLM Error (explain_scam): {e}")
            raise e
    
    def generate_warning_advice(self):
        """Generate warning and advice from SLM when SCAM detected 3 times"""
        try:
            scam_text = "\n".join([f"- {msg}" for msg in self.scam_messages])
            chain = self.warning_prompt | self.explainer_slm
            response = chain.invoke({"scam_messages": scam_text})
            return response.content.strip()
        except Exception as e:
            print(f"   SLM Error (generate_warning_advice): {e}")
            raise e
    
    def update_memory(self, text, status, confidence):
        """Update memory"""
        self.recent_memory.append(text)
        if len(self.recent_memory) > 5:
            self.recent_memory.pop(0)
        
        if status in ["WAIT", "SCAM"] and confidence > 0.5:
            if text not in self.suspicious_memory:
                self.suspicious_memory.append(text)
                if len(self.suspicious_memory) > 5:
                    self.suspicious_memory.pop(0)
        
        if status == "SAFE" and confidence > 0.8:
            self.suspicious_memory = []
    
    def identify_caller(self, segments):
        """Identify which speaker is CALLER (first speaker = CALLER)"""
        if not segments:
            return None
        first_speaker = segments[0]["speaker"]
        return first_speaker
    
    def run_hybrid_streaming(self, audio_path: str, simulate_realtime=True):
        """
        Generator: Use Pre-computed Diarization + Realtime ASR/BERT/SLM
        """
        self.reset_state()
        print(f"Hybrid Streaming: {audio_path}")
        
        # 1. Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"   - Duration: {duration:.1f}s")
        
        # 2. Use Pre-computed Diarization (from cache)
        diarization_result = self.diarization_cache.get(os.path.basename(audio_path)) # Use basename for cache key
        if not diarization_result:
            print(f"No cache found for {os.path.basename(audio_path)}. Computing now...")

            diarization_result = self.precompute_diarization(audio_path)
            if not diarization_result:
                print("   Error: No diarization segments found!")
                return
        caller_speaker = self.identify_caller(diarization_result)
        print(f"   - Caller: {caller_speaker}")

        stream_start_time = time.time()
        print(f"   Real-time streaming started...")
        
        # Iterate segments
        for seg in diarization_result: # Iterate over the list of dicts
            start_time = seg["start"]
            end_time = seg["end"]
            speaker = seg["speaker"]


            if simulate_realtime:
                elapsed = time.time() - stream_start_time
                wait_time = end_time - elapsed  # Wait until END not START
                if wait_time > 0:
                    time.sleep(wait_time)
            
            # Send Log: Start Processing
            yield {
                "type": "log",
                "step": "PROCESS",
                "message": f"Processing segment {self.segment_count + 1} ({start_time:.1f}s - {end_time:.1f}s)...",
                "timestamp": time.time()
            }
            
            # Extract audio segment
            start_sample = int(start_time * SAMPLE_RATE)
            end_sample = int(end_time * SAMPLE_RATE)
            speech_audio = y[start_sample:end_sample]
            
            if len(speech_audio) < SAMPLE_RATE * 0.3:
                yield {
                    "type": "log", 
                    "step": "SKIP", 
                    "message": "Segment too short, skipping...",
                    "timestamp": time.time()
                }
                continue
            
            # ========== REALTIME: ASR ==========
            yield {
                "type": "log",
                "step": "ASR",
                "message": "Running Whisper ASR...",
                "timestamp": time.time()
            }
            
            text = self.transcribe(speech_audio)
            
            if not text:
                yield {
                    "type": "log",
                    "step": "ASR",
                    "message": "ASR returned empty text.",
                    "timestamp": time.time()
                }
                continue

            yield {
                "type": "log",
                "step": "ASR",
                "message": f"Transcribed: \"{text}\"",
                "timestamp": time.time()
            }
            
            self.segment_count += 1
            
            # Set role
            role = "CALLER" if speaker == caller_speaker else "RECEIVER"
            
            result = {
                "type": "result", # Mark as normal result
                "start": start_time,
                "end": end_time,
                "speaker": speaker,
                "text": text,
                "status": "SAFE",
                "role": role,
                "reason": "",
                "confidence": 0
            }
            
            # ========== REALTIME: BERT (Scam Detection) ==========
            pending_warning = None
            
            if role == "CALLER":
                # Show input text being analyzed
                short_text = text[:40] + "..." if len(text) > 40 else text
                yield {
                    "type": "log",
                    "step": "BERT",
                    "message": f"New: \"{short_text}\"",
                    "timestamp": time.time()
                }
                
                # Show context (memory)
                if self.suspicious_memory:
                    yield {
                        "type": "log",
                        "step": "BERT",
                        "message": f"History: {len(self.suspicious_memory)} suspicious",
                        "timestamp": time.time()
                    }
                if self.recent_memory:
                    yield {
                        "type": "log",
                        "step": "BERT",
                        "message": f"Context: {len(self.recent_memory)} recent msgs",
                        "timestamp": time.time()
                    }
                
                status, confidence, context = self.detect_scam(text)
                
                # Show results
                yield {
                    "type": "log",
                    "step": "BERT",
                    "message": f"[{status}] ({confidence:.0%})",
                    "timestamp": time.time()
                }
                
                result["status"] = status
                result["confidence"] = confidence
                
                # If BERT detects SCAM
                if status == "SCAM":
                    result["reason"] = "ตรวจพบพฤติกรรมน่าสงสัย"  # Default message
                    
                    self.scam_count += 1
                    self.scam_messages.append(text)
                    print(f"   - SCAM #{self.scam_count}: {text[:50]}...")

                    yield {
                        "type": "log",
                        "step": "BERT",
                        "message": f"SCAM #{self.scam_count}/3",
                        "timestamp": time.time()
                    }
                    
                    if self.scam_count >= 3 and not self.warning_sent:
                        self.warning_sent = True
                        print("   - SCAM detected 3 times! Sending to SLM...")

                        yield {
                            "type": "log",
                            "step": "SLM",
                            "message": " Sending context to Qwen SLM for advice...",
                            "timestamp": time.time()
                        }
                        

                        warning_advice = self.generate_warning_advice()
                        
                        yield {
                            "type": "log",
                            "step": "SLM",
                            "message": "Agent received advice.",
                            "timestamp": time.time()
                        }
                        
                        pending_warning = {
                            "type": "result",
                            "start": result["start"],
                            "end": result["end"],
                            "speaker": "SYSTEM",
                            "text": "",
                            "status": "WARNING",
                            "role": "SYSTEM",
                            "reason": warning_advice,
                            "confidence": 1.0,
                            "is_warning": True
                        }
                
                self.update_memory(text, status, confidence)
            
            # Send segment first
            yield result
            
            # Send WARNING after segment (if exists)
            if pending_warning:
                yield pending_warning
            
            # No delay needed after process because we waited before process
        
        print("Hybrid Streaming Complete!")


# Singleton instance
_pipeline_instance = None

def get_hybrid_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = HybridPipeline()
    return _pipeline_instance

def precompute_audio(audio_path):
    """Pre-compute diarization (startup)"""
    pipeline = get_hybrid_pipeline()
    pipeline.precompute_diarization(audio_path)