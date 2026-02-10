import json
import threading
import time

# ================= INFINITE STORY MODE =====================
infinite_story_active = False
infinite_story_thread = None
infinite_story_data = {
    'title': 'Untitled Story',
    'chapters': [],
    'timestamp': None,
    'model': None
}
infinite_story_lock = threading.Lock()

def get_story_ai():
    """Return the AI instance for story generation based on config/model settings."""
    global echo_ai
    # Use config or fallback to EchoAI
    model = getattr(config, 'story_model', 'echoai')
    if model == 'pollinations':
        from plaix import PollinationsAI
        return PollinationsAI()
    # Default: EchoAI
    return echo_ai

def infinite_story_worker():
    global infinite_story_active, infinite_story_data
    ai = get_story_ai()
    chapter_num = 1
    print(f"[INFINITE STORY] Worker started. AI: {type(ai).__name__}")
    while infinite_story_active:
        # Build story so far as context (last 5 chapters for context window)
        with infinite_story_lock:
            chapters = list(infinite_story_data.get('chapters', []))
        story_so_far = ""
        if chapters:
            # Only include the last 5 chapters for context (to avoid context overflow)
            for c in chapters[-5:]:
                story_so_far += f"Chapter {c['chapter']}:\n{c['content']}\n\n"
        prompt = f"Continue the story. Chapter {chapter_num}:\n\n{story_so_far}"
        print(f"[INFINITE STORY] Sending prompt: {prompt[:200]}...")
        try:
            # Pass story_so_far as context if supported
            if hasattr(ai, 'generate'):
                chapter = ai.generate(prompt, context={'history': story_so_far} if story_so_far else None)
            else:
                chapter = ai.generate(prompt)
            print(f"[INFINITE STORY] Chapter response: {repr(chapter)}")
            if not chapter or not isinstance(chapter, str) or chapter.strip() == "":
                print(f"[INFINITE STORY] ERROR: Chapter generation failed (empty response)")
                with infinite_story_lock:
                    infinite_story_data['error'] = 'Failed to generate story chapter. Check EchoAI/Ollama backend.'
                break
            with infinite_story_lock:
                infinite_story_data['chapters'].append({
                    'chapter': chapter_num,
                    'content': chapter,
                    'timestamp': time.time()
                })
                infinite_story_data['timestamp'] = time.time()
                infinite_story_data['model'] = getattr(config, 'story_model', 'echoai')
                infinite_story_data.pop('error', None)

            # --- Art Generation for Infinite Story ---
            if getattr(config, 'image_mode_enabled', False):
                print(f"[INFINITE STORY] 🎨 Image mode enabled - generating art for chapter")
                try:
                    import urllib.parse
                    extract_prompt = f"""Based on this story chapter:\n{chapter}\n\nGenerate a creative, vivid image prompt (1-2 sentences) for an AI image generator. Focus on visual concepts, scenes, art styles, colors, and mood. Be specific and artistic.\n\nImage prompt:"""
                    image_prompt = None
                    if ai:
                        image_prompt = ai.generate(extract_prompt)
                    if not image_prompt:
                        image_prompt = f"Abstract art inspired by: {chapter[:100]}"
                    image_prompt = image_prompt.strip('"\'').split('\n')[0].strip()
                    print(f"[INFINITE STORY] [IMAGE] Auto-generated prompt: {image_prompt}")
                    encoded_prompt = urllib.parse.quote(image_prompt)
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
                    global overlay_image_url, overlay_last_update
                    overlay_image_url = image_url
                    overlay_last_update = time.time()
                    print(f"[INFINITE STORY] [IMAGE] ✓ Image sent to overlay: {image_url}")
                except Exception as img_error:
                    print(f"[INFINITE STORY] [IMAGE] Error generating image: {img_error}")

            chapter_num += 1
        except Exception as e:
            print(f"[INFINITE STORY] Exception: {e}")
            with infinite_story_lock:
                infinite_story_data['error'] = f'Exception: {e}'
            break
        time.sleep(2)  # Wait before next chapter

def start_infinite_story(title='Untitled Story'):
    global infinite_story_active, infinite_story_thread, infinite_story_data
    with infinite_story_lock:
        infinite_story_active = True
        infinite_story_data = {
            'title': title,
            'chapters': [],
            'timestamp': time.time(),
            'model': getattr(config, 'story_model', 'echoai')
        }
    infinite_story_thread = threading.Thread(target=infinite_story_worker, daemon=True)
    infinite_story_thread.start()
    print(f"[INFINITE STORY] Started: {title}")

def stop_infinite_story():
    global infinite_story_active
    infinite_story_active = False
    print("[INFINITE STORY] Stopped.")

def get_infinite_story_progress():
    with infinite_story_lock:
        return dict(infinite_story_data)

def save_infinite_story(filename=None):
    with infinite_story_lock:
        data = dict(infinite_story_data)
    if not filename:
        filename = f"story_{int(time.time())}.json"
    print(f"[INFINITE STORY] Saving story to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[INFINITE STORY] Saved to {filename}")
    return filename

def load_infinite_story(filename):
    print(f"[INFINITE STORY] Loading story from {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with infinite_story_lock:
        global infinite_story_data
        infinite_story_data = data
    print(f"[INFINITE STORY] Loaded from {filename}")

    @api_app.route('/api/start_infinite_story', methods=['POST'])
    def api_start_infinite_story():
        title = request.json.get('title', 'Untitled Story')
        print(f"[INFINITE STORY] API handler called. Title: {title}")
        start_infinite_story(title)
        # Wait briefly for the thread to start and possibly fail
        time.sleep(1)
        story = get_infinite_story_progress()
        print(f"[INFINITE STORY] API handler story progress: {story}")
        if 'error' in story:
            print(f"[INFINITE STORY] API returning error: {story['error']}")
            return jsonify({'status': 'error', 'message': story['error'], 'title': title}), 500
        print(f"[INFINITE STORY] API returning started for title: {title}")
        return jsonify({'status': 'started', 'title': title})

    @api_app.route('/api/stop_infinite_story', methods=['POST'])
    def api_stop_infinite_story():
        stop_infinite_story()
        return jsonify({'status': 'stopped'})

    @api_app.route('/api/get_infinite_story', methods=['GET'])
    def api_get_infinite_story():
        story = get_infinite_story_progress()
        return jsonify(story)

    @api_app.route('/api/save_infinite_story', methods=['POST'])
    def api_save_infinite_story():
        filename = request.json.get('filename')
        print(f"[INFINITE STORY] API: Save requested for filename: {filename}")
        fname = save_infinite_story(filename)
        print(f"[INFINITE STORY] API: Save completed for filename: {fname}")
        return jsonify({'status': 'saved', 'filename': fname})

    @api_app.route('/api/load_infinite_story', methods=['POST'])
    def api_load_infinite_story():
        filename = request.json.get('filename')
        print(f"[INFINITE STORY] API: Load requested for filename: {filename}")
        if not filename:
            print(f"[INFINITE STORY] API: Load failed, no filename provided")
            return jsonify({'status': 'error', 'message': 'No filename provided'}), 400
        load_infinite_story(filename)
        print(f"[INFINITE STORY] API: Load completed for filename: {filename}")
        return jsonify({'status': 'loaded', 'filename': filename})
# ================= COMMENTARY LINES PROVIDER =================
# Returns (goal_lines, shot_lines, possession_lines, save_lines) for current mode
def get_commentary_lines(mode=None):
    """
    Returns tuples of (goal_lines, shot_lines, possession_lines, save_lines) for the given mode.
    If mode is None, uses config.game_mode if available, else defaults to 'rocket_league'.
    """
    global config
    try:
        game_mode = mode or getattr(config, 'game_mode', 'rocket_league')
    except Exception:
        game_mode = 'rocket_league'
    if game_mode == 'hockey':
        goal_lines = [
            "SCORES! What a shot!",
            "It's in the net! Goal!",
            "He buries the puck!",
            "GOAL! The crowd goes wild!",
            "He lights the lamp!",
            "That's a beauty!",
            "He finds twine!",
            "He beats the goalie!",
            "He puts it past the netminder!",
            "He snipes it home!"
        ]
        shot_lines = [
            "fires a rocket on net!",
            "lets it rip!",
            "unleashes a slapshot!",
            "takes a wrister!",
            "blasts a one-timer!",
            "puts the puck on goal!",
            "shoots from the point!",
            "snaps it on target!",
            "sends a backhander!",
            "tests the goalie!"
        ]
        possession_lines = [
            "controls the puck!",
            "skates into the zone!",
            "leads the attack!",
            "sets up the play!",
            "dangles through defenders!",
            "moves up ice!",
            "carries the puck!",
            "starts the rush!",
            "keeps the play alive!",
            "holds the blue line!"
        ]
        save_lines = [
            "makes a huge save!",
            "robs him with the glove!",
            "stones the shooter!",
            "denies the goal!",
            "flashes the leather!",
            "kicks it away!",
            "stands tall in net!",
            "turns aside the shot!",
            "keeps it out!",
            "what a stop!"
        ]
    elif game_mode == 'soccer':
        goal_lines = [
            "GOAL! What a strike!",
            "It's in! GOLAZO!",
            "He scores! Brilliant finish!",
            "GOAL! The net bulges!",
            "He finds the back of the net!",
            "What a goal! Spectacular!",
            "He beats the keeper!",
            "Goal! Pure class!",
            "He buries it! GOAL!",
            "Incredible finish!"
        ]
        shot_lines = [
            "shoots on target!",
            "fires toward goal!",
            "unleashes a strike!",
            "takes a shot!",
            "blasts it goalward!",
            "puts the ball on frame!",
            "shoots from range!",
            "sends one on target!",
            "takes a crack at goal!",
            "tests the goalkeeper!"
        ]
        possession_lines = [
            "controls the ball!",
            "drives forward!",
            "leads the attack!",
            "sets up the play!",
            "dribbles past defenders!",
            "moves upfield!",
            "carries possession!",
            "starts the counter!",
            "keeps it alive!",
            "holds in midfield!"
        ]
        save_lines = [
            "brilliant save by the keeper!",
            "denies the shot!",
            "keeps it out!",
            "what a save!",
            "parries it away!",
            "stretches to stop it!",
            "magnificent from the goalkeeper!",
            "tips it over the bar!",
            "gets a hand to it!",
            "outstanding save!"
        ]
    else:
        # Default: Rocket League
        goal_lines = [
            "SCORES! What a goal!",
            "It's in! Goal!",
            "He puts it in the net!",
            "GOAL! The crowd erupts!",
            "That's a rocket!",
            "He finds the back of the net!",
            "He beats the keeper!",
            "He slams it home!",
            "He buries it!",
            "He nails the shot!"
        ]
        shot_lines = [
            "fires a shot on goal!",
            "lets it fly!",
            "unleashes a powerful shot!",
            "takes a quick shot!",
            "blasts it on target!",
            "puts the ball on net!",
            "shoots from distance!",
            "snaps it on frame!",
            "sends a banger!",
            "tests the goalie!"
        ]
        possession_lines = [
            "controls the ball!",
            "drives upfield!",
            "leads the attack!",
            "sets up the play!",
            "dribbles past defenders!",
            "moves downfield!",
            "carries the ball!",
            "starts the push!",
            "keeps the play alive!",
            "holds midfield!"
        ]
        save_lines = [
            "makes a clutch save!",
            "denies the shot!",
            "blocks it on the line!",
            "keeps it out!",
            "what a stop!",
            "turns it away!",
            "stands tall in goal!",
            "parries it clear!",
            "gets a big save!",
            "incredible stop!"
        ]
    return goal_lines, shot_lines, possession_lines, save_lines
import threading
# ==========================================
# Optimized startup, resources, and model loading
# ==========================================

import sys
import os
import threading
import time
import traceback
import subprocess

# ==========================================
# 1. FIXED: PYINSTALLER & VENV LOGIC
# ==========================================
def is_frozen():
    return hasattr(sys, '_MEIPASS')

# ONLY relaunch with venv if we are NOT running as an .exe
if not is_frozen():
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(this_dir, 'py312_venv', 'Scripts' if os.name == 'nt' else 'bin', 'python.exe' if os.name == 'nt' else 'python')
        if os.path.exists(venv_python):
            current = os.path.abspath(sys.executable)
            if os.path.abspath(venv_python).lower() != current.lower():
                print(f"[ENV] Relaunching with venv python: {venv_python}")
                os.execv(venv_python, [venv_python] + sys.argv)
    except Exception as e:
        print(f"[ENV] Venv relaunch check failed: {e}")

# ==========================================
# 2. FIXED: RESOURCE PATHING
# ==========================================
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if is_frozen():
        # Inside EXE: Look in the temp folder (_MEIPASS)
        return os.path.join(sys._MEIPASS, relative_path)
    
    # In Dev: Look in the actual script directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# ==========================================
# 3. FIXED: LOGGING & STREAM REDIRECT
# ==========================================
this_dir = os.path.dirname(os.path.abspath(sys.executable if is_frozen() else __file__))

def _ensure_stream(stream_name):
    stream = getattr(sys, stream_name, None)
    if stream is None or not hasattr(stream, 'write'):
        try:
            log_path = os.path.join(this_dir, 'plaix.log')
            f = open(log_path, 'a', encoding='utf-8')
            setattr(sys, stream_name, f)
        except:
            pass

_ensure_stream('stdout')
_ensure_stream('stderr')

# ==========================================
# 4. IMPORTS & MODEL LOADING
# ==========================================
print("=== PLAIX.PY STARTING ===")
try:
    import cv2
    import numpy as np
    import torch
    import queue
    import easyocr
    from collections import deque
    from ultralytics import YOLO
    print("[OK] Critical modules loaded.")
except Exception as e:
    print(f"[FATAL] Import error: {e}")
    sys.exit(1)

# --- Initialize EasyOCR availability flag (always defined) ---
EASY_OCR_AVAILABLE = 'easyocr' in globals()
if EASY_OCR_AVAILABLE:
    print("[OK] EasyOCR available")
else:
    print("[WARN] EasyOCR not available (optional)")


def load_yolo_model(model_name, device):
    # This matches the folder structure inside your PyInstaller --add-data
    path = resource_path(os.path.join("weights", model_name))
    print(f"[MODEL] Attempting to load: {path}")
    
    if not os.path.exists(path):
        # Fallback: check if the file is just sitting next to the .exe
        alt_path = os.path.join(os.path.dirname(sys.executable), "weights", model_name)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Could not find {model_name} at {path} or {alt_path}")
            
    model = YOLO(path)
    try:
        model.to(device)
        if device == 'cuda':
            import torch
            torch.cuda.synchronize()
            print("[OK] YOLO running on CUDA")
        else:
            print("[OK] YOLO running on CPU")
    except Exception as e:
        print(f"[WARN] Device move failed or CUDA unavailable: {e}")
    return model

# Determine device and load default model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Initial device: {device}")
model = load_yolo_model("yolov11x-rocketleague-best.pt", device)
print("Model classes:", model.names)

TEAM_ABBR_MAP = {
    "Toronto Maple Leafs": "TOR", "Montreal Canadiens": "MTL", "Boston Bruins": "BOS", "Edmonton Oilers": "EDM",
    "Vancouver Canucks": "VAN", "Calgary Flames": "CGY", "Ottawa Senators": "OTT", "Winnipeg Jets": "WPG",
    "Colorado Avalanche": "COL", "Chicago Blackhawks": "CHI", "Detroit Red Wings": "DET", "Dallas Stars": "DAL",
    "Nashville Predators": "NSH", "St. Louis Blues": "STL", "Minnesota Wild": "MIN", "Arizona Coyotes": "ARI",
    "Vegas Golden Knights": "VGK", "San Jose Sharks": "SJS", "Los Angeles Kings": "LAK", "Anaheim Ducks": "ANA",
    "Tampa Bay Lightning": "TBL", "Florida Panthers": "FLA", "Carolina Hurricanes": "CAR", "Columbus Blue Jackets": "CBJ",
    "New York Rangers": "NYR", "New York Islanders": "NYI", "New Jersey Devils": "NJD", "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "Washington Capitals": "WSH", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF"
}
rosters = {"home": {}, "away": {}}
jersey_map = {}
ocr_queue = queue.Queue()
if EASY_OCR_AVAILABLE:
    try:
        reader = easyocr.Reader(['en'], gpu=True)
        print("[OK] EasyOCR reader initialized")
    except Exception as e:
        EASY_OCR_AVAILABLE = False
        reader = None
        print(f"[WARNING] EasyOCR reader initialization failed: {e}")

def fetch_roster(team_name):
    abbr = TEAM_ABBR_MAP.get(team_name)
    if not abbr: return {}
    try:
        url = f"https://api-web.nhle.com/v1/roster/{abbr}/20252026"
        data = requests.get(url, timeout=5).json()
        roster = {}
        for pos in ["forwards", "defensemen", "goaltenders"]:
            for p in data.get(pos, []):
                num = p.get("sweaterNumber")
                name = f"{p['firstName']['default']} {p['lastName']['default']}"
                if num: roster[str(num)] = name
        return roster
    except: return {}

def load_rosters(home, away):
    global rosters
    rosters["home"] = fetch_roster(home) if home else {}
    rosters["away"] = fetch_roster(away) if away else {}
    print(f"Rosters loaded: {home or 'None'} vs {away or 'None'}")

def ocr_worker():
    import traceback
    while True:
        try:
            track_id, crop = ocr_queue.get(timeout=0.1)
            print(f"[OCR] Processing track_id={track_id}")
            if crop is None:
                print(f"[OCR] Crop is None for track_id={track_id}")
            else:
                print(f"[OCR] Crop type: {type(crop)}, shape: {getattr(crop, 'shape', 'N/A')}")
            result = reader.readtext(crop, allowlist='0123456789')
            print(f"[OCR] Result for track_id={track_id}: {result}")
            num = max((r[1] for r in result if r[1].isdigit() and len(r[1]) <= 2), default=None, key=lambda x: result[0][2] if result else 0)
            if num:
                team = "home"
                if num in rosters["away"]: team = "away"
                elif num in rosters["home"]: team = "home"
                jersey_map[track_id] = {"num": num, "team": team}
                print(f"[OCR] Jersey mapped: track_id={track_id}, num={num}, team={team}")
        except Exception as e:
            import queue as _queue
            if isinstance(e, _queue.Empty):
                pass  # Suppress normal queue timeout
            else:
                print(f"[OCR] Exception: {e}")
                traceback.print_exc()
if EASY_OCR_AVAILABLE:
    threading.Thread(target=ocr_worker, daemon=True).start()
import pygame
import tempfile
import requests
import webbrowser
from echo_ai import EchoAI
from datetime import datetime

import wave
import soundfile as sf
# Edge TTS
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    print("[OK] Edge TTS available")
except Exception as e:
    EDGE_TTS_AVAILABLE = False
    print(f"[WARNING] Edge TTS not available: {e}")


# Try Whisper for voice chat (prefer CUDA when available)
try:
    import whisper
    import pyaudio
    # Prefer CUDA if PyTorch reports availability, otherwise fallback to CPU
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("base", device=whisper_device)
    WHISPER_AVAILABLE = True
    print(f"[OK] Whisper loaded for voice chat ({whisper_device.upper()} mode)")
    if whisper_device == 'cpu':
        print('[WARNING] CUDA not available - Whisper running on CPU. For best performance install CUDA-enabled PyTorch and ensure GPU is accessible.')
except Exception as e:
    WHISPER_AVAILABLE = False
    print(f"[WARNING] Whisper not available: {e}")

# Try Kokoro TTS (faster alternative) - FIXED VERSION
try:
    from kokoro_onnx import Kokoro
    
    model_path = resource_path("src/voices/kokoro-v1.0.onnx")
    voices_path = resource_path("src/voices/voices-v1.0.bin")
    
    if os.path.exists(model_path) and os.path.exists(voices_path):
        kokoro_engine = Kokoro(model_path, voices_path)
        KOKORO_AVAILABLE = True
        print("[OK] Kokoro TTS loaded successfully (v1.0)")
    else:
        raise FileNotFoundError("Missing kokoro-v1.0.onnx or voices-v1.0.bin")
        
except Exception as e:
    print(f"[WARNING] Kokoro TTS not available: {e}")
    print("   Download from: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0")
    KOKORO_AVAILABLE = False
    kokoro_engine = None



# ============================================================
#                   HARD CUDA FAILSAFE (GLOBAL)
# ============================================================

CUDA_DISABLED = False

def disable_cuda_permanently():
    global CUDA_DISABLED
    CUDA_DISABLED = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("🔥 CUDA permanently disabled. Running on CPU only.")

# ============================================================
#                   PYGAME AUDIO INIT
# ============================================================

pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=2048)
pygame.mixer.set_num_channels(16)

# ============================================================
#                   DEVICE DETECTION + SAFETY
# ============================================================

device = "cuda"

print(f"🔧 Initial device: {device}")

# At startup, purge CUDA errors if any
if device == "cuda":
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"⚠ Could not synchronize CUDA: {e}")
        raise RuntimeError("CUDA is required but not available or failed to initialize.")

# ============================================================
#                SAFE YOLO LOAD (PATCHED)
# ============================================================

def load_yolo_model(path, device):
    global CUDA_DISABLED

    try:
        print("📦 Loading YOLO model…")
        model = YOLO(path)
        try:
            model.to("cuda")
            torch.cuda.synchronize()
            print("[OK] YOLO running on CUDA")
        except Exception as e:
            print(f"[ERROR] CUDA load failed: {e}")
            raise RuntimeError("CUDA is required but not available or failed to initialize.")
        return model
    except Exception as e:
        print(f"[ERROR] YOLO failed to load: {e}")
        raise

model = load_yolo_model(resource_path(os.path.join("weights", "yolov11x-rocketleague-best.pt")), device)

print("Model classes:", model.names)

# ============================================================
#                   SAFE TTS LOAD (PATCHED)
# ============================================================

def load_tts_engine(device):
    global CUDA_DISABLED
    # Only return an actual TTS engine object for XTTS fallback
    if KOKORO_AVAILABLE:
        print("🔊 Using Kokoro TTS (fast mode)")
        return None  # Kokoro is handled directly, not as an object
    if EDGE_TTS_AVAILABLE:
        print("🔊 Using Edge TTS")
        return None  # Edge TTS is handled directly, not as an object
    print("[ERROR] No TTS engine available!")
    return None

tts = load_tts_engine(device)

# ============================================================
#                   VOICES
# ============================================================

KOKORO_VOICES = [
    "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_fenrir", "am_michael", "am_puck",
    "bf_alice", "bf_emma", "bf_isabella",
    "bm_daniel", "bm_george", "bm_lewis"
]



EDGE_TTS_VOICES = []
def get_edge_voices():
    global EDGE_TTS_VOICES
    if not EDGE_TTS_AVAILABLE:
        return []
    import asyncio
    async def fetch_voices():
        voices = await edge_tts.list_voices()
        return [v['ShortName'] for v in voices]
    try:
        EDGE_TTS_VOICES = asyncio.run(fetch_voices())
    except Exception as e:
        print(f"[EdgeTTS] Failed to fetch voices: {e}")
        EDGE_TTS_VOICES = []
    return EDGE_TTS_VOICES

# Pollinations models and voices
POLLINATIONS_MODELS = [
    {"name": "openai", "type": "chat", "censored": True, "description": "OpenAI GPT-4o-mini", "baseModel": True, "vision": True},
    {"name": "openai-large", "type": "chat", "censored": True, "description": "OpenAI GPT-4o", "baseModel": True, "vision": True},
    {"name": "openai-reasoning", "type": "chat", "censored": True, "description": "OpenAI o3-mini", "baseModel": True, "reasoning": True, "vision": False},
    {"name": "qwen-coder", "type": "chat", "censored": True, "description": "Qwen 2.5 Coder 32B", "baseModel": True, "vision": False},
    {"name": "llama", "type": "chat", "censored": False, "description": "Llama 3.3 70B", "baseModel": True, "vision": False},
    {"name": "mistral", "type": "chat", "censored": False, "description": "Mistral Small 3.1 2503", "baseModel": True, "vision": True},
    {"name": "mistral-roblox", "type": "chat", "censored": False, "description": "Mistral Roblox on Scaleway", "baseModel": True, "vision": False},
    {"name": "roblox-rp", "type": "chat", "censored": True, "description": "Roblox Roleplay Assistant", "baseModel": True, "vision": False},
    {"name": "unity", "type": "chat", "censored": False, "description": "Unity with Mistral Large by Unity AI Lab", "baseModel": False, "vision": False},
    {"name": "midijourney", "type": "audio", "censored": True, "description": "Midijourney (Stable Audio via Stability)", "baseModel": False},
    {"name": "rtist", "type": "chat", "censored": True, "description": "Rtist image generator by @bqrio", "baseModel": False, "vision": False},
    {"name": "searchgpt", "type": "chat", "censored": True, "description": "SearchGPT with realtime news and web search", "baseModel": False, "vision": False},
    {"name": "deepseek", "type": "chat", "censored": True, "description": "DeepSeek-V3", "baseModel": True, "vision": False},
    {"name": "deepseek-r1", "type": "chat", "censored": True, "description": "DeepSeek R1 Distill Qwen 32B", "baseModel": True, "reasoning": True, "provider": "cloudflare", "vision": False},
    {"name": "deepseek-reasoner", "type": "chat", "censored": True, "description": "DeepSeek R1 - Full", "baseModel": True, "reasoning": True, "provider": "deepseek", "vision": False},
    {"name": "deepseek-r1-llama", "type": "chat", "censored": True, "description": "DeepSeek R1 - Llama 70B", "baseModel": True, "reasoning": True, "provider": "scaleway", "vision": False},
    {"name": "qwen-reasoning", "type": "chat", "censored": True, "description": "Qwen QWQ 32B - Advanced Reasoning", "baseModel": True, "reasoning": True, "provider": "groq", "vision": False},
    {"name": "llamalight", "type": "chat", "censored": False, "description": "Llama 3.1 8B Instruct", "baseModel": True, "maxTokens": 7168, "vision": False},
    {"name": "llamaguard", "type": "safety", "censored": False, "description": "Llamaguard 7B AWQ", "baseModel": False, "provider": "cloudflare", "maxTokens": 4000, "vision": False},
    {"name": "phi", "type": "chat", "censored": True, "description": "Phi-4 Instruct", "baseModel": True, "provider": "cloudflare", "vision": False},
    {"name": "phi-mini", "type": "chat", "censored": True, "description": "Phi-4 Mini Instruct", "baseModel": True, "provider": "azure", "vision": False},
    {"name": "gemini", "type": "chat", "censored": True, "description": "Gemini 2.0 Flash", "baseModel": True, "provider": "google", "vision": True},
    {"name": "gemini-thinking", "type": "chat", "censored": True, "description": "Gemini 2.0 Flash Thinking", "baseModel": True, "provider": "google", "vision": True},
    {"name": "hormoz", "type": "chat", "description": "Hormoz 8b by Muhammadreza Haghiri", "baseModel": True, "provider": "modal", "vision": False},
    {"name": "hypnosis-tracy", "type": "chat", "description": "Hypnosis Tracy 7B - Self-help AI assistant", "baseModel": False, "provider": "openai", "vision": False},
    {"name": "sur", "type": "chat", "censored": True, "description": "Sur AI Assistant", "baseModel": False, "vision": False},
    {"name": "sur-mistral", "type": "chat", "censored": True, "description": "Sur AI Assistant (Mistral)", "baseModel": False, "vision": False},
    {"name": "llama-scaleway", "type": "chat", "censored": False, "description": "Llama (Scaleway)", "baseModel": True, "vision": False},
    {"name": "openai-audio", "type": "tts", "censored": True, "description": "OpenAI GPT-4o-audio-preview (TTS)", "baseModel": True, "audio": True}
]

POLLINATIONS_VOICES = ['nova', 'echo', 'spark', 'shimmer']

# ============================================================
#                   POLLINATIONS API FUNCTIONS
# ============================================================

def pollinations_generate_text(prompt, model="openai", max_tokens=32000):
    """Generate text using Pollinations API"""
    try:
        url = f"https://text.pollinations.ai/{prompt}?model={model}&private=true"
        if max_tokens:
            url += f"&max_tokens={max_tokens}"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            text = response.text.strip()
            
            # Try to parse as JSON if it looks like JSON response
            if text.startswith('{') and '"role"' in text:
                try:
                    data = json.loads(text)
                    # Extract just the content, ignore reasoning
                    if isinstance(data, dict):
                        # Try different common response formats
                        if 'content' in data:
                            return data['content']
                        elif 'message' in data and isinstance(data['message'], dict):
                            return data['message'].get('content', text)
                        elif 'choices' in data and len(data['choices']) > 0:
                            return data['choices'][0].get('message', {}).get('content', text)
                    # If we couldn't extract, return original
                    return text
                except json.JSONDecodeError:
                    # Not valid JSON, return as-is
                    return text
            
            return text
        else:
            print(f"[Pollinations] Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"[Pollinations] Text generation failed: {e}")
        return None

def pollinations_generate_audio(text, voice="nova", model="openai-audio"):
    """Generate audio using Pollinations API"""
    try:
        # Encode the text for URL
        encoded_text = requests.utils.quote(text)
        url = f"https://text.pollinations.ai/{encoded_text}?model={model}&voice={voice}&private=true"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            print(f"[Pollinations] Audio generation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"[Pollinations] Audio generation failed: {e}")
        return None

def pollinations_generate_image(prompt, model="openai", enhance=True, nologo=True):
    """Generate image using Pollinations API"""
    try:
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?private=true"
        if not enhance:
            url += "&enhance=false"
        if nologo:
            url += "&nologo=true"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            print(f"[Pollinations] Image generation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"[Pollinations] Image generation failed: {e}")
        return None

# ============================================================
#                   POLLINATIONS AI CLASS (SIMILAR TO ECHOAI)
# ============================================================

class PollinationsAI:
    def __init__(self, model="openai", voice="nova", max_tokens=32000):
        self.model = model
        self.voice = voice
        self.max_tokens = max_tokens
        self.personality = "default"
        self.system_prompt = "You are a helpful AI assistant."
    
    def get_prompt_by_id(self, personality_id):
        """Get personality prompt from EchoAI's personalities"""
        try:
            from echo_ai import EchoAI
            personalities = EchoAI.get_personalities_js()
            for p in personalities:
                if p.get('id') == personality_id:
                    return p.get('prompt', '')
            return None
        except Exception as e:
            print(f"[Pollinations] Failed to get personality: {e}")
            return None

    def set_personality(self, personality: str, game_mode: str = "rocket_league", is_voice_chat: bool = False):
        """Change personality using EchoAI's personality database"""
        print(f"[Pollinations DEBUG] Attempting to set personality: '{personality}' (voice_chat={is_voice_chat})")
        
        # Try to get prompt from PERSONALITIES_JS (new format)
        prompt = self.get_prompt_by_id(personality)
        
        if prompt:
            # If in voice chat mode, clean streaming/follower messages
            if is_voice_chat:
                prompt = clean_prompt_for_voice_chat(prompt)
                print(f"[Pollinations] Cleaned prompt for voice chat mode")
            
            self.personality = personality
            self.system_prompt = prompt
            print(f"[Pollinations] ✓ Personality changed to: {personality}")
            print(f"[Pollinations] ✓ System prompt set to: {self.system_prompt[:100]}...")
        else:
            print(f"[Pollinations] ❌ Unknown personality: '{personality}'")
            print(f"[Pollinations] Using default personality as fallback")
            self.personality = "default"
            # Try to get default, or use generic fallback
            default_prompt = self.get_prompt_by_id("default")
            self.system_prompt = default_prompt if default_prompt else "You are a helpful AI assistant."

    def set_model(self, model):
        self.model = model

    def set_voice(self, voice):
        self.voice = voice

    def chat(self, user_message):
        """Generate a chat response using personality system prompt"""
        # Combine system prompt with user message
        full_prompt = f"{self.system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        return pollinations_generate_text(full_prompt, self.model, self.max_tokens)
    
    def generate(self, prompt, context=None):
        """Generate response (compatible with EchoAI interface)"""
        # Use personality system prompt + user prompt + optional history
        history_text = ""
        if context and 'history' in context and context['history']:
            history_text = f"\n\nStory so far:\n{context['history']}"
        full_prompt = f"{self.system_prompt}{history_text}\n\n{prompt}\n\nAssistant:"
        return pollinations_generate_text(full_prompt, self.model, self.max_tokens)

    def goal_scored(self, speed, scorer):
        """Generate goal commentary using personality"""
        context = f"GOAL SCORED! {scorer} just scored with a {speed} mph shot!"
        full_prompt = f"{self.system_prompt}\n\nGenerate exciting goal commentary for this event: {context}\n\nKeep it under 50 words and stay in character.\n\nCommentary:"
        return pollinations_generate_text(full_prompt, self.model, 100)

    def shot_on_net(self, speed, shooter):
        """Generate shot commentary using personality"""
        context = f"BIG SHOT! {shooter} fires a {speed} mph rocket at the net!"
        full_prompt = f"{self.system_prompt}\n\nGenerate exciting shot commentary for this event: {context}\n\nKeep it under 40 words and stay in character.\n\nCommentary:"
        return pollinations_generate_text(full_prompt, self.model, 80)

    def filler_commentary(self, context, game="Rocket League"):
        """Generate filler commentary using personality"""
        full_prompt = f"{self.system_prompt}\n\nGenerate casual {game} commentary about: {context}\n\nKeep it under 30 words and stay in character.\n\nCommentary:"
        return pollinations_generate_text(full_prompt, self.model, 60)

# ============================================================
#                   RL CONFIG OBJECT
# ============================================================

class RLConfig:
    def __init__(self):
        self.aspect_ratio_min = 0.3
        self.aspect_ratio_max = 3.5
        self.min_ball_area = 50
        self.max_ball_pct = 0.1

        self.goal_cooldown = 4.0
        self.shot_speed_threshold = 15
        self.shot_cooldown = 5.0

        self.your_player_name = None
        self.min_gap = 1.5
        self.volume = 0.8
        self.enable_debug = False

        self.use_echo_ai = True
        self.echo_max_tokens = 32000
        self.ollama_model = "dolphin-mistral:latest"
        self.echo_personality = "Professional"
        self.custom_character_prompt = None

        self.tts_speaker = "af_bella"  # Default Kokoro voice
        self.tts_engine = "kokoro"  # Default TTS engine

        self.team_color = "Blue"
        self.home_team = "Toronto Maple Leafs"
        self.away_team = "Montreal Canadiens"
        self.monitor_index = 1  # Default to monitor 1 (primary 1920x1080)

        self.voice_chat_enabled = True
        self.use_wake_word = False
        self.wake_word = "echo"
        
        # TikTok chat settings
        self.tiktok_enabled = False
        self.tiktok_username = ""
        self.tiktok_cookie = ""  # Optional session/cookie string if needed
        self.tiktok_roast_on_mention = True
        self.tiktok_max_roasts_per_minute = 3

        # AI Provider settings
        self.ai_provider = "pollinations"  # "echo_ai" or "pollinations"
        self.pollinations_model = "openai"  # Default Pollinations model
        self.pollinations_voice = "nova"  # Default Pollinations voice
        
        # Highlight clipper settings
        self.enable_highlights = True
        self.clip_duration = 15  # seconds before event
        self.auto_save_goals = True
        self.auto_save_shots = False  # Only epic shots
        
        # Overlay visualization settings
        self.show_tracking = False  # Show detection boxes in overlay
        self.enable_ball_trail = False
        self.trail_length = 30  # Number of trail points
        self.trail_style = "glow"  # glow, solid, fade, rainbow, comet, neon
        self.ball_color_override = None  # None or (R, G, B) tuple
        self.game_mode = "rocket_league"  # Game mode: "rocket_league" or "hockey"
        self.manual_yolo_model = None  # If set, overrides auto model selection

        # Optional manual calibration: unreal units per pixel (uu/px). If None, auto-estimation will be used when possible.
        self.uu_per_pixel = None

config = RLConfig()

def save_config():
    """Save configuration to JSON file"""
    import json
    try:
        config_data = {
            'your_player_name': config.your_player_name,
            'min_gap': config.min_gap,
            'volume': config.volume,
            'tts_speaker': config.tts_speaker,
            'tts_engine': config.tts_engine,
            'team_color': config.team_color,
            'use_echo_ai': config.use_echo_ai,
            'ollama_model': config.ollama_model,
            'echo_personality': config.echo_personality,
            'echo_max_tokens': config.echo_max_tokens,
            'custom_character_prompt': config.custom_character_prompt,
            'show_tracking': config.show_tracking,
            'enable_ball_trail': config.enable_ball_trail,
            'trail_length': config.trail_length,
            'trail_style': config.trail_style,
            'enable_highlights': config.enable_highlights,
            'clip_duration': config.clip_duration,
            'auto_save_goals': config.auto_save_goals,
            'auto_save_shots': config.auto_save_shots,
            'ai_provider': config.ai_provider,
            'home_team': getattr(config, 'home_team', ''),
            'away_team': getattr(config, 'away_team', ''),
            'pollinations_model': config.pollinations_model,
            'game_mode': getattr(config, 'game_mode', 'rocket_league'),
            'pollinations_voice': config.pollinations_voice,
            'twitch_enabled': getattr(config, 'twitch_enabled', False),
            'twitch_channel': getattr(config, 'twitch_channel', ''),
            'twitch_oauth': getattr(config, 'twitch_oauth', ''),
            'twitch_respond_to_chat': getattr(config, 'twitch_respond_to_chat', True),
            'twitch_respond_to_subs': getattr(config, 'twitch_respond_to_subs', True),
            'twitch_respond_to_follows': getattr(config, 'twitch_respond_to_follows', True),
            'tiktok_enabled': getattr(config, 'tiktok_enabled', False),
            'tiktok_username': getattr(config, 'tiktok_username', ''),
            'tiktok_cookie': getattr(config, 'tiktok_cookie', ''),
            'tiktok_roast_on_mention': getattr(config, 'tiktok_roast_on_mention', True),
            'tiktok_max_roasts_per_minute': getattr(config, 'tiktok_max_roasts_per_minute', 3),
            'uu_per_pixel': getattr(config, 'uu_per_pixel', None)
        }
        with open('plaix_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        print("[CONFIG] Settings saved to plaix_config.json")
    except Exception as e:
        print(f"[CONFIG] Failed to save settings: {e}")

def load_config():
    """Load configuration from JSON file"""
    import json
    import os
    try:
        if os.path.exists('plaix_config.json'):
            with open('plaix_config.json', 'r') as f:
                config_data = json.load(f)
            
            config.your_player_name = config_data.get('your_player_name')
            config.min_gap = config_data.get('min_gap', 1.5)
            config.volume = config_data.get('volume', 0.8)
            config.tts_speaker = config_data.get('tts_speaker', 'af_bella')
            if not isinstance(config.tts_speaker, str):
                config.tts_speaker = 'af_bella'  # Reset if corrupted
            config.tts_engine = config_data.get('tts_engine', 'kokoro')
            config.team_color = config_data.get('team_color', 'Blue')
            config.use_echo_ai = config_data.get('use_echo_ai', True)
            config.ollama_model = config_data.get('ollama_model', 'dolphin-mistral:latest')
            config.echo_personality = config_data.get('echo_personality', 'Professional')
            config.echo_max_tokens = config_data.get('echo_max_tokens', 32000)
            config.custom_character_prompt = config_data.get('custom_character_prompt')
            config.show_tracking = config_data.get('show_tracking', False)
            config.enable_ball_trail = config_data.get('enable_ball_trail', False)
            config.trail_length = int(config_data.get('trail_length', 30))
            config.trail_style = config_data.get('trail_style', 'glow')
            config.enable_highlights = config_data.get('enable_highlights', True)
            config.clip_duration = config_data.get('clip_duration', 15)
            config.auto_save_goals = config_data.get('auto_save_goals', True)
            config.auto_save_shots = config_data.get('auto_save_shots', False)
            config.ai_provider = config_data.get('ai_provider', 'echo_ai')
            config.pollinations_model = config_data.get('pollinations_model', 'openai')
            config.pollinations_voice = config_data.get('pollinations_voice', 'nova')
            config.game_mode = config_data.get('game_mode', 'rocket_league')
            config.home_team = config_data.get('home_team', 'Toronto Maple Leafs')
            # Calibration: allow persisting a manual scale (uu per pixel)
            config.uu_per_pixel = config_data.get('uu_per_pixel', None)
            config.away_team = config_data.get('away_team', 'Montreal Canadiens')
            config.twitch_enabled = config_data.get('twitch_enabled', False)
            config.twitch_channel = config_data.get('twitch_channel', '')
            config.twitch_oauth = config_data.get('twitch_oauth', '')
            config.twitch_respond_to_chat = config_data.get('twitch_respond_to_chat', True)
            config.twitch_respond_to_subs = config_data.get('twitch_respond_to_subs', True)
            config.twitch_respond_to_follows = config_data.get('twitch_respond_to_follows', True)
            # TikTok settings
            config.tiktok_enabled = config_data.get('tiktok_enabled', False)
            config.tiktok_username = config_data.get('tiktok_username', '')
            config.tiktok_cookie = config_data.get('tiktok_cookie', '')
            config.tiktok_roast_on_mention = config_data.get('tiktok_roast_on_mention', True)
            config.tiktok_max_roasts_per_minute = config_data.get('tiktok_max_roasts_per_minute', 3)
            load_rosters(config.home_team, config.away_team)
            
            print("[CONFIG] Settings loaded from plaix_config.json")
        else:
            print("[CONFIG] No saved settings found, using defaults")
    except Exception as e:
        print(f"[CONFIG] Failed to load settings: {e}")
# Model switching logic
def get_model_path_for_mode(mode):
    # If manual override is set, always use it
    if getattr(config, 'manual_yolo_model', None):
        return config.manual_yolo_model
    if mode == "hockey":
        return os.path.join("weights", "yolov11x.pt")
    elif mode == "soccer":
        # Soccer mode requires manual model selection from overlay
        # No default - will use last selected model or prompt user to choose
        if getattr(config, 'manual_yolo_model', None):
            return config.manual_yolo_model
        # If no manual selection, fall back to Rocket League model
        return os.path.join("weights", "yolov11x-rocketleague-best.pt")
    return os.path.join("weights", "yolov11x-rocketleague-best.pt")

def reload_yolo_model():
    global model
    model_path = get_model_path_for_mode(getattr(config, 'game_mode', 'rocket_league'))
    resolved_model_path = resource_path(model_path)
    print(f"[MODEL] Loading YOLO model for mode: {getattr(config, 'game_mode', 'rocket_league')} -> {resolved_model_path}")
    model = load_yolo_model(resolved_model_path, device)
    global loaded_yolo_model_path
    loaded_yolo_model_path = resolved_model_path
    print("Model classes:", model.names)

reload_yolo_model()

# ============================================================
#           COMMENTARY LINES
# ============================================================

goal_lines = ["GOOOOOOAAAAAALLLLLL!!!", "THEY SCORE!!!", "IT'S IN THE NET!!!", "WHAT A GOAL!!!", "UNBELIEVABLE!"]
shot_lines = ["Big shot on goal!", "Rips it!", "Dangerous attempt!", "Hammered toward the net!", "What a strike!"]
possession_lines = ["has possession!", "driving forward!", "on the attack!", "looking dangerous!", "controls the ball!"]
filler_lines = ["What a match!", "The intensity is rising!", "Both teams battling hard!", "Incredible action!", "This is Rocket League!"]

# ============================================================
#                     GLOBAL STATE
# ============================================================

BALL = 0
ENEMY = 1
ENEMY_GOALPOST = 2
MY_GOALPOST = 3
TEAMMATE = 4


ball_trail = deque(maxlen=300)
last_comment_time = 0
last_chat_response_time = 0  # Separate timer for Twitch/chat responses
last_goal_time = -10
last_shot_time = -10
last_possession_time = -10
GAME_STATE = "menu"
processing = False
state_change_counter = 0
ball_in_goal_prev = False
goals_scored = 0
in_replay = False
replay_cooldown_time = 0
last_activity_score = 0

tts_speaking = False
voice_chat_active = False
voice_chat_status = "⚪ Idle"  # Status indicator for UI
live_preview_frame = None  # Stores current frame for preview
speech_queue = []  # Queue for speech to prevent overlaps
speech_lock = threading.Lock()  # Lock for thread-safe queue access
ollama_lock = threading.Lock()  # Lock to prevent simultaneous Ollama requests
echo_ai = None  # Global AI instance for text chat and commentary
last_tts_finish_time = 0  # Track when TTS last finished (for cooldown)
last_spoken_text = ""  # Track last thing spoken to filter feedback
last_any_commentary_time = 0  # Track when ANY audio played (commentary OR voice chat)

# Interactive content box state
content_box_data = None  # Data for content box display requests
content_box_sent = False  # Track if content_box_data has been sent to overlay

# Initialize AI at startup (must be before Flask API starts)
print(f"[STARTUP] 🤖 Initializing AI provider: {config.ai_provider}")
try:
    from echo_ai import EchoAI
    
    # Note: PollinationsAI is just a mode in EchoAI, not a separate class
    # Always use EchoAI, it handles both Ollama and Pollinations internally
    echo_ai = EchoAI(
        enabled=True,
        temperature=0.8,
        max_tokens=config.echo_max_tokens,
        ollama_model=config.ollama_model,
        personality=config.echo_personality
    )
    print(f"[STARTUP] ✓ Initialized EchoAI with model: {config.ollama_model}")
    
    # Set personality
    if config.custom_character_prompt:
        echo_ai.system_prompt = config.custom_character_prompt
        print(f"[STARTUP] Using custom character prompt")
    else:
        if hasattr(echo_ai, 'set_personality'):
            echo_ai.set_personality(config.echo_personality, game_mode=getattr(config, 'game_mode', 'rocket_league'))
            print(f"[STARTUP] Set personality to: {config.echo_personality}")

    # --- Background Ollama warm-up (non-blocking) ---
    try:
        if getattr(echo_ai, 'use_ollama', False) and hasattr(echo_ai, 'warmup_model'):
            # Start background warmup to avoid long first-request stalls (adjust max_wait/interval as needed)
            threading.Thread(target=lambda: echo_ai.warmup_model(max_wait=180, interval=5), daemon=True).start()
            print("[STARTUP] Background Ollama warmup started")
    except Exception as e:
        print(f"[STARTUP] Warmup failed to start: {e}")
except Exception as ai_init_error:
    print(f"[STARTUP] ⚠️ Failed to initialize AI: {ai_init_error}")
    print(f"[STARTUP] AI will be initialized on first use")
    import traceback
    traceback.print_exc()

# Soccer goal detection state
soccer_goalkeeper_positions = {"left": None, "right": None}  # Track goalkeeper positions
soccer_last_ball_position = None
soccer_ball_missing_frames = 0

# Phoneme data for lip-sync
current_phonemes = []  # List of {phoneme, start, duration} for current TTS
phonemes_sent = False  # Track if current phonemes have been sent to overlay

# Overlay variables for game UI
overlay_state = "idle"  # idle, thinking, speaking, listening
overlay_caption_text = {"text": "", "speaker": "Echo AI"}  # Current TTS text and speaker
overlay_voice_input = ""  # What the user said (transcription)
overlay_image_url = None  # Generated image URL (for image mode)
overlay_last_update = time.time()
overlay_tracking_data = {}  # Detection boxes, ball position, etc.
ball_trail_points = deque(maxlen=100)  # Store ball positions for trail

# Calibration: pixels <-> Unreal Units (1 uu = 1 cm).
# Auto-estimated from goal post detections or set manually via config (config.uu_per_pixel)
pixel_to_uu = None  # units: uu per pixel (uu/px). Multiply pixel distance by this to get uu.
pixel_to_uu_alpha = 0.20  # EWMA smoothing factor when auto-updating scale
pixel_to_uu_last_update = 0  # timestamp of last auto calibration

# Image saving directory
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)
print(f"[IMAGES] Save directory: {os.path.abspath(IMAGES_DIR)}")

# Highlight clipper variables
clip_recording = False
clip_buffer = deque(maxlen=600)  # Store last 20 seconds at 30fps
clip_writer = None
highlight_clips = []  # List of saved clips

# --- Enemy goal smoothing state ---
smoothed_enemy_goal_x = None
enemy_goal_smooth_alpha = 0.2  # Smoothing factor (0.1-0.3 is typical)
enemy_goal_last_detected = 0
enemy_goal_confidence = 0
enemy_goal_confidence_max = 10

# ============================================================
#                       SAFE TTS SPEAK (KOKORO FIXED)
# ============================================================

def estimate_phonemes_from_text(text):
    """
    Estimate phoneme sequence from text for lip-sync.
    Returns list of {phoneme, start, duration} dicts.
    This is a simplified version - for production, use a proper G2P (grapheme-to-phoneme) library.
    """
    words = text.split()
    phonemes = []
    time_offset = 0.0
    
    # Simple vowel/consonant to phoneme mapping
    phoneme_map = {
        'a': 'AA', 'e': 'EH', 'i': 'IY', 'o': 'OW', 'u': 'UW',
        'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
        'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
        'n': 'N', 'p': 'P', 'r': 'R', 's': 'S', 't': 'T',
        'v': 'V', 'w': 'W', 'y': 'Y', 'z': 'Z'
    }
    
    for word in words:
        word_lower = word.lower().strip('.,!?;:')
        if not word_lower:
            continue
            
        # Simple phoneme generation with deduplication
        last_phoneme = None
        for char in word_lower:
            if char in phoneme_map:
                phoneme = phoneme_map[char]
                # Skip duplicate consecutive phonemes
                if phoneme == last_phoneme:
                    continue
                # Base durations: vowels 0.08s, consonants 0.05s
                duration = 0.08 if phoneme in ['AA', 'EH', 'IY', 'OW', 'UW'] else 0.05
                phonemes.append({
                    'phoneme': phoneme,
                    'start': time_offset,
                    'duration': duration
                })
                time_offset += duration
                last_phoneme = phoneme
        
        # Add slight pause between words
        phonemes.append({'phoneme': 'sil', 'start': time_offset, 'duration': 0.05})
        time_offset += 0.05
    
    # Scale durations to match realistic speech rate (150 words/min = 2.5 words/sec)
    if phonemes and len(words) > 0:
        # Calculate expected duration: ~0.4 seconds per word at 150 wpm
        expected_duration = len(words) * 0.4
        actual_duration = sum(p['duration'] for p in phonemes)
        
        if actual_duration > 0:
            scale_factor = expected_duration / actual_duration
            # Clamp scale factor to reasonable range (0.5x to 2x)
            scale_factor = max(0.5, min(2.0, scale_factor))
            
            for p in phonemes:
                p['duration'] *= scale_factor
            
            print(f"[PHONEME] Scaled {len(phonemes)} phonemes from {actual_duration:.2f}s to {expected_duration:.2f}s (factor: {scale_factor:.2f})")
    
    return phonemes

def save_generated_image(image_url, prompt_text=""):
    """
    Download and save an AI-generated image to the images folder.
    Returns the saved file path or None if failed.
    """
    try:
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize prompt for filename (first 50 chars, remove special chars)
        safe_prompt = "".join(c for c in prompt_text[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')
        
        if safe_prompt:
            filename = f"{timestamp}_{safe_prompt}.jpg"
        else:
            filename = f"{timestamp}.jpg"
        
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # Download image
        print(f"[IMAGE] Downloading: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Save to file
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"[IMAGE] ✓ Saved to: {filepath} ({len(response.content)} bytes)")
        return filepath
        
    except Exception as e:
        print(f"[IMAGE] Failed to save: {e}")
        return None

def speak(text, force=False, priority=False, is_chat_response=False):
    global last_comment_time, last_chat_response_time, tts_speaking, speech_queue, overlay_state, overlay_caption_text, overlay_last_update, current_phonemes, last_spoken_text

    # Entry log for debugging speak calls
    try:
        preview = (text[:120] + '...') if text and len(text) > 120 else text
        print(f"[SPEAK] Called (priority={priority} force={force} is_chat_response={is_chat_response}) -> {preview}")
    except Exception as _:
        print("[SPEAK] Called (preview failed to render)")

    if not text or not text.strip():
        print("[SPEAK] Empty text, skipping")
        return
    
    # Prevent duplicate audio playback - skip if same text was just spoken
    if hasattr(speak, 'last_audio_text') and text == speak.last_audio_text:
        print(f"⏭️ Skipped (duplicate audio): {text[:50]}...")
        return
    
    # Use speech_lock to ensure only ONE speak() runs at a time
    with speech_lock:
        # If already speaking, queue chat responses so they play after current speech.
        if tts_speaking:
            if priority or is_chat_response:
                speech_queue.append((text, force, priority, is_chat_response))
                tag = "priority" if priority else "chat"
                print(f"📋 Queued ({tag}): {text[:50]}...")
            return

        # Check cooldown - chat responses don't block game commentary
        if not force:
            if is_chat_response:
                # Chat responses have their own cooldown
                if time.time() - last_chat_response_time < config.min_gap:
                    return
            else:
                # Game commentary checks both timers (don't speak too soon after chat)
                time_since_comment = time.time() - last_comment_time
                time_since_chat = time.time() - last_chat_response_time
                min_wait = min(time_since_comment, time_since_chat)
                if min_wait < config.min_gap:
                    return

        print(f"🎙️ {text}")
        
        # Clean text for TTS FIRST - before setting overlay state
        import re
        clean_text = text.replace('*', '').replace('_', '').replace('~', '')
        # Fix "IT" being pronounced as "I-T" by converting standalone IT to lowercase
        clean_text = re.sub(r'\bIT\b', 'it', clean_text)
        # Fix name pronunciations - phonetic replacements
        clean_text = re.sub(r'\bplaix\b', 'plays', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\bPLAIX\b', 'PLAYS', clean_text)
        # Don't split words - only clean extra spaces between words
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Pre-generate phonemes but DON'T send to overlay yet (wait until audio plays)
        pending_phonemes = estimate_phonemes_from_text(clean_text)
        print(f"[PHONEME] Pre-generated {len(pending_phonemes)} phonemes for lip-sync")
        
        # DON'T update overlay yet - wait until phonemes are ready to send with caption
        overlay_state = "speaking"
        current_phonemes = []  # Clear phonemes until audio starts
        print(f"[OVERLAY] 🔇 Caption ready (will send with phonemes): {text[:50]}...")


        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            print(f"[SPEAK] Temp WAV path: {tmp_path}")

            # Pollinations TTS
            if hasattr(config, 'tts_engine') and config.tts_engine == 'pollinations':
                audio_data = pollinations_generate_audio(clean_text, config.pollinations_voice)
                if audio_data:
                    with open(tmp_path, 'wb') as f:
                        f.write(audio_data)
                else:
                    raise Exception("Pollinations TTS failed")
            # Edge TTS with phoneme extraction
            elif hasattr(config, 'tts_engine') and config.tts_engine == 'edge' and EDGE_TTS_AVAILABLE:
                import asyncio
                async def edge_tts_speak():
                    communicate = edge_tts.Communicate(clean_text, config.tts_speaker)
                    
                    # Extract phonemes from Edge TTS metadata
                    phoneme_list = []
                    audio_chunks = []
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_chunks.append(chunk["data"])
                        elif chunk["type"] == "WordBoundary":
                            # Edge TTS doesn't provide phonemes directly, we'll estimate from words
                            pass
                    
                    # Write audio
                    with open(tmp_path, 'wb') as f:
                        for audio_chunk in audio_chunks:
                            f.write(audio_chunk)
                
                asyncio.run(edge_tts_speak())
                print(f"[PHONEME] Edge TTS audio generated, using pre-generated {len(current_phonemes)} phonemes")
                
            # Kokoro TTS
            elif (not hasattr(config, 'tts_engine') or config.tts_engine == 'kokoro') and KOKORO_AVAILABLE and kokoro_engine:
                samples, sample_rate = kokoro_engine.create(clean_text, voice=config.tts_speaker, speed=1.2)
                sf.write(tmp_path, samples, sample_rate)
                print(f"[PHONEME] Kokoro audio generated, using pre-generated {len(current_phonemes)} phonemes")
                
            # XTTS fallback (disabled, raise error if no TTS engine)
            else:
                raise RuntimeError("No valid TTS engine available. Please enable Kokoro or Edge TTS.")

            sound = pygame.mixer.Sound(tmp_path)
            sound.set_volume(config.volume)
            try:
                filesize = os.path.getsize(tmp_path)
            except Exception:
                filesize = None
            print(f"[SPEAK] Audio file created: {tmp_path} ({filesize} bytes)")
            
            # Get actual audio duration and scale phonemes to match exactly
            actual_audio_duration = sound.get_length()  # in seconds
            print(f"[SPEAK] Audio duration (s): {actual_audio_duration:.3f}")
            phoneme_total_duration = sum(p['duration'] for p in pending_phonemes)
            
            if phoneme_total_duration > 0:
                scale_factor = actual_audio_duration / phoneme_total_duration
                
                # CRITICAL: Scale BOTH start times AND durations
                cumulative_time = 0.0
                for p in pending_phonemes:
                    p['start'] = cumulative_time  # Recalculate start from scaled durations
                    p['duration'] *= scale_factor
                    cumulative_time += p['duration']
                    
                print(f"[PHONEME] Scaled phonemes to match audio: {phoneme_total_duration:.2f}s -> {actual_audio_duration:.2f}s (factor: {scale_factor:.2f})")
                print(f"[PHONEME] First phoneme: start={pending_phonemes[0]['start']:.3f}s, Last: start={pending_phonemes[-1]['start']:.3f}s")

            # Set speaking flag BEFORE playing to prevent voice chat from listening
            tts_speaking = True
            
            # Calculate sentence timings for sequential display
            sentences = [s.strip() + '.' if not s.strip().endswith(('.', '!', '?')) else s.strip() 
                        for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            sentence_timings = []
            if sentences and pending_phonemes:
                # Distribute audio duration across sentences by character count
                total_chars = sum(len(s) for s in sentences)
                current_time = 0
                for sentence in sentences:
                    char_ratio = len(sentence) / total_chars if total_chars > 0 else 1.0 / len(sentences)
                    duration = actual_audio_duration * char_ratio
                    sentence_timings.append({
                        'text': sentence,
                        'start': current_time,
                        'duration': duration
                    })
                    current_time += duration
            
            # Send caption AND phonemes together RIGHT BEFORE audio plays
            global phonemes_sent
            overlay_caption_text = {
                "text": text, 
                "speaker": config.echo_personality or "Echo AI",
                "sentences": sentence_timings  # Include sentence timing
            }
            current_phonemes = pending_phonemes
            phonemes_sent = False  # Mark phonemes as new/unsent
            overlay_last_update = time.time()  # Update timestamp to trigger overlay poll
            print(f"[PHONEME] Prepared {len(current_phonemes)} phonemes for overlay (will send once)")
            print(f"[CAPTION] Split into {len(sentences)} sentences with timing")
            print(f"[OVERLAY] 📺 Caption & phonemes ready - sending NOW")
            
            # Small delay to ensure overlay receives both before audio starts
            time.sleep(0.05)
            
            # Mark this text as the current audio being played
            speak.last_audio_text = text
            
            sound.play()

            def wait_and_cleanup():
                global tts_speaking, speech_queue, overlay_state, overlay_caption_text, last_tts_finish_time, last_spoken_text, last_any_commentary_time, current_phonemes, phonemes_sent
                while pygame.mixer.get_busy():
                    time.sleep(0.1)
                time.sleep(0.2)
                tts_speaking = False
                last_tts_finish_time = time.time()  # Track when TTS finished for cooldown
                last_spoken_text = text.lower().strip()  # Store for feedback filtering
                last_any_commentary_time = time.time()  # Track ANY audio playback
                
                # Reset overlay state and clear phonemes
                overlay_state = "idle"
                overlay_caption_text = {"text": "", "speaker": config.echo_personality or "Echo AI"}
                current_phonemes = []  # Clear phonemes when done speaking
                phonemes_sent = False  # Reset flag
                speak.last_audio_text = None  # Clear duplicate protection after playback finishes
                print("[PHONEME] Cleared phonemes (speech finished)")
                
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                # Process queue after finishing (allow chat responses or priority items)
                with speech_lock:
                    if speech_queue:
                        # Find the next item that should be auto-played (priority OR chat responses OR forced)
                        next_to_play = None
                        while speech_queue:
                            next_item = speech_queue.pop(0)
                            next_text, next_force, next_priority, next_is_chat = next_item if len(next_item) == 4 else (*next_item, False)
                            if next_priority or next_is_chat or next_force:
                                next_to_play = (next_text, next_force, next_priority, next_is_chat)
                                break
                            else:
                                # Drop non-priority commentary items to avoid backlog
                                print("[SPEAK] Dropping non-priority queued item to prevent backlog")
                        if next_to_play:
                            nt_text, nt_force, nt_priority, nt_is_chat = next_to_play
                            threading.Thread(target=lambda: speak(nt_text, nt_force, nt_priority, nt_is_chat), daemon=True).start()

            threading.Thread(target=wait_and_cleanup, daemon=True).start()

        except Exception as e:
            print(f"[TTS ERROR] {e}")
            tts_speaking = False

        # Update appropriate timer based on speech type
        if not force:
            if is_chat_response:
                last_chat_response_time = time.time()
                # Don't update last_comment_time for chat - allows game commentary to continue
            else:
                last_comment_time = time.time()

# ============================================================
#                   INTERACTIVE CONTENT BOX HELPERS
# ============================================================
def show_image(image_url, title="🖼️ Image"):
    """Display an image in the content box"""
    global content_box_data, content_box_sent
    content_box_data = {
        'action': 'show_image',
        'url': image_url,
        'title': title,
        'html': None
    }
    content_box_sent = False  # Reset flag so overlay will detect it
    print(f"[CONTENT BOX] Displaying image: {image_url}")

def show_video(video_url, title="🎥 Video"):
    """Display a video in the content box"""
    global content_box_data, content_box_sent
    content_box_data = {
        'action': 'show_video',
        'url': video_url,
        'title': title,
        'html': None
    }
    content_box_sent = False  # Reset flag so overlay will detect it
    print(f"[CONTENT BOX] Displaying video: {video_url}")

def show_webpage(url, title="🌐 Webpage"):
    """Display a webpage in the content box"""
    global content_box_data, content_box_sent
    content_box_data = {
        'action': 'show_webpage',
        'url': url,
        'title': title,
        'html': None
    }
    content_box_sent = False  # Reset flag so overlay will detect it
    print(f"[CONTENT BOX] Displaying webpage: {url}")

def show_external_url(url, title="🌐 Opening..."):
    """Open a URL in a new Electron window (for sites that block iframes like YouTube)"""
    global content_box_data, content_box_sent
    content_box_data = {
        'action': 'open_window',
        'url': url,
        'title': title,
        'html': None
    }
    content_box_sent = False  # Reset flag so overlay will detect it
    print(f"[CONTENT BOX] Opening in new window: {url}")

def show_html(html_content, title="📄 Content"):
    """Display custom HTML in the content box"""
    global content_box_data, content_box_sent
    content_box_data = {
        'action': 'show_html',
        'url': None,
        'title': title,
        'html': html_content
    }
    content_box_sent = False  # Reset flag so overlay will detect it
    print(f"[CONTENT BOX] Displaying HTML content")

def hide_content_box():
    """Hide the content box"""
    global content_box_data
    content_box_data = {
        'action': 'hide',
        'url': None,
        'title': None,
        'html': None
    }
    print(f"[CONTENT BOX] Hiding")

# ============================================================
#                   TTS ENGINE SELECTION (UI/CONFIG)
# ============================================================

def set_tts_engine(engine):
    config.tts_engine = engine
    if engine == 'edge':
        voices = get_edge_voices()
    elif engine == 'kokoro':
        voices = KOKORO_VOICES
    elif engine == 'pollinations':
        voices = POLLINATIONS_VOICES
    else:
        voices = []
    if voices:
        if engine == 'pollinations':
            config.pollinations_voice = voices[0]
        else:
            config.tts_speaker = voices[0]
    return voices

# ============================================================
#                   SAFE YOLO INFERENCE WRAPPER
# ============================================================

def safe_yolo_infer(frame):
    global device, CUDA_DISABLED, model

    try:
        torch.cuda.synchronize()
        with torch.no_grad():
            results = model.predict(
                frame,
                conf=0.25,
                iou=0.5,
                verbose=False,
                imgsz=640,
                device=device
            )[0]

    except Exception as e:
        print(f"\n[ERROR] YOLO GPU inference error: {e}")
        print("➡ Switching to CPU permanently.")
        disable_cuda_permanently()

        try:
            model.to("cpu")
        except:
            pass

        device = "cpu"
        return []

    try:
        if not hasattr(results, "boxes"):
            return []

        if results.boxes is None or len(results.boxes) == 0:
            return []

        return results.boxes.cpu().numpy()

    except Exception as e:
        print(f"[ERROR] YOLO box extraction failure: {e}")
        return []

# ============================================================
#                    BROADCAST WORKER
# ============================================================

def broadcast_worker(debug_mode):
    global processing, GAME_STATE, ball_activity_counter, last_comment_time, last_chat_response_time
    global last_goal_time, last_shot_time, last_possession_time, goals_scored, ball_in_goal_prev, state_change_counter
    global in_replay, replay_cooldown_time, last_activity_score
    global echo_ai  # Make echo_ai accessible to API handlers
    global soccer_goalkeeper_positions, soccer_last_ball_position, soccer_ball_missing_frames

    config.enable_debug = debug_mode
    # Ensure config values are correct types
    try:
        config.trail_length = int(config.trail_length or 30)
    except (TypeError, ValueError):
        config.trail_length = 30
        print("[WARNING] Invalid trail_length in config, using default 30")
    ball_trail.clear()
    ball_in_goal_prev = False
    last_comment_time = time.time() - 10
    last_goal_time = time.time() - 10
    last_shot_time = time.time() - 10
    last_possession_time = time.time() - 10
    GAME_STATE = "menu"
    ball_activity_counter = 0
    state_change_counter = 0
    goals_scored = 0
    in_replay = False
    replay_cooldown_time = 0
    last_activity_score = 0

    # Initialize AI provider
    if config.ai_provider == 'pollinations':
        ai = PollinationsAI(
            model=config.pollinations_model,
            voice=config.pollinations_voice,
            max_tokens=config.echo_max_tokens
        )
        echo_ai = ai  # Store in global variable for API access
        print(f"🤖 Using Pollinations AI ({config.pollinations_model})")
    else:
        # Use custom character prompt if provided
        echo_personality = config.echo_personality
        game_mode = getattr(config, 'game_mode', 'rocket_league')
        if config.custom_character_prompt:
            ai = EchoAI(
                enabled=config.use_echo_ai,
                temperature=0.7,
                max_tokens=config.echo_max_tokens,
                ollama_model=config.ollama_model,
                personality="Custom"
            )
            ai.system_prompt = config.custom_character_prompt
        else:
            ai = EchoAI(
                enabled=config.use_echo_ai,
                temperature=0.7,
                max_tokens=config.echo_max_tokens,
                ollama_model=config.ollama_model,
                personality=config.echo_personality
            )
            # Set hockey personality prompt if in hockey mode
            if game_mode == "hockey":
                ai.set_personality(config.echo_personality, game_mode="hockey")
        
        # Store in global variable for API access
        echo_ai = ai
        print(f"🤖 Using EchoAI ({config.ollama_model}, {config.echo_personality}, mode={game_mode})")

    # speak("PLAIX online! Commentary system ready!")  # Disabled to prevent errors

    sct = mss.mss()
    print("Available monitors:", sct.monitors)

    monitor_index = config.monitor_index
    try:
        monitor = sct.monitors[monitor_index]
        print(f"Using monitor[{monitor_index}]: {monitor}")
    except IndexError:
        monitor = sct.monitors[1]
        print(f"Invalid index, falling back to monitor[1]: {monitor}")
        config.monitor_index = 1

    frame_count = 0
    start_time = time.time()
    print("🚀 Broadcast started!")

    while processing:
        try:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            h, w = frame.shape[:2]

            boxes = safe_yolo_infer(frame)
            
            # Create visualization frame for overlay if tracking enabled
            overlay_frame = frame.copy() if config.show_tracking else None
            
            # Update live preview frame (downsample for performance)
            global live_preview_frame, clip_buffer, overlay_tracking_data, ball_trail_points
            preview_h, preview_w = 360, 640
            live_preview_frame = cv2.resize(frame, (preview_w, preview_h))
            
            # Store frames in buffer for highlight clips (keep last 20 seconds)
            if config.enable_highlights:
                clip_buffer.append(frame.copy())

            ball = None
            ball_box = None
            goalposts = {"my": None, "enemy": None}
            players = []
            current_time = time.time()
            detections = []  # For tracking overlay

            # Set class mapping based on game mode
            if getattr(config, 'game_mode', 'rocket_league') == 'hockey' or (model and hasattr(model, 'names') and 'PUCK' in [str(n).upper() for n in model.names]):
                PUCK, GOAL, GOALIE, PLAYER = 5, 2, 3, 4
                BALL, ENEMY, ENEMY_GOALPOST, MY_GOALPOST, TEAMMATE = PUCK, GOALIE, GOAL, GOAL, PLAYER
            elif getattr(config, 'game_mode', 'rocket_league') == 'soccer' or (model and hasattr(model, 'names') and 'GOALKEEPER' in [str(n).upper() for n in model.names]):
                # Soccer model: 0=ball, 1=goalkeeper, 2=player, 3=referee
                # Map: BALL=0, ENEMY=1 (goalkeeper), TEAMMATE=2 (player), no goalposts in this model
                BALL, ENEMY, ENEMY_GOALPOST, MY_GOALPOST, TEAMMATE = 0, 1, -1, -1, 2
            else:
                BALL, ENEMY, ENEMY_GOALPOST, MY_GOALPOST, TEAMMATE = 0, 1, 2, 3, 4

            for b in boxes:
                try:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                except Exception as e:
                    print(f"[WARNING] Invalid box data: {e}, skipping")
                    continue

                # Hockey mode class mapping with OCR and roster
                if getattr(config, 'game_mode', 'rocket_league') == 'hockey':
                    if cls_id == PUCK and conf > 0.30:
                        bw, bh = x2 - x1, y2 - y1
                        area = bw * bh
                        aspect = bw / (bh + 1e-5)
                        if (config.aspect_ratio_min < aspect < config.aspect_ratio_max and 
                            config.min_ball_area < area < w * h * config.max_ball_pct):
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            ball = center
                            ball_box = (x1, y1, x2, y2)
                            ball_trail.append((center[0], center[1], current_time))
                            if config.enable_ball_trail and isinstance(center, (tuple, list)) and len(center) == 2 and all(isinstance(x, (int, float)) for x in center):
                                ball_trail_points.append(center)
                            if config.ball_color_override and overlay_frame is not None:
                                ball_region = frame[y1:y2, x1:x2].copy()
                                color_overlay = np.zeros_like(ball_region)
                                color_overlay[:, :] = config.ball_color_override[::-1]
                                frame[y1:y2, x1:x2] = cv2.addWeighted(ball_region, 0.4, color_overlay, 0.6, 0)
                            detections.append({"type": "puck", "box": (x1, y1, x2, y2), "conf": conf})
                    elif cls_id == GOAL and conf > 0.5:
                        goalposts["my"] = (x1 + x2) // 2
                        detections.append({"type": "goal", "box": (x1, y1, x2, y2), "conf": conf})
                    elif cls_id == GOALIE and conf > 0.5:
                        detections.append({"type": "goalie", "box": (x1, y1, x2, y2), "conf": conf})
                    elif cls_id == PLAYER and conf > 0.55:
                        # Use YOLO track_id if available for OCR mapping
                        track_id = int(b.id[0]) if hasattr(b, 'id') and b.id is not None else None
                        player_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        players.append((track_id, player_center[0], player_center[1]))
                        detections.append({"type": "player", "box": (x1, y1, x2, y2), "conf": conf})
                        # OCR jersey number if not already mapped
                        if EASY_OCR_AVAILABLE and track_id is not None and track_id not in jersey_map:
                            crop = frame[y1:y2, x1:x2]
                            ocr_queue.put((track_id, crop))
                            print(f"[OCR] Queued for track_id={track_id}")
                        # Draw player/goalie name above box if available
                        name = None
                        if track_id is not None and track_id in jersey_map:
                            info = jersey_map[track_id]
                            team = info.get("team", "home")
                            num = info.get("num", "97")
                            name = rosters[team].get(num, None)
                            print(f"[NAME] track_id={track_id}, num={num}, team={team}, name={name}")
                        label = name if name else ("Goalie" if cls_id == GOALIE else "Player")
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
                else:
                    if cls_id == BALL and conf > 0.30:
                        bw, bh = x2 - x1, y2 - y1
                        area = bw * bh
                        aspect = bw / (bh + 1e-5)
                        if (config.aspect_ratio_min < aspect < config.aspect_ratio_max and 
                            config.min_ball_area < area < w * h * config.max_ball_pct):
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            ball = center
                            ball_box = (x1, y1, x2, y2)
                            ball_trail.append((center[0], center[1], current_time))
                            # Add to trail points for overlay
                            if config.enable_ball_trail and isinstance(center, (tuple, list)) and len(center) == 2 and all(isinstance(x, (int, float)) for x in center):
                                ball_trail_points.append(center)
                            detections.append({"type": "ball", "box": (x1, y1, x2, y2), "conf": conf})

                    elif cls_id == MY_GOALPOST and conf > 0.5:
                        goalposts["my"] = (x1 + x2) // 2
                        detections.append({"type": "my_goal", "box": (x1, y1, x2, y2), "conf": conf})
                    elif cls_id == ENEMY_GOALPOST and conf > 0.5:
                        # --- Smoothing for enemy goal position ---
                        detected_x = (x1 + x2) // 2
                        global smoothed_enemy_goal_x, enemy_goal_smooth_alpha, enemy_goal_last_detected, enemy_goal_confidence, enemy_goal_confidence_max
                        now = time.time()
                        if smoothed_enemy_goal_x is None:
                            smoothed_enemy_goal_x = detected_x
                            enemy_goal_confidence = 1
                        else:
                            # If the detected position is not a huge jump, smooth it
                            if abs(detected_x - smoothed_enemy_goal_x) < 120:
                                smoothed_enemy_goal_x = int(enemy_goal_smooth_alpha * detected_x + (1 - enemy_goal_smooth_alpha) * smoothed_enemy_goal_x)
                                enemy_goal_confidence = min(enemy_goal_confidence + 1, enemy_goal_confidence_max)
                            else:
                                # If it's a big jump, only update if confidence is low
                                if enemy_goal_confidence < 3:
                                    smoothed_enemy_goal_x = detected_x
                                    enemy_goal_confidence = 1
                                # else, ignore this frame's detection
                        enemy_goal_last_detected = now
                        goalposts["enemy"] = smoothed_enemy_goal_x
                        detections.append({"type": "enemy_goal", "box": (x1, y1, x2, y2), "conf": conf, "smoothed_x": smoothed_enemy_goal_x})
                    
                    elif cls_id in [ENEMY, TEAMMATE] and conf > 0.55:
                        # For soccer mode, use position-based team detection
                        if getattr(config, 'game_mode', 'rocket_league') == 'soccer':
                            player_x = (x1 + x2) // 2
                            # Determine team by field position
                            # Left half = Team A (enemy), Right half = Team B (teammate)
                            # This assumes camera is on the right team's side
                            is_left_team = player_x < w * 0.5
                            
                            # Goalkeeper (cls_id == 1): Left keeper = enemy, Right keeper = teammate
                            # Player (cls_id == 2): Use position to determine team
                            if cls_id == 1:  # Goalkeeper
                                actual_team = ENEMY if is_left_team else TEAMMATE
                            else:  # Player (cls_id == 2)
                                actual_team = ENEMY if is_left_team else TEAMMATE
                            
                            players.append((actual_team, player_x, (y1 + y2) // 2))
                            player_type = "enemy" if actual_team == ENEMY else "teammate"
                            detections.append({"type": player_type, "box": (x1, y1, x2, y2), "conf": conf})
                        else:
                            # Rocket League / Hockey: use class_id directly
                            players.append((cls_id, (x1 + x2) // 2, (y1 + y2) // 2))
                            player_type = "enemy" if cls_id == ENEMY else "teammate"
                            detections.append({"type": player_type, "box": (x1, y1, x2, y2), "conf": conf})

                # If no enemy goal detected this frame, decay confidence and keep last smoothed position
            # After all boxes processed, if enemy goal not detected this frame, decay confidence
            if 'smoothed_enemy_goal_x' in globals() and smoothed_enemy_goal_x is not None:
                if goalposts["enemy"] is None:
                    # If not detected for 0.5s, confidence drops
                    if time.time() - enemy_goal_last_detected > 0.5:
                        enemy_goal_confidence = max(enemy_goal_confidence - 1, 0)
                        # If confidence still positive, keep using last smoothed position
                        if enemy_goal_confidence > 0:
                            goalposts["enemy"] = smoothed_enemy_goal_x
            
            # Draw puck trail and effects (comet tail, glow, impact burst)
            if config.enable_ball_trail and len(ball_trail_points) > 2:
                # Validate trail points to prevent corrupted data errors
                if not all(isinstance(pt, (tuple, list)) and len(pt) >= 2 and all(isinstance(x, (int, float)) for x in pt[:2]) for pt in ball_trail_points):
                    print("[WARNING] Ball trail points corrupted, clearing trail")
                    ball_trail_points.clear()
                    trail_list = []
                else:
                    trail_list = list(ball_trail_points)
                if len(trail_list) > 2:
                    # Filter out large jumps (false positives)
                    filtered_trail = [trail_list[0]]
                    for pt in trail_list[1:]:
                        if isinstance(pt, (tuple, list)) and len(pt) >= 2 and all(isinstance(x, (int, float)) for x in pt[:2]):
                            if np.linalg.norm(np.array(pt[:2]) - np.array(filtered_trail[-1][:2])) < 120:
                                filtered_trail.append(pt)
                num_points = min(int(config.trail_length), len(filtered_trail))

                style = getattr(config, 'trail_style', 'glow')

                for i in range(1, num_points):
                    idx = len(filtered_trail) - num_points + i
                    if idx < 1 or idx >= len(filtered_trail):
                        continue
                    pt1 = filtered_trail[idx - 1][:2]
                    pt2 = filtered_trail[idx][:2]
                    if not isinstance(pt1, (tuple, list)) or len(pt1) != 2 or not isinstance(pt2, (tuple, list)) or len(pt2) != 2:
                        continue
                    if not all(isinstance(x, (int, float)) for x in pt1) or not all(isinstance(x, (int, float)) for x in pt2):
                        continue
                    alpha = i / num_points

                    if style == "glow":
                        # Glow layers (outer to inner)
                        for g in range(6, 0, -2):
                            glow_alpha = alpha * (g/6) * 0.5
                            glow_color = (0, int(180 * glow_alpha), int(255 * glow_alpha))
                            thickness = max(1, int(24 * glow_alpha))
                            cv2.line(frame, pt1, pt2, glow_color, thickness, cv2.LINE_AA)
                        thickness = max(1, int(16 * alpha))
                        color = (0, int(255 * alpha), 255)
                        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
                    elif style == "solid":
                        thickness = max(2, int(10 * alpha))
                        color = (0, 200, 255)
                        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
                    elif style == "fade":
                        thickness = max(2, int(10 * alpha))
                        color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
                    elif style == "rainbow":
                        thickness = max(2, int(10 * alpha))
                        hue = int(180 * alpha)
                        color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue,255,255]]]), cv2.COLOR_HSV2BGR)[0,0])
                        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
                    elif style == "comet":
                        # Comet: white-hot head, fading blue tail
                        thickness = int(18 * alpha)
                        color = (int(255 * (1 - alpha)), int(255 * alpha), 255)
                        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
                        if i == num_points - 1:
                            # Head glow
                            px, py = pt2
                            for r in range(24, 4, -4):
                                cv2.circle(frame, (px, py), r, (255, 255, 255, 40), -1)
                            cv2.circle(frame, (px, py), 8, (255, 255, 255), -1)
                    elif style == "neon":
                        # Neon: bright magenta/cyan, sharp edges
                        thickness = max(2, int(12 * alpha))
                        color = (255, 0, int(255 * alpha))
                        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
                        if i % 3 == 0:
                            # Neon pulse
                            cv2.circle(frame, pt2, 6, (0,255,255), 2)

                # Draw puck glow at head (for all styles)
                if filtered_trail:
                    head = filtered_trail[-1]
                    if isinstance(head, (tuple, list)) and len(head) >= 2:
                        px, py = head[:2]
                        if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                            for r in range(30, 5, -5):
                                cv2.circle(frame, (px, py), r, (0, 255, 255, 60), -1)
                            cv2.circle(frame, (px, py), 8, (0, 255, 255), -1)
                            cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)

                # Impact burst on goal/shot
                def draw_burst(img, center, color=(255,255,0), radius=60, rays=12):
                    if not isinstance(center, (tuple, list)) or len(center) != 2:
                        return
                    if not all(isinstance(x, (int, float)) for x in center):
                        return
                    for i in range(rays):
                        angle = 2 * np.pi * i / rays
                        x2 = int(center[0] + radius * np.cos(angle))
                        y2 = int(center[1] + radius * np.sin(angle))
                        cv2.line(img, center, (x2, y2), color, 4)

                # Show burst for 0.5s after goal/shot
                if 'burst_time' not in globals():
                    burst_time = 0
                    burst_pos = None
                    burst_color = (255,255,0)
                if ball and time.time() - last_goal_time < 0.5:
                    if filtered_trail and isinstance(filtered_trail[-1], (tuple, list)) and len(filtered_trail[-1]) >= 2:
                        burst_pos = filtered_trail[-1][:2]
                        burst_color = (0,255,255)
                        draw_burst(frame, burst_pos, burst_color, 80, 16)
                if ball and shot_speed_mph > 70 and time.time() - last_shot_time < 0.5:
                    if filtered_trail and isinstance(filtered_trail[-1], (tuple, list)) and len(filtered_trail[-1]) >= 2:
                        burst_pos = filtered_trail[-1][:2]
                        burst_color = (255,0,255)
                        draw_burst(frame, burst_pos, burst_color, 50, 10)
            
            # Update overlay tracking data
            try:
                trail_points = list(ball_trail_points)[-int(config.trail_length):] if config.enable_ball_trail else []
            except Exception as e:
                print(f"[WARNING] Error getting trail points: {e}, using empty")
                trail_points = []
            overlay_tracking_data = {
                "detections": detections,
                "ball_position": ball,
                "trail_points": trail_points,
                "resolution": (w, h)
            }
            
            activity_score = 0
            if ball: activity_score += 4
            if goalposts["my"] or goalposts["enemy"]: activity_score += 3
            if len(players) >= 2: activity_score += 2

            # Update game state with hysteresis - just require ball for in-game
            # (goalposts/players detection is too unreliable)
            if ball:
                state_change_counter += 1
                if state_change_counter >= 5:  # 5 consecutive frames with ball
                    GAME_STATE = "in-game"
                    state_change_counter = 15  # Cap it
            else:
                state_change_counter -= 1
                if state_change_counter <= -10:  # 10 frames without ball
                    GAME_STATE = "menu"
                    state_change_counter = -15  # Cap it

            # Detect replay: DISABLED - was too sensitive and blocking goals
            # Will re-implement with better detection after goals work properly
            # if GAME_STATE == "in-game" and last_activity_score >= 9 and activity_score < 6 and not in_replay:
            #     in_replay = True
            #     replay_cooldown_time = time.time() + 8.0
            #     print("[REPLAY] Detected replay start - pausing goal announcements")
            
            # if in_replay and time.time() > replay_cooldown_time:
            #     in_replay = False
            #     print("[REPLAY] Replay ended - resuming goal detection")
            
            last_activity_score = activity_score

            ball_activity_counter += activity_score
            ball_activity_counter = max(0, ball_activity_counter - 1.5)
            shot_speed_mph = 0
                        # Ensure shot_speed_mph is always defined, even if the above block is skipped
                        # (prevents UnboundLocalError in later code)
            if GAME_STATE == "in-game" and ball and len(ball_trail) >= 10:
                dx = ball_trail[-1][0] - ball_trail[-10][0]
                dy = ball_trail[-1][1] - ball_trail[-10][1]
                dt = max(ball_trail[-1][2] - ball_trail[-10][2], 0.01)
                pixel_speed = (dx**2 + dy**2)**0.5 / dt  # pixels per second

                # Use manual config first if provided
                global pixel_to_uu, pixel_to_uu_last_update
                if getattr(config, 'uu_per_pixel', None):
                    pixel_to_uu = config.uu_per_pixel
                else:
                    # Auto-estimate scale when both goal centers are detected: goal width ≈ 880 uu
                    try:
                        if goalposts.get('my') is not None and goalposts.get('enemy') is not None:
                            measured_px = abs(goalposts['enemy'] - goalposts['my'])
                            if measured_px > 10:
                                estimated = 880.0 / float(measured_px)  # uu per pixel
                                if pixel_to_uu is None:
                                    pixel_to_uu = estimated
                                else:
                                    # EWMA smoothing to avoid jitter
                                    pixel_to_uu = pixel_to_uu * (1.0 - pixel_to_uu_alpha) + estimated * pixel_to_uu_alpha
                                pixel_to_uu_last_update = time.time()
                                if config.enable_debug and frame_count % 120 == 0:
                                    print(f"[CALIB] Auto-estimated pixel_to_uu={pixel_to_uu:.6f} uu/px from goal separation {measured_px:.1f}px")
                    except Exception as e:
                        if config.enable_debug:
                            print(f"[CALIB] Failed to auto-estimate scale: {e}")

                # If we have a scale, convert pixel speed -> uu/s -> m/s -> km/h -> mph
                if pixel_to_uu and pixel_to_uu > 0:
                    uu_per_s = pixel_speed * pixel_to_uu  # uu per second
                    kmh = uu_per_s * 0.036  # uu/s * 0.036 -> km/h
                    mph = kmh * 0.621371
                    shot_speed_mph = mph
                    shot_speed_kmh = kmh
                else:
                    # Fallback heuristic (old behavior scaled slightly)
                    shot_speed_mph = (pixel_speed / w) * 60
                    shot_speed_kmh = shot_speed_mph * 1.60934

                # Debug logging occasionally
                if frame_count % 120 == 0 and config.enable_debug:
                    print(f"  [DEBUG] pixel_speed={pixel_speed:.1f}px/s, pixel_to_uu={pixel_to_uu}, km/h={shot_speed_kmh:.1f}, mph={shot_speed_mph:.1f}")

                near_goal = ball[0] < w * 0.2 or ball[0] > w * 0.8
                
                # Debug: Show speed calculation every 120 frames
                if frame_count % 120 == 0:
                    print(f"  [DEBUG] Ball speed: {shot_speed_mph:.1f} mph | near_goal={near_goal} | threshold={config.shot_speed_threshold}")
                # Only trigger shot commentary in-game, not in menu
                # Also skip if ball is stationary (kickoff position)
                if GAME_STATE == "in-game" and not ball_is_stationary:
                    # Hockey-specific: require higher speed for shot commentary, use 'puck' not 'ball'
                    is_hockey = getattr(config, 'game_mode', 'rocket_league') == 'hockey'
                    min_hockey_speed = 30  # Minimum mph for hockey shot commentary
                    if is_hockey:
                        if shot_speed_mph < min_hockey_speed:
                            pass  # Do not commentate on slow shots in hockey
                        elif (shot_speed_mph > config.shot_speed_threshold and near_goal and 
                              time.time() - last_shot_time > config.shot_cooldown and not in_replay):
                            # Use hockey shot lines and 'puck'
                            print(f"[SHOT DETECTED] Hockey shot at {shot_speed_mph:.1f} mph")
                            # speak(random.choice(shot_lines).replace('shot', 'puck').replace('ball', 'puck'))  # Disabled to prevent errors
                            if config.enable_highlights and config.auto_save_shots and shot_speed_mph >= 60:
                                threading.Thread(target=lambda: save_highlight_clip("epic_shot", config.clip_duration), daemon=True).start()
                            if config.use_echo_ai:
                                captured_speed = int(shot_speed_mph)
                                def async_shot():
                                    time.sleep(1.8)
                                    print(f"[AI] 🧠 Generating shot commentary for {captured_speed} mph shot...")
                                    with ollama_lock:
                                        sport = "Hockey"
                                        if hasattr(ai, "shot_on_net"):
                                            bonus = ai.shot_on_net(captured_speed, config.your_player_name or "the attacker", sport)
                                        else:
                                            bonus = f"{config.your_player_name or 'The attacker'} {random.choice(shot_lines).replace('shot', 'puck').replace('ball', 'puck')} ({captured_speed:.1f} mph)"
                                    if bonus:
                                        print(f"[AI] 📣 Shot commentary: {bonus}")
                                        try:
                                            speak(bonus, priority=True)
                                        except Exception as e:
                                            print(f"[SPEAK ERROR] {e}")
                                    else:
                                        print(f"[AI] ⚠️ No shot commentary generated")
                                threading.Thread(target=async_shot, daemon=True).start()
                            last_shot_time = time.time()
                    else:
                        if (shot_speed_mph > config.shot_speed_threshold and near_goal and 
                            time.time() - last_shot_time > config.shot_cooldown and not in_replay):
                            print(f"[SHOT DETECTED] Rocket League shot at {shot_speed_mph:.1f} mph")
                            # speak(random.choice(shot_lines))  # Disabled to prevent errors
                            if config.enable_highlights and config.auto_save_shots and shot_speed_mph >= 60:
                                threading.Thread(target=lambda: save_highlight_clip("epic_shot", config.clip_duration), daemon=True).start()
                            if config.use_echo_ai:
                                captured_speed = int(shot_speed_mph)
                                def async_shot():
                                    time.sleep(1.8)
                                    print(f"[AI] 🧠 Generating shot commentary for {captured_speed} mph shot...")
                                    with ollama_lock:
                                        if hasattr(ai, "shot_on_net"):
                                            bonus = ai.shot_on_net(captured_speed, config.your_player_name or "the attacker")
                                        else:
                                            bonus = f"{config.your_player_name or 'The attacker'} {random.choice(shot_lines)} ({captured_speed:.1f} mph)"
                                    if bonus:
                                        print(f"[AI] 📣 Shot commentary: {bonus}")
                                        try:
                                            speak(bonus, priority=True)
                                        except Exception as e:
                                            print(f"[SPEAK ERROR] {e}")
                                    else:
                                        print(f"[AI] ⚠️ No shot commentary generated")
                                threading.Thread(target=async_shot, daemon=True).start()
                            last_shot_time = time.time()

            ball_in_goal = False
            scoring_side = None
            direction_toward = False

            # Goal detection - always run to maintain state properly
            if GAME_STATE == "in-game" and ball and (goalposts["my"] or goalposts["enemy"]):
                # VERY tight goal detection - only trigger when ball is RIGHT AT the goalpost
                goal_width = w * 0.05  # Very narrow - only when ball is at the post (was 0.15)
                goal_height_min = 0.30 * h  # Tighter vertical bounds (was 0.22)
                goal_height_max = 0.70 * h  # Tighter vertical bounds (was 0.78)
                
                # Check if ball is in goal zone (only check goalposts that exist)
                if goalposts["enemy"] and abs(ball[0] - goalposts["enemy"]) < goal_width and goal_height_min < ball[1] < goal_height_max:
                    ball_in_goal = True
                    scoring_side = "enemy"
                elif goalposts["my"] and abs(ball[0] - goalposts["my"]) < goal_width and goal_height_min < ball[1] < goal_height_max:
                    ball_in_goal = True
                    scoring_side = "my"

                # Require longer trail and stronger directional movement
                if ball_in_goal and len(ball_trail) >= 15 and scoring_side:
                    prev_x = ball_trail[-15][0]  # Look back further (was -7)
                    movement_threshold = w * 0.05  # Require significant movement toward goal
                    
                    if scoring_side == "enemy" and goalposts["enemy"]:
                        if goalposts["enemy"] > w//2:
                            direction_toward = (ball[0] - prev_x) > movement_threshold
                        else:
                            direction_toward = (prev_x - ball[0]) > movement_threshold
                    elif scoring_side == "my" and goalposts["my"]:
                        if goalposts["my"] > w//2:
                            direction_toward = (ball[0] - prev_x) > movement_threshold
                        else:
                            direction_toward = (prev_x - ball[0]) > movement_threshold

                # Stricter conditions: higher speed requirement, confirmed direction, cooldown
                if (not ball_in_goal_prev and ball_in_goal and direction_toward and shot_speed_mph > 12 and
                    time.time() - last_shot_time > config.shot_cooldown and not in_replay):
                    # Save highlight clip for epic shots (60+ mph)
                    if config.enable_highlights and config.auto_save_shots and shot_speed_mph >= 60:
                        threading.Thread(target=lambda: save_highlight_clip("epic_shot", config.clip_duration), daemon=True).start()
                    if config.use_echo_ai:
                        captured_speed = int(shot_speed_mph)
                        def async_shot():
                            time.sleep(1.8)
                            print(f"[AI] \U0001f9e0 Generating shot commentary for {captured_speed} mph shot...")
                            with ollama_lock:
                                sport = "Hockey" if getattr(config, 'game_mode', 'rocket_league') == 'hockey' else "Rocket League"
                                bonus = ai.shot_on_net(captured_speed, config.your_player_name or "the attacker")
                            if bonus:
                                print(f"[AI] \U0001f4e3 Shot commentary: {bonus}")
                                try:
                                    speak(bonus, priority=False)
                                except Exception as e:
                                    print(f"[SPEAK ERROR] {e}")
                            else:
                                print(f"[AI] \u26a0\ufe0f No shot commentary generated")
                        threading.Thread(target=async_shot, daemon=True).start()
                    else:
                        # speak(random.choice(shot_lines))  # Disabled to prevent errors
                        pass
                    last_shot_time = time.time()
            ball_in_goal_prev = ball_in_goal

            # ============ SOCCER MODE: HYBRID GOAL DETECTION ============
            if getattr(config, 'game_mode', 'rocket_league') == 'soccer' and GAME_STATE == "in-game":
                global soccer_goalkeeper_positions, soccer_last_ball_position, soccer_ball_missing_frames
                
                # Track goalkeeper positions (left/right zones)
                goalkeepers = [p for p in players if p[0] == ENEMY]  # Goalkeeper class
                if len(goalkeepers) >= 1:
                    # Determine which goalkeeper is on left vs right
                    for gk in goalkeepers:
                        gk_x = gk[1]
                        if gk_x < w * 0.3:  # Left goalkeeper
                            soccer_goalkeeper_positions["left"] = gk_x
                        elif gk_x > w * 0.7:  # Right goalkeeper
                            soccer_goalkeeper_positions["right"] = gk_x
                
                if ball:
                    ball_x, ball_y = ball
                    
                    # Method 1: Fixed goal zones (left/right 15% of screen)
                    in_left_goal_zone = ball_x < w * 0.15
                    in_right_goal_zone = ball_x > w * 0.85
                    in_goal_zone = in_left_goal_zone or in_right_goal_zone
                    
                    # Method 2: Ball passed behind goalkeeper
                    passed_left_keeper = False
                    passed_right_keeper = False
                    if soccer_goalkeeper_positions["left"] and ball_x < soccer_goalkeeper_positions["left"] - 30:
                        passed_left_keeper = True
                    if soccer_goalkeeper_positions["right"] and ball_x > soccer_goalkeeper_positions["right"] + 30:
                        passed_right_keeper = True
                    
                    # Method 3: Ball velocity check
                    high_velocity = shot_speed_mph > 8  # Lower threshold for soccer
                    
                    # Method 4: Ball trajectory toward goal
                    moving_toward_goal = False
                    if len(ball_trail) >= 10:
                        prev_x = ball_trail[-10][0]
                        if in_left_goal_zone and (prev_x - ball_x) > w * 0.03:
                            moving_toward_goal = True
                        elif in_right_goal_zone and (ball_x - prev_x) > w * 0.03:
                            moving_toward_goal = True
                    
                    # Hybrid detection: Combine multiple signals
                    goal_confidence = 0
                    if in_goal_zone: goal_confidence += 2
                    if passed_left_keeper or passed_right_keeper: goal_confidence += 3
                    if high_velocity: goal_confidence += 1
                    if moving_toward_goal: goal_confidence += 2
                    
                    # Trigger goal if confidence is high enough
                    if (goal_confidence >= 5 and not ball_in_goal_prev and 
                        time.time() - last_shot_time > config.shot_cooldown and not in_replay):
                        
                        scoring_side = "left" if in_left_goal_zone else "right"
                        ball_in_goal = True
                        
                        # Save highlight clip
                        if config.enable_highlights and config.auto_save_shots:
                            threading.Thread(target=lambda: save_highlight_clip("soccer_goal", config.clip_duration), daemon=True).start()
                        
                        # Generate commentary
                        if config.use_echo_ai:
                            captured_speed = int(shot_speed_mph)
                            def async_goal():
                                time.sleep(1.8)
                                print(f"[AI] ⚽ Generating goal commentary for {captured_speed} mph goal...")
                                with ollama_lock:
                                    bonus = ai.shot_on_net(captured_speed, config.your_player_name or "the striker", "Soccer")
                                if bonus:
                                    print(f"[AI] 📢 Goal commentary: {bonus}")
                                    try:
                                        speak(bonus, priority=True)
                                    except Exception as e:
                                        print(f"[SPEAK ERROR] {e}")
                                else:
                                    print(f"[AI] ⚠️ No goal commentary generated")
                            threading.Thread(target=async_goal, daemon=True).start()
                        else:
                            # speak(random.choice(goal_lines))  # Disabled to prevent errors
                            pass
                        last_shot_time = time.time()
                        ball_in_goal_prev = True
                    
                    # Update last ball position
                    soccer_last_ball_position = (ball_x, ball_y)
                    soccer_ball_missing_frames = 0
                else:
                    # Method 5: Ball disappeared in goal zone
                    soccer_ball_missing_frames += 1
                    if (soccer_last_ball_position and soccer_ball_missing_frames >= 5 and 
                        soccer_ball_missing_frames <= 15):
                        last_x, last_y = soccer_last_ball_position
                        was_in_zone = last_x < w * 0.15 or last_x > w * 0.85
                        if was_in_zone and not ball_in_goal_prev and time.time() - last_shot_time > config.shot_cooldown:
                            print(f"[SOCCER] 🎯 Goal detected via ball disappearance!")
                            if config.use_echo_ai:
                                def async_goal():
                                    time.sleep(1.8)
                                    with ollama_lock:
                                        bonus = ai.shot_on_net(0, config.your_player_name or "the striker", "Soccer")
                                    if bonus:
                                        try:
                                            speak(bonus, priority=True)
                                        except Exception as e:
                                            print(f"[SPEAK ERROR] {e}")
                                threading.Thread(target=async_goal, daemon=True).start()
                            else:
                                # speak(random.choice(goal_lines))  # Disabled to prevent errors
                                pass
                            last_shot_time = time.time()
                            ball_in_goal_prev = True
            # ============ END SOCCER GOAL DETECTION ============

            # Check if ball is stationary (kickoff/menu)
            ball_is_stationary = False
            if ball and len(ball_trail) >= 5:
                # Check if ball has moved significantly in last 5 frames
                # Deque doesn't support slicing; convert to list first
                recent_positions = [pos[0:2] for pos in list(ball_trail)[-5:]]
                max_movement = max(abs(recent_positions[i][0] - recent_positions[0][0]) + abs(recent_positions[i][1] - recent_positions[0][1]) for i in range(len(recent_positions)))
                ball_is_stationary = max_movement < 15  # Less than 15 pixels = stationary
                
                # Additional check: ball in center of field (kickoff position)
                in_center_zone = abs(ball[0] - w//2) < w * 0.15 and abs(ball[1] - h//2) < h * 0.15
                if in_center_zone and ball_is_stationary:
                    ball_is_stationary = True  # Definitely kickoff position
            
            # Possession commentary only during gameplay (NHL: use team/player names)
            # Skip if ball is stationary (kickoff or menu)
            if GAME_STATE == "in-game" and ball and players and time.time() - last_possession_time > 15 and not ball_is_stationary:
                closest = min(players, key=lambda p: (p[1] - ball[0])**2 + (p[2] - ball[1])**2)
                name = None
                team_name = None
                is_hockey = getattr(config, 'game_mode', 'rocket_league') == 'hockey'
                if closest[0] is not None and closest[0] in jersey_map:
                    info = jersey_map[closest[0]]
                    team = info.get("team", "home")
                    num = info.get("num", "97")
                    name = rosters[team].get(num, None)
                    if is_hockey:
                        team_name = config.home_team if team == "home" else config.away_team
                    else:
                        team_name = "Your Team" if team == "home" else "Opponent"
                if not name:
                    if is_hockey:
                        team_name = config.home_team if closest[0] == TEAMMATE else config.away_team
                        name = team_name
                    else:
                        team_name = "Your Team" if closest[0] == TEAMMATE else "Opponent"
                        name = team_name
                _, _, possession_lines, _ = get_commentary_lines()
                if is_hockey and team_name:
                    # Prevent repeated team name if name is missing or equals team_name
                    if not name or name == team_name:
                        line = f"{team_name} {random.choice(possession_lines)}"
                    else:
                        line = f"{team_name}'s {name} {random.choice(possession_lines)}"
                else:
                    # Rocket League: use 'Your Team' or 'Opponent' for team_name
                    if not name or name == team_name:
                        line = f"{team_name} {random.choice(possession_lines)}"
                    else:
                        line = f"{team_name}'s {name} {random.choice(possession_lines)}"
                if random.random() < 0.5:
                    # speak(line)  # Disabled to prevent errors
                    last_possession_time = time.time()
            # Goalie save detection and commentary (NHL mode)
            if getattr(config, 'game_mode', 'rocket_league') == 'hockey' and GAME_STATE == "in-game":
                # Look for puck near goalie box and high speed to trigger save
                for det in detections:
                    if det.get('type') == 'goalie':
                        gx1, gy1, gx2, gy2 = det['box']
                        goalie_center = ((gx1 + gx2) // 2, (gy1 + gy2) // 2)
                        # If puck is close to goalie and shot_speed_mph > 10
                        if ball and abs(ball[0] - goalie_center[0]) < 60 and abs(ball[1] - goalie_center[1]) < 60 and shot_speed_mph > 10:
                            # Find goalie name
                            goalie_name = None
                            for p in players:
                                tid = p[0]
                                if tid in jersey_map:
                                    info = jersey_map[tid]
                                    team = info.get("team", "home")
                                    num = info.get("num", "97")
                                    n = rosters[team].get(num, None)
                                    if n:
                                        goalie_name = n
                                        break
                            if goalie_name:
                                # speak(f"Goalie {goalie_name} makes a great save!")  # Disabled to prevent errors
                                pass
                            else:
                                # speak("The goalie makes a great save!")  # Disabled to prevent errors
                                pass
                            break

            # Menu commentary - casual chat when not playing
            if GAME_STATE == "menu" and frame_count % 3000 == 0 and time.time() - last_comment_time > 90:
                # Check if ball is sitting still (speed near zero) - means waiting to start
                if ball and shot_speed_mph < 2:
                    ready_lines = [
                        "Time to get ready!",
                        "Match is about to start!",
                        "Get ready for kickoff!",
                        "Here we go, let's play!",
                        "Game starting soon!"
                    ]
                    # speak(random.choice(ready_lines))  # Disabled to prevent errors
                    last_comment_time = time.time()
                elif config.use_echo_ai:
                    menu_prompts = [
                        "waiting for the next match to start",
                        "chilling in the menu",
                        "taking a break between games",
                        "checking out the lobby"
                    ]
                    with ollama_lock:
                        sport = "Hockey" if getattr(config, 'game_mode', 'rocket_league') == 'hockey' else "Rocket League"
                        casual = ai.filler_commentary(random.choice(menu_prompts), sport)
                    if casual: 
                        print(f"[AI] Menu commentary: {casual}")
                        # speak(casual, priority=False)  # Disabled to prevent errors
                    last_comment_time = time.time()

            # Gameplay filler commentary
            if GAME_STATE == "in-game" and frame_count % 2000 == 0 and time.time() - last_comment_time > 60:
                if config.use_echo_ai:
                    with ollama_lock:
                        sport = "Hockey" if getattr(config, 'game_mode', 'rocket_league') == 'hockey' else "Rocket League"
                        filler = ai.filler_commentary("intense match", sport)
                else:
                    filler = None
                # speak(filler or random.choice(filler_lines), priority=False)  # Disabled to prevent errors

            frame_count += 1
            if frame_count % 60 == 0:
                fps = frame_count / (time.time() - start_time + 1e-5)
                in_replay_text = " [REPLAY]" if in_replay else ""
                print(f"FPS: {fps:.1f} | State: {GAME_STATE} | Activity: {activity_score} | Ball: {ball is not None} | Goals: {goals_scored}{in_replay_text}")
                # Debug: Show key settings
                shot_cooldown_remaining = max(0, (last_shot_time + config.shot_cooldown) - current_time)
                print(f"  [DEBUG] use_echo_ai={config.use_echo_ai} | processing={processing} | shot_cooldown remaining={shot_cooldown_remaining:.1f}s")

        except Exception as e:
            print(f"[ERROR IN LOOP] {e}")
            traceback.print_exc()
            time.sleep(0.1)

    # speak("Broadcast ended. Great game!")  # Disabled to prevent errors
    print("[OK] Broadcast stopped")

# ============================================================
#                VOICE CHAT WORKER
# ============================================================

def clean_prompt_for_voice_chat(prompt: str) -> str:
    """Remove streaming/follower marketing from prompts for personal voice chat"""
    import re
    
    # Patterns to remove - anything about followers, streaming, subscribing
    patterns_to_remove = [
        r"We're .*? trying to .*? 1,000 followers.*?!",
        r"We're .*? reach.*? 1,000 followers.*?!",
        r"We're.*? building to 1,000 followers.*?!",
        r"We're.*? grinding to 1,000 followers.*?!",
        r"We're.*? compiling 1,000 followers.*?!",
        r"The spirits tell me we're destined for 1,000 followers!",
        r"[Hh]it that [Ff][Oo][Ll][Ll][Oo][Ww].*?!",
        r"[Ss][Mm][Aa][Ss][Hh] that follow.*?!",
        r"[Ss][Uu][Bb][Ss][Cc][Rr][Ii][Bb][Ee].*?!",
        r"execute FOLLOW and SUBSCRIBE protocols!",
        r"Follow and subscribe to align your chakras.*?!",
        r"streaming.*? LIVE.*?",
        r"We're LIVE on stream.*?",
        r"you'll be in my monologue!",
        r"please execute FOLLOW and SUBSCRIBE protocols!"
    ]
    
    cleaned = prompt
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up any double spaces or punctuation issues
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s+!', '!', cleaned)
    cleaned = re.sub(r'\s+\.', '.', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def voice_chat_worker():
    """Dedicated voice chat mode - no commentary, just conversation"""
    global voice_chat_active, voice_chat_status, overlay_state, overlay_last_update
    
    if not WHISPER_AVAILABLE:
        print("[ERROR] Whisper not available for voice chat")
        # speak("Sorry, voice chat requires Whisper to be installed.")  # Disabled to prevent errors
        return
    
    # Initialize AI for voice chat
    if config.ai_provider == 'pollinations':
        ai_chat = PollinationsAI(
            model=config.pollinations_model,
            voice=config.pollinations_voice,
            max_tokens=config.echo_max_tokens
        )
    else:
        ai_chat = EchoAI(
            enabled=True,
            temperature=0.9,  # Higher temp for more creative/varied personality responses
            max_tokens=config.echo_max_tokens,
            ollama_model=config.ollama_model,
            personality=config.echo_personality
        )
    
    # speak("Voice chat activated! I'm listening...")  # Disabled to prevent errors
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 3
    
    p = pyaudio.PyAudio()
    
    print("🎤 Voice chat active - speak naturally")
    
    while voice_chat_active:
        try:
            # Simple protection: Mute mic when TTS is actively speaking
            # No cooldown delay - allows natural conversation flow
            if tts_speaking or pygame.mixer.get_busy():
                print("🔇 (TTS speaking, mic muted...)")
                time.sleep(0.1)  # Short check interval
                continue
            
            # Record audio
            voice_chat_status = "🎤 Listening for your voice..."
            overlay_state = "listening"
            overlay_last_update = time.time()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            print("🎤 Listening...")
            
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                if not voice_chat_active:
                    break
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            if not voice_chat_active:
                break
            
            # Check again AFTER recording - if TTS started during recording, skip this audio
            if tts_speaking or pygame.mixer.get_busy():
                print("🔇 (TTS started during recording, discarding audio)")
                continue
            
            # Check audio level - skip if too quiet (likely just noise/echo)
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_level = np.abs(audio_data).mean()
            if audio_level < 2:  # Skip very quiet audio (background noise) - very low threshold for quiet mics
                print(f"🔇 (Audio too quiet: {audio_level:.0f}, skipping)")
                continue
            
            # Convert to float for Whisper transcription
            voice_chat_status = "🔄 Transcribing your speech..."
            audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
            
            # Transcribe directly from numpy array (bypasses file I/O issues on Windows)
            result = whisper_model.transcribe(audio_float, language="en", fp16=False)
            text = result["text"].strip()
            
            # CRITICAL: Filter out if this matches what we just spoke (feedback loop)
            text_lower = text.lower().strip()
            if last_spoken_text and (text_lower in last_spoken_text or last_spoken_text in text_lower):
                print(f"🔇 Filtered feedback: '{text[:30]}...' matches recent TTS")
                continue
            
            # Check for voice commands BEFORE noise filtering
            command_executed = False
            
            # Command: Launch Rocket League
            if any(phrase in text_lower for phrase in ['launch rocket league', 'open rocket league', 'start rocket league', 'play rocket league']):
                command_executed = True
                # speak("Launching Rocket League!")  # Disabled to prevent errors
                threading.Thread(target=lambda: os.system('start steam://rungameid/252950'), daemon=True).start()
            
            # Command: Open YouTube
            elif any(phrase in text_lower for phrase in ['open youtube', 'go to youtube', 'show me youtube']):
                command_executed = True
                # speak("Opening YouTube!")  # Disabled to prevent errors
                threading.Thread(target=lambda: webbrowser.open('https://www.youtube.com'), daemon=True).start()
            
            # Command: Search YouTube for video
            elif 'search youtube for' in text_lower or 'find on youtube' in text_lower or 'youtube search' in text_lower:
                command_executed = True
                # Extract search query
                query = text_lower.split('search youtube for')[-1] if 'search youtube for' in text_lower else text_lower.split('find on youtube')[-1] if 'find on youtube' in text_lower else text_lower.split('youtube search')[-1]
                query = query.strip()
                if query:
                    # speak(f"Searching YouTube for {query}")  # Disabled to prevent errors
                    search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                    threading.Thread(target=lambda: webbrowser.open(search_url), daemon=True).start()
                else:
                    # speak("What would you like me to search for on YouTube?")  # Disabled to prevent errors
                    pass
            
            # Command: Web search (Google)
            elif any(phrase in text_lower for phrase in ['search for', 'google', 'look up', 'search the web', 'find information about']):
                command_executed = True
                # Extract search query
                query = None
                for trigger in ['search for', 'google', 'look up', 'search the web for', 'find information about']:
                    if trigger in text_lower:
                        query = text_lower.split(trigger)[-1].strip()
                        break
                if query and len(query) > 2:
                    # speak(f"Searching for {query}")  # Disabled to prevent errors
                    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                    threading.Thread(target=lambda: webbrowser.open(search_url), daemon=True).start()
                else:
                    # speak("What would you like me to search for?")  # Disabled to prevent errors
                    pass
            
            # Command: Open website
            elif any(phrase in text_lower for phrase in ['open website', 'go to', 'visit']):
                command_executed = True
                # Extract URL/domain
                url_part = text_lower.split('open website')[-1] if 'open website' in text_lower else text_lower.split('go to')[-1] if 'go to' in text_lower else text_lower.split('visit')[-1]
                url_part = url_part.strip().replace(' ', '')
                if url_part:
                    if not url_part.startswith('http'):
                        url_part = 'https://' + url_part
                    if not url_part.endswith('.com') and '.' not in url_part:
                        url_part += '.com'
                    # speak(f"Opening {url_part}")  # Disabled to prevent errors
                    threading.Thread(target=lambda: webbrowser.open(url_part), daemon=True).start()
                else:
                    # speak("Which website would you like to visit?")  # Disabled to prevent errors
                    pass
            
            # If command was executed, continue to next iteration
            if command_executed:
                continue
            
            # Enhanced noise pattern list including common false triggers AND garbled TTS
            noise_patterns = ['thank you', 'thanks for watching', 'subtitle', 'like and subscribe', 
                            'music', 'applause', 'laughter', '[', ']', '♪', 'www.', 
                            'bye-bye', 'good luck', 'hold on', "i'm waiting", "i'm not even sure",
                            'see you', 'goodbye', 'bon voyage', 'what were we', 'wait what',
                            'como', 'seles', 'patent', 'tasquiz', 'incoming', 'opponents', 'possession',
                            'poseesion', 'has possession', 'have possession', 'the opponent']
            is_noise = any(pattern in text_lower for pattern in noise_patterns)
            
            # Also filter very short or very repetitive text (increased threshold from 0.5 to 0.6)
            words = text.split()
            is_repetitive = len(words) > 3 and len(set(words)) / len(words) < 0.6
            
            if text and len(text) > 3 and not is_noise and not is_repetitive:
                print(f"👤 You: {text}")
                
                # Update overlay with what user said
                global overlay_voice_input
                overlay_voice_input = text
                
                # 🔥 Detect content box commands (show image, play video, open webpage)
                text_lower = text.lower()
                handled_command = False
                
                # SHOW IMAGE / PICTURE
                if ("show" in text_lower or "display" in text_lower or "let me see" in text_lower) and ("image" in text_lower or "picture" in text_lower or "photo" in text_lower or "pic" in text_lower):
                    # Extract subject - try multiple patterns, robust to phrasing
                    import re, urllib.parse
                    subject = None
                    # Try to extract between command and image word
                    match = re.search(r'(?:show|display|let me see)(?:\s+me)?\s+(?:a|an|the)?\s*(.+?)(?:\s+(?:image|picture|photo|pic))', text_lower)
                    if match:
                        subject = match.group(1).strip()
                    # Fallback: after 'of' or 'about'
                    if not subject:
                        for kw in [" of ", " about "]:
                            if kw in text_lower:
                                subject = text_lower.split(kw, 1)[-1].strip()
                                break
                    if not subject or len(subject) < 2:
                        subject = "a cool image"
                    encoded_prompt = urllib.parse.quote(subject)
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
                    show_external_url(image_url, f"🖼️ {subject.capitalize()}")
                    print(f"[CONTENT BOX] Opening image in browser: {subject}")
                    handled_command = True
                
                # PLAY VIDEO
                elif ("play" in text_lower or "show" in text_lower or "watch" in text_lower) and "video" in text_lower:
                    # Extract topic after "about" or "of" or between command and 'video'
                    topic = None
                    match = re.search(r'(?:play|show|watch)(?:\s+me)?\s+(?:a|an|the)?\s*(.+?)(?:\s+video)', text_lower)
                    if match:
                        topic = match.group(1).strip()
                    if not topic:
                        for kw in [" about ", " of "]:
                            if kw in text_lower:
                                topic = text_lower.split(kw, 1)[-1].strip()
                                break
                    if not topic or len(topic) < 2:
                        topic = "space"
                    video_url = f"https://www.youtube.com/embed?listType=search&list={topic.replace(' ', '+')}&autoplay=1"
                    show_external_url(video_url, f"🎥 {topic.capitalize()}")
                    print(f"[CONTENT BOX] Auto-playing video: {topic}")
                    handled_command = True
                
                # OPEN WEBSITE
                elif "open" in text_lower or "go to" in text_lower or "launch" in text_lower:
                    # Map common site names
                    site_map = {
                        "youtube": "https://youtube.com",
                        "google": "https://google.com",
                        "twitter": "https://twitter.com",
                        "reddit": "https://reddit.com",
                        "twitch": "https://twitch.tv",
                        "github": "https://github.com",
                        "discord": "https://discord.com",
                        "spotify": "https://spotify.com"
                    }
                    
                    for site_name, site_url in site_map.items():
                        if site_name in text_lower:
                            show_webpage(site_url, f"🌐 {site_name.capitalize()}")
                            print(f"[CONTENT BOX] Opening {site_name}")
                            handled_command = True
                            break
                
                # SEARCH / GOOGLE (only for explicit "search google for X" or "google X")
                elif ("search google" in text_lower or (text_lower.startswith("google ") and len(text_lower) > 7)):
                    # Extract query
                    query = text
                    if "search google for" in text_lower:
                        query = text.split("search google for", 1)[-1].strip()
                    elif text_lower.startswith("google "):
                        query = text[7:].strip()
                    
                    # Open Google search in external browser
                    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                    show_external_url(search_url, f"🔍 Search: {query}")
                    print(f"[CONTENT BOX] Searching Google: {query}")
                    handled_command = True
                
                # If a command was handled, the AI should acknowledge it
                if handled_command:
                    print(f"[VOICE CHAT] Content box command detected and executed")
                
                # AUTO-SEARCH for questions needing real-time data
                if not handled_command:
                    needs_search = False
                    search_triggers = [
                        ("who won", "game"), ("what's the score", ""), ("who's winning", ""),
                        ("what's the weather", ""), ("how's the weather", ""),
                        ("what time", ""), ("when is", ""), ("when does", ""),
                        ("who is", "playing"), ("who are", "playing"),
                        ("latest", "news"), ("recent", "news"), ("current", ""),
                        ("stock price", ""), ("how much is", "")
                    ]
                    
                    for trigger1, trigger2 in search_triggers:
                        if trigger1 in text_lower and (not trigger2 or trigger2 in text_lower):
                            needs_search = True
                            break
                    
                    if needs_search:
                        # Fetch answer from Google and give to AI as context
                        try:
                            import requests
                            from bs4 import BeautifulSoup
                            
                            search_url = f"https://www.google.com/search?q={text.replace(' ', '+')}"
                            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                            response = requests.get(search_url, headers=headers, timeout=5)
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # Try to find the featured snippet or answer
                            answer = None
                            
                            # Check for featured snippet
                            snippet = soup.find('div', {'class': 'BNeawe'})
                            if snippet:
                                answer = snippet.get_text()
                            
                            # Check for sports score widget
                            if not answer:
                                score_divs = soup.find_all('div', {'class': 'imso_mh__l-tm-sc'})
                                if len(score_divs) >= 2:
                                    teams = soup.find_all('div', {'class': 'imso_mh__first-tn-ed'})
                                    if len(teams) >= 2:
                                        answer = f"{teams[0].get_text()} {score_divs[0].get_text()} - {teams[1].get_text()} {score_divs[1].get_text()}"
                            
                            if answer:
                                print(f"[AUTO SEARCH] Found answer: {answer}")
                                # Give answer to AI as context
                                text = f"{text}\n\nSearch result: {answer}\n\nTell me this answer in your personality."
                            else:
                                print(f"[AUTO SEARCH] No clear answer found, opening search")
                                search_url = f"https://www.google.com/search?q={text.replace(' ', '+')}"
                                show_external_url(search_url, f"🔍 {text}")
                        except Exception as e:
                            print(f"[AUTO SEARCH] Error fetching answer: {e}")
                            # Fallback: open search
                            search_url = f"https://www.google.com/search?q={text.replace(' ', '+')}"
                            show_external_url(search_url, f"🔍 {text}")

                
                # Update AI before each response (in case user changed it in UI)
                if config.ai_provider == 'pollinations':
                    ai_chat.set_model(config.pollinations_model)
                    ai_chat.set_voice(config.pollinations_voice)
                    print(f"[VOICE CHAT] Using Pollinations ({config.pollinations_model})")
                    # Apply personality to Pollinations too!
                    if config.custom_character_prompt:
                        print(f"[VOICE CHAT] Using custom character")
                        ai_chat.system_prompt = clean_prompt_for_voice_chat(config.custom_character_prompt)
                    else:
                        print(f"[VOICE CHAT] Loading personality: {config.echo_personality}")
                        ai_chat.set_personality(config.echo_personality, game_mode=getattr(config, 'game_mode', 'rocket_league'), is_voice_chat=True)
                else:
                    if config.custom_character_prompt:
                        print(f"[VOICE CHAT] Using custom character")
                        ai_chat.system_prompt = clean_prompt_for_voice_chat(config.custom_character_prompt)
                    else:
                        print(f"[VOICE CHAT] Loading personality: {config.echo_personality}")
                        ai_chat.set_personality(config.echo_personality, game_mode=getattr(config, 'game_mode', 'rocket_league'), is_voice_chat=True)
                        # Set voice to match personality if available
                        if getattr(config, 'game_mode', 'rocket_league') == 'hockey':
                            hockey_voice_map = {
                                'Doc Emrick': 'en-US-GuyNeural',
                                'Canadian Color': 'en-CA-LiamNeural',
                                'Old School Coach': 'en-US-TonyNeural',
                                'Goalie Analyst': 'en-US-SteffanNeural',
                                'Hockey Mom': 'en-US-JennyNeural',
                                'Rink DJ': 'en-US-BrandonNeural',
                                'Enforcer': 'en-US-AndrewNeural',
                                'French-Canadian': 'fr-CA-SylvieNeural',
                                'Sara X': 'en-US-SaraNeural'
                            }
                            selected_voice = hockey_voice_map.get(config.echo_personality, None)
                            if selected_voice:
                                config.tts_speaker = selected_voice
                
                # Get AI response (locked to prevent simultaneous requests)
                provider_name = "Pollinations" if config.ai_provider == 'pollinations' else "Echo"
                voice_chat_status = f"🧠 {provider_name} is thinking about: '{text[:50]}...'"
                overlay_state = "thinking"
                overlay_last_update = time.time()
                
                with ollama_lock:
                    response = ai_chat.chat(text)
                    
                voice_chat_status = f"🗣️ {provider_name} is speaking..."
                if response:
                    # speak() handles both caption and phonemes together
                    # speak(response, priority=True)  # Disabled to prevent errors
                    
                    # Auto-generate image if image mode is enabled
                    if getattr(config, 'image_mode_enabled', False):
                        print(f"[VOICE CHAT] 🎨 Image mode enabled - generating art for conversation")
                        try:
                            import urllib.parse
                            # Create image generation prompt from conversation
                            extract_prompt = f"""Based on this conversation:
You: {text}
AI: {response}

Generate a creative, vivid image prompt (1-2 sentences) for an AI image generator. Focus on visual concepts, scenes, art styles, colors, and mood. Be specific and artistic.

Image prompt:"""
                            
                            # Get AI to generate the image prompt
                            image_prompt = None
                            if ai_chat:
                                image_prompt = ai_chat.chat(extract_prompt)
                            
                            if not image_prompt:
                                # Fallback: use keywords from conversation
                                image_prompt = f"Abstract art inspired by: {text[:100]}"
                            
                            # Clean up the prompt
                            image_prompt = image_prompt.strip('"\'').split('\n')[0].strip()
                            
                            print(f"[VOICE CHAT] [IMAGE] Auto-generated prompt: {image_prompt}")
                            
                            # Generate image URL (Pollinations)
                            encoded_prompt = urllib.parse.quote(image_prompt)
                            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
                            
                            print(f"[VOICE CHAT] [IMAGE] ✓ Auto-generated image URL: {image_url}")
                            
                            # Send to overlay
                            global current_background_image
                            current_background_image = image_url
                            print(f"[VOICE CHAT] [IMAGE] ✓ Image sent to overlay")
                        except Exception as img_error:
                            print(f"[VOICE CHAT] [IMAGE] Error generating image: {img_error}")
                    
                    time.sleep(0.5)  # Brief pause after speaking
                    voice_chat_status = "✅ Ready for next question"
            
        except Exception as e:
            print(f"[VOICE CHAT ERROR] {e}")
            time.sleep(0.5)
    
    p.terminate()
    voice_chat_status = "⚪ Voice chat stopped"
    # speak("Voice chat ended.")  # Disabled to prevent errors
    print("[OK] Voice chat stopped")

# ============================================================
#                HIGHLIGHT CLIP FUNCTIONS
# ============================================================

def save_highlight_clip(clip_type="goal", duration=10):
    """Save a highlight clip from the buffer"""
    global clip_buffer, highlight_clips
    
    if len(clip_buffer) < 30:  # Need at least 1 second of footage
        print("[CLIP] Not enough frames in buffer")
        return None
    
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        clips_dir = "highlights"
        os.makedirs(clips_dir, exist_ok=True)
        
        filename = f"{clips_dir}/PLAIX_{clip_type}_{timestamp}.mp4"
        
        # Get frames from buffer (last N seconds)
        fps = 30
        playback_speed = 1.0  # 1.0 = normal speed
        num_frames = min(int(duration * fps), len(clip_buffer))
        frames_to_save = list(clip_buffer)[-num_frames:]

        if not frames_to_save:
            return None

        # Get frame dimensions
        h, w = frames_to_save[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, int(fps * playback_speed), (w, h))

        for frame in frames_to_save:
            out.write(frame)
        
        out.release()
        
        clip_info = {
            'filename': filename,
            'type': clip_type,
            'timestamp': timestamp,
            'duration': duration
        }
        highlight_clips.append(clip_info)
        
        print(f"[CLIP] [OK] Saved {clip_type} highlight: {filename}")
        return filename
        
    except Exception as e:
        print(f"[CLIP] Error saving highlight: {e}")
        return None

def get_highlights_list():
    """Get list of saved highlights with sharing info"""
    global highlight_clips
    if not highlight_clips:
        return "📹 **No highlights saved yet.**\n\nPlay some games and score goals to auto-capture epic moments!"
    
    result = f"📹 **Saved Highlights ({len(highlight_clips)} clips):**\n\n"
    
    total_duration = sum(clip['duration'] for clip in highlight_clips)
    result += f"*Total footage: {total_duration} seconds*\n\n"
    
    for i, clip in enumerate(highlight_clips, 1):
        clip_type_emoji = "⚽" if clip['type'] == "goal" else "🚀"
        clip_type_str = str(clip['type']) if not isinstance(clip['type'], str) else clip['type']
        result += f"{i}. {clip_type_emoji} **{clip_type_str.upper()}** - {clip['timestamp']}\n"
        result += f"   Duration: {clip['duration']}s | File: `{os.path.basename(clip['filename'])}`\n\n"
    
    result += "\n---\n\n"
    result += "**📤 Sharing Options:**\n"
    result += "• Upload to **Twitter/X, TikTok, Instagram Reels**\n"
    result += "• Share to **Reddit** r/RocketLeague\n"
    result += "• Send in **Discord** servers\n"
    result += "• Upload to **YouTube Shorts**\n"
    result += f"\n📂 Clips saved in: `{os.path.abspath('highlights')}`"
    
    return result

# ============================================================
#                BROADCAST CONTROL FUNCTIONS
# ============================================================

def start_broadcast(debug_mode):
    global processing
    processing = True
    t = threading.Thread(target=broadcast_worker, args=(debug_mode,), daemon=True)
    t.start()
    return "Commentary started! Avatar overlay should open automatically."

def stop_broadcast():
    global processing
    processing = False
    return "Commentary stopped!"

def start_voice_chat():
    global voice_chat_active
    voice_chat_active = True
    t = threading.Thread(target=voice_chat_worker, daemon=True)
    t.start()
    return "Voice chat started!"

def stop_voice_chat():
    global voice_chat_active
    voice_chat_active = False
    return "Voice chat stopped!"

# ============================================================
#                       HELPER FUNCTIONS
# ============================================================

def get_ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return models if models else ["llama3.2"]
    except:
        pass
    return ["llama3.2"]

# ============================================================
#                  OVERLAY-ONLY SYSTEM (NO GRADIO)
# ============================================================

def get_available_monitors():
    """Get list of available monitors"""
    try:
        with mss.mss() as sct:
            monitors = sct.monitors[1:]  # Skip [0] which is all monitors combined
            return [f"Monitor {i+1} ({m['width']}x{m['height']})" for i, m in enumerate(monitors)]
    except:
        return ["Monitor 1", "Monitor 2", "Monitor 3"]

def get_preview_frame():
    """Get current preview frame for display"""
    global live_preview_frame
    if live_preview_frame is not None:
        # Convert BGR to RGB for display
        return cv2.cvtColor(live_preview_frame, cv2.COLOR_BGR2RGB)
    else:
        # Return black frame if no preview available
        return np.zeros((360, 640, 3), dtype=np.uint8)

def get_voice_status():
    """Get current voice chat status for UI"""
    global voice_chat_status
    return voice_chat_status

def update_monitor(monitor_str):
    """Extract monitor index from dropdown selection"""
    try:
        index = int(monitor_str.split()[1])
        config.monitor_index = index
        return f"[OK] Monitor set to {index}"
    except:
        return "[ERROR] Error setting monitor"

# ============================================================
#                  API ENDPOINT FOR OVERLAY
# ============================================================

# API endpoint function for overlay status
def get_overlay_status():
    """API endpoint for game overlay to poll status"""
    global overlay_state, overlay_caption_text, overlay_voice_input, overlay_last_update, overlay_tracking_data, current_phonemes, phonemes_sent, overlay_image_url, content_box_data, content_box_sent
    
    # Auto-reset state after 5 seconds of no updates (but keep phonemes until next speak)
    if time.time() - overlay_last_update > 5:
        overlay_state = "idle"
        overlay_caption_text = {"text": "", "speaker": config.echo_personality or "Echo AI"}
        overlay_voice_input = ""
        # Don't clear phonemes here - let them persist until next speak() call
    # Always return caption_text as dict with 'text' and 'speaker'
    caption = overlay_caption_text
    if not isinstance(caption, dict):
        caption = {"text": str(caption), "speaker": config.echo_personality or "Echo AI"}
    
    # Only send phonemes once when they're new (not yet sent)
    phonemes_to_send = []
    try:
        if current_phonemes and not phonemes_sent:
            phonemes_to_send = current_phonemes
            phonemes_sent = True
            print(f"[API DEBUG] Sending {len(phonemes_to_send)} phonemes to overlay (ONCE)", flush=True)
            # Also attach phonemes into the caption dict for overlays that expect them there
            if isinstance(caption, dict):
                caption['phonemes'] = phonemes_to_send
    except Exception as e:
        print(f"[API ERROR] while preparing phonemes to send: {e}")
        import traceback
        traceback.print_exc()
    
    # Only send content_box once when it's new (not yet sent)
    content_box_to_send = None
    if content_box_data and not content_box_sent:
        content_box_to_send = content_box_data
        content_box_sent = True  # Mark as sent
        print(f"[API DEBUG] Sending content_box to overlay (ONCE): {content_box_data.get('action', 'unknown')}", flush=True)
        # Clear IMMEDIATELY after sending to prevent infinite loop
        content_box_data = None
        content_box_sent = False  # Reset for next time
    
    return {
        "state": overlay_state,
        "status_text": voice_chat_status if voice_chat_active else "",
        "caption_text": caption,
        "voice_input": overlay_voice_input,
        "image_url": overlay_image_url,  # Generated image URL for image mode
        "show_tracking": config.show_tracking,
        "tracking_data": overlay_tracking_data,  # Always send tracking data
        "ball_trail_enabled": config.enable_ball_trail,
        "trail_style": config.trail_style,
        "phonemes": phonemes_to_send,  # Only send when new
        "avatar_url": getattr(config, 'avatar_url', None),  # Avatar URL from gallery
        "show_avatar_window": getattr(config, 'show_avatar_window', False),  # Avatar window flag
        "content_box": content_box_to_send  # Only send once, then None
    }



# ================== API: OVERLAY PERSONALITY ENDPOINTS (MUST BE AT BOTTOM) ===================
try:
    import flask
    from flask import Flask, jsonify, request
    import threading as _threading

    if 'api_app' not in globals():
        api_app = Flask(__name__)

    @api_app.route('/api/get_personalities', methods=['GET'])
    def api_get_personalities():
        try:
            from echo_ai import EchoAI
            game_mode = request.args.get('mode', getattr(config, 'game_mode', 'rocket_league'))
            # Return the full PERSONALITIES_JS list for the overlay
            personalities_full = EchoAI.get_personalities_js()
            personalities = EchoAI.get_personality_list(game_mode)
            current = getattr(config, 'echo_personality', None)
            
            print(f"[API] personalities_full type: {type(personalities_full)}")
            print(f"[API] personalities_full length: {len(personalities_full) if personalities_full else 'None'}")
            if personalities_full and len(personalities_full) > 0:
                print(f"[API] First personality: {personalities_full[0]}")
            
            result = {
                'personalities': personalities,
                'personalities_js': personalities_full,
                'current': current
            }
            print(f"[API] About to send response with {len(personalities_full)} personalities_js")
            return jsonify(result)
        except Exception as e:
            print(f"[API ERROR] get_personalities failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'personalities': [], 'personalities_js': [], 'current': None}), 500

    @api_app.route('/api/set_personality/<personality>', methods=['GET'])
    def api_set_personality(personality):
        from echo_ai import EchoAI
        decoded = personality

    @api_app.route('/api/health', methods=['GET'])
    def api_health():
        try:
            warmup = False
            if 'echo_ai' in globals() and echo_ai is not None:
                warmup = getattr(echo_ai, '_warmup_ready', False) or not getattr(echo_ai, 'use_ollama', False)
            # Ready when server is up; warmup indicates model readiness
            return jsonify({'status': 'ok', 'ready': bool(warmup), 'warmup': bool(warmup)})
        except Exception as e:
            print(f"[API ERROR] health check failed: {e}")
            return jsonify({'status': 'error', 'ready': False}), 500

        try:
            import urllib.parse
            decoded = urllib.parse.unquote(personality)
        except Exception:
            pass
        game_mode = getattr(config, 'game_mode', 'rocket_league')
        if decoded in EchoAI.get_personality_list(game_mode):
            config.echo_personality = decoded
            save_config()
            return jsonify({'status': 'success', 'personality': decoded})
        else:
            return jsonify({'status': 'error', 'message': f'Unknown personality: {decoded}'}), 400
    
    @api_app.route('/api/get_team_color', methods=['GET'])
    def api_get_team_color():
        try:
            print(f"[API] get_team_color called")
            team = getattr(config, 'team_color', 'Blue')  # Default to Blue if not set
            print(f"[API] Returning team color: {team}")
            result = jsonify({'team_color': team})
            print(f"[API] JSON result created successfully")
            return result
        except Exception as e:
            print(f"[API ERROR] get_team_color failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'team_color': 'Blue'})
    
    @api_app.route('/api/set_team_color/<team>', methods=['GET'])
    def api_set_team_color(team):
        try:
            if team in ['Blue', 'Orange']:
                config.team_color = team
                save_config()
                print(f"[API] Team color set to: {team}")
                return jsonify({'status': 'success', 'team_color': team})
            else:
                return jsonify({'status': 'error', 'message': f'Invalid team color: {team}'}), 400
        except Exception as e:
            print(f"[API ERROR] set_team_color failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @api_app.route('/api/overlay_status', methods=['GET', 'OPTIONS'])
    def api_overlay_status():
        try:
            data = get_overlay_status()
            return jsonify(data)
        except Exception as e:
            print(f"[API ERROR] overlay_status failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @api_app.route('/api/get_nhl_teams', methods=['GET'])
    def api_get_nhl_teams():
        """Return list of NHL teams"""
        try:
            nhl_teams = sorted(list(TEAM_ABBR_MAP.keys()))
            return jsonify({'teams': nhl_teams})
        except Exception as e:
            print(f"[API ERROR] get_nhl_teams failed: {e}")
            return jsonify({'teams': []}), 500

    @api_app.route('/api/get_teams', methods=['GET'])
    def api_get_teams():
        """Return current home and away teams"""
        try:
            home_team = getattr(config, 'home_team', 'Edmonton Oilers')
            away_team = getattr(config, 'away_team', 'Philadelphia Flyers')
            return jsonify({'home_team': home_team, 'away_team': away_team})
        except Exception as e:
            print(f"[API ERROR] get_teams failed: {e}")
            return jsonify({'home_team': '', 'away_team': ''}), 500

    @api_app.route('/api/set_teams/<home_team>/<away_team>', methods=['GET'])
    def api_set_teams(home_team, away_team):
        """Set home and away teams"""
        try:
            import urllib.parse
            home = urllib.parse.unquote(home_team)
            away = urllib.parse.unquote(away_team)
            config.home_team = home
            config.away_team = away
            load_rosters(home, away)
            save_config()
            return jsonify({'status': 'success', 'home_team': home, 'away_team': away})
        except Exception as e:
            print(f"[API ERROR] set_teams failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @api_app.route('/api/open_highlights', methods=['POST'])
    def api_open_highlights():
        """Open highlights folder in file explorer"""
        try:
            highlights_dir = os.path.abspath('highlights')
            if not os.path.exists(highlights_dir):
                os.makedirs(highlights_dir)
            
            # Open folder based on OS
            import platform
            if platform.system() == 'Windows':
                os.startfile(highlights_dir)
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open "{highlights_dir}"')
            else:  # Linux
                os.system(f'xdg-open "{highlights_dir}"')
            
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @api_app.route('/api/load_avatar', methods=['POST'])
    def api_load_avatar():
        """Load 3D avatar model from URL"""
        try:
            data = request.get_json()
            avatar_url = data.get('url', '')
            
            if not avatar_url:
                return jsonify({'status': 'error', 'message': 'No URL provided'}), 400
            
            # Store avatar URL in config for overlay to pick up
            config.avatar_url = avatar_url
            print(f"[API] Avatar URL set to: {avatar_url}")
            
            return jsonify({'status': 'success', 'url': avatar_url})
        except Exception as e:
            print(f"[API ERROR] load_avatar failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @api_app.route('/api/show_avatar_window', methods=['POST'])
    def api_show_avatar_window():
        """Signal to show avatar window/controls"""
        try:
            # This just sets a flag that the overlay can check
            config.show_avatar_window = True
            print(f"[API] Avatar window flag set")
            
            return jsonify({'status': 'success'})
        except Exception as e:
            print(f"[API ERROR] show_avatar_window failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @api_app.route('/api/clear_avatar_window_flag', methods=['POST'])
    def api_clear_avatar_window_flag():
        """Clear the avatar window show flag after overlay has processed it"""
        try:
            config.show_avatar_window = False
            print(f"[API] Avatar window flag cleared")
            
            return jsonify({'status': 'success'})
        except Exception as e:
            print(f"[API ERROR] clear_avatar_window_flag failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @api_app.route('/api/show_content_box', methods=['POST'])
    def api_show_content_box():
        """Display content in the interactive content box"""
        global content_box_data
        try:
            data = request.get_json()
            action = data.get('action', 'show_html')  # show_image, show_video, show_webpage, show_html, hide
            
            content_box_data = {
                'action': action,
                'url': data.get('url'),
                'html': data.get('html'),
                'title': data.get('title', 'AI Content')
            }
            
            print(f"[CONTENT BOX] Request received: {action}, title: {content_box_data['title']}")
            return jsonify({'status': 'success', 'action': action})
        except Exception as e:
            print(f"[API ERROR] show_content_box failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @api_app.route('/api/clear_content_box_flag', methods=['POST'])
    def api_clear_content_box_flag():
        """Clear the content box flag after overlay has processed it"""
        global content_box_data
        try:
            content_box_data = None
            return jsonify({'status': 'success'})
        except Exception as e:
            print(f"[API ERROR] clear_content_box_flag failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @api_app.route('/api/get_tts_engine', methods=['GET'])
    def api_get_tts_engine():
        """Get current TTS engine"""
        try:
            engine = getattr(config, 'tts_engine', 'edge')
            return jsonify({'engine': engine})
        except Exception as e:
            print(f"[API ERROR] open_highlights failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @api_app.route('/api/test_tts', methods=['POST'])
    def api_test_tts():
        """Trigger a test TTS playback from API. JSON: {"text":"..."}"""
        try:
            data = request.get_json(force=True, silent=True) or {}
            text = data.get('text', 'Test TTS - Hello! This is a test of the TTS system.')
            print(f"[API] TTS test requested: {text[:120]}")
            threading.Thread(target=lambda: speak(text, priority=True), daemon=True).start()
            return jsonify({'status': 'started', 'text': text})
        except Exception as e:
            print(f"[API ERROR] test_tts failed: {e}")
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def _run_api():
        api_app.run(port=7862, host='0.0.0.0', threaded=True)

    if not getattr(globals(), '_api_thread_started', False):
        _api_thread = _threading.Thread(target=_run_api, daemon=True)
        _api_thread.start()
        globals()['_api_thread_started'] = True
        print(f"[API] Registered routes: {[rule.rule for rule in api_app.url_map.iter_rules()]}")
except Exception as e:
    print(f"[API INIT ERROR] Could not start overlay API: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
#                       MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import socket
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    # Load saved configuration
    load_config()
    
    # Simple HTTP server for overlay API on port 7862
    class OverlayAPIHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            global processing, voice_chat_active, echo_ai

            if self.path == '/api/get_teams':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'home_team': getattr(config, 'home_team', ''), 'away_team': getattr(config, 'away_team', '')}).encode())
                return
            elif self.path.startswith('/api/set_teams/'):
                import urllib.parse
                # TODO: Implement /api/set_teams/ logic if needed
                return

            if self.path == '/api/get_game_mode':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'game_mode': getattr(config, 'game_mode', 'rocket_league')}).encode())
            elif self.path.startswith('/api/set_game_mode/'):
                import urllib.parse
                mode = urllib.parse.unquote(self.path.split('/')[-1])
                if mode in ['rocket_league', 'hockey', 'soccer']:
                    config.game_mode = mode
                    save_config()
                    reload_yolo_model()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success', 'game_mode': mode}).encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Invalid game mode'}).encode())
            elif self.path == '/api/overlay_status':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', '*')
                self.end_headers()
                data = get_overlay_status()
                self.wfile.write(json.dumps(data).encode())
            elif self.path == '/api/start_commentary':
                if not processing:
                    start_broadcast(False)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'started', 'active': processing}).encode())
            elif self.path == '/api/stop_commentary':
                stop_broadcast()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'stopped', 'active': processing}).encode())
            elif self.path == '/api/start_voice':
                if not voice_chat_active:
                    start_voice_chat()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'started', 'active': voice_chat_active}).encode())
            elif self.path == '/api/stop_voice':
                stop_voice_chat()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'stopped', 'active': voice_chat_active}).encode())
            elif self.path == '/api/get_player_name':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                current = config.your_player_name or ""
                self.wfile.write(json.dumps({'player_name': current}).encode())
            elif self.path == '/api/health':
                try:
                    warmup = False
                    if 'echo_ai' in globals() and echo_ai is not None:
                        warmup = getattr(echo_ai, '_warmup_ready', False) or not getattr(echo_ai, 'use_ollama', False)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'ok', 'ready': warmup, 'warmup': warmup}).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'error', 'ready': False}).encode())
            elif self.path.startswith('/api/set_player_name/'):
                import urllib.parse
                player_name = urllib.parse.unquote(self.path.split('/')[-1])
                config.your_player_name = player_name if player_name.strip() else None
                save_config()  # Save settings
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'player_name': player_name}).encode())
            elif self.path == '/api/get_personalities':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                # Return hockey personalities if in hockey mode
                mode = getattr(config, 'game_mode', 'rocket_league')
                personalities = EchoAI.get_personality_list(mode)
                personalities_full = EchoAI.get_personalities_js()  # Full data with names
                current = config.echo_personality
                print(f"[API] Sending {len(personalities_full)} personalities (personalities_js)")
                self.wfile.write(json.dumps({'personalities': personalities, 'personalities_js': personalities_full, 'current': current}).encode())
            elif self.path.startswith('/api/set_personality/'):
                import urllib.parse
                personality_name = urllib.parse.unquote(self.path.split('/')[-1])
                # Check if personality ID exists in PERSONALITIES_JS
                personalities_full = EchoAI.get_personalities_js()
                valid_ids = [p['id'] for p in personalities_full]
                
                if personality_name in valid_ids:
                    config.echo_personality = personality_name
                    save_config()  # Save settings
                    
                    # Update the global echo_ai instance's personality (only for EchoAI, not PollinationsAI)
                    if 'echo_ai' in globals() and echo_ai is not None:
                        if hasattr(echo_ai, 'set_personality'):
                            game_mode = getattr(config, 'game_mode', 'rocket_league')
                            echo_ai.set_personality(personality_name, game_mode=game_mode)
                    
                    print(f"[API] Personality changed to: {personality_name}")
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success', 'personality': personality_name}).encode())
                else:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'error', 'message': 'Personality not found'}).encode())
            elif self.path == '/api/get_team_color':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                team = getattr(config, 'team_color', 'Blue')
                print(f"[API] Returning team color: {team}")
                self.wfile.write(json.dumps({'team_color': team}).encode())
            elif self.path.startswith('/api/set_team_color/'):
                import urllib.parse
                team = urllib.parse.unquote(self.path.split('/')[-1])
                if team in ['Blue', 'Orange']:
                    config.team_color = team
                    save_config()
                    print(f"[API] Team color set to: {team}")
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success', 'team_color': team}).encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'error', 'message': f'Invalid team color: {team}'}).encode())
            
            # ============ TWITCH API ENDPOINTS ============
            elif self.path == '/api/get_twitch_settings':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                settings = {
                    'enabled': getattr(config, 'twitch_enabled', False),
                    'channel': getattr(config, 'twitch_channel', ''),
                    'oauth': getattr(config, 'twitch_oauth', '')
                }
                self.wfile.write(json.dumps(settings).encode())
            
            elif self.path.startswith('/api/set_twitch_enabled/'):
                import urllib.parse
                enabled = self.path.split('/')[-1].lower() == 'true'
                config.twitch_enabled = enabled
                save_config()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'enabled': enabled}).encode())
            
            elif self.path.startswith('/api/set_twitch_settings/'):
                import urllib.parse
                parts = self.path.split('/')
                channel = urllib.parse.unquote(parts[-3])
                oauth = urllib.parse.unquote(parts[-2])
                enabled = parts[-1].lower() == 'true'
                
                config.twitch_channel = channel
                config.twitch_oauth = oauth
                config.twitch_enabled = enabled
                save_config()
                
                print(f"[API] Twitch settings saved: channel={channel}, enabled={enabled}")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode())

            # ============ TIKTOK API ENDPOINTS ============
            elif self.path == '/api/get_tiktok_settings':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                settings = {
                    'enabled': getattr(config, 'tiktok_enabled', False),
                    'username': getattr(config, 'tiktok_username', ''),
                    'cookie': getattr(config, 'tiktok_cookie', '')
                }
                self.wfile.write(json.dumps(settings).encode())

            elif self.path.startswith('/api/set_tiktok_enabled/'):
                import urllib.parse
                enabled = self.path.split('/')[-1].lower() == 'true'
                config.tiktok_enabled = enabled
                save_config()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'enabled': enabled}).encode())

            elif self.path.startswith('/api/set_tiktok_settings/'):
                import urllib.parse
                parts = self.path.split('/')
                username = urllib.parse.unquote(parts[-4])
                cookie = urllib.parse.unquote(parts[-3])
                enabled = parts[-1].lower() == 'true'

                config.tiktok_username = username
                config.tiktok_cookie = cookie
                config.tiktok_enabled = enabled
                save_config()

                print(f"[API] TikTok settings saved: username=@{username}, enabled={enabled}")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode())
            
            elif self.path == '/api/get_voices':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                # Return voices based on current TTS engine
                if getattr(config, 'tts_engine', 'kokoro') == 'edge':
                    voices = get_edge_voices()
                else:
                    voices = KOKORO_VOICES
                current = config.tts_speaker
                self.wfile.write(json.dumps({'voices': voices, 'current': current}).encode())
            elif self.path.startswith('/api/set_voice/'):
                import urllib.parse
                voice_name = urllib.parse.unquote(self.path.split('/')[-1])
                # Check if voice is valid for current engine
                valid_voices = []
                if getattr(config, 'tts_engine', 'kokoro') == 'edge':
                    valid_voices = get_edge_voices()
                else:
                    valid_voices = KOKORO_VOICES
                
                if voice_name in valid_voices:
                    config.tts_speaker = voice_name
                    save_config()  # Save settings
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success', 'voice': voice_name}).encode())
                else:
                    self.send_response(404)
            elif self.path == '/api/preview_voice':
                # Preview the current voice with "Voice preview"
                import threading
                import uuid
                def preview_voice():
                    try:
                        print(f"[VOICE PREVIEW] Playing preview with voice: {config.tts_speaker}")
                        # Stop any current playback to release file
                        pygame.mixer.music.stop()
                        pygame.time.wait(100)  # Wait for file to be released
                        
                        # Use unique filename to avoid permission issues
                        preview_filename = f'voice_preview_{uuid.uuid4().hex[:8]}'
                        
                        if getattr(config, 'tts_engine', 'kokoro') == 'edge':
                            import edge_tts
                            import asyncio
                            async def speak_edge():
                                communicate = edge_tts.Communicate("Voice preview", config.tts_speaker)
                                await communicate.save(f'{preview_filename}.mp3')
                                pygame.mixer.music.load(f'{preview_filename}.mp3')
                                pygame.mixer.music.play()
                                # Clean up old preview files after playing
                                pygame.time.wait(100)
                                try:
                                    import os
                                    for f in os.listdir('.'):
                                        if f.startswith('voice_preview_') and f.endswith('.mp3'):
                                            try:
                                                os.remove(f)
                                            except:
                                                pass
                                except:
                                    pass
                            asyncio.run(speak_edge())
                        else:
                            # Kokoro TTS preview
                            text = "Voice preview"
                            try:
                                samples, sample_rate = kokoro_engine.create(text, voice=config.tts_speaker, speed=1.0)
                                # Write WAV using soundfile (samples can be numpy array or list)
                                sf.write(f'{preview_filename}.wav', samples, sample_rate)
                                pygame.mixer.music.load(f'{preview_filename}.wav')
                                pygame.mixer.music.play()
                            except Exception as preview_error:
                                print(f"[KOKORO PREVIEW] Error generating preview: {preview_error}")
                            # Clean up old preview files after playing
                            pygame.time.wait(100)
                            try:
                                import os
                                for f in os.listdir('.'):
                                    if f.startswith('voice_preview_') and f.endswith('.wav'):
                                        try:
                                            os.remove(f)
                                        except:
                                            pass
                            except:
                                pass
                        print(f"[VOICE PREVIEW] Preview played successfully")
                    except Exception as e:
                        print(f"[VOICE PREVIEW] Error: {e}")
                
                threading.Thread(target=preview_voice, daemon=True).start()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'previewing'}).encode())
            elif self.path == '/api/get_tts_engines':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                engines = []
                if KOKORO_AVAILABLE:
                    engines.append({'value': 'kokoro', 'label': 'Kokoro TTS'})
                if EDGE_TTS_AVAILABLE:
                    engines.append({'value': 'edge', 'label': 'Edge TTS'})
                # XTTS support removed
                    engines.append({'value': 'xtts', 'label': 'XTTS'})
                current = getattr(config, 'tts_engine', 'kokoro')
                self.wfile.write(json.dumps({'engines': engines, 'current': current}).encode())
            elif self.path.startswith('/api/set_tts_engine/'):
                import urllib.parse
                engine_name = urllib.parse.unquote(self.path.split('/')[-1])
                valid_engines = []
                if KOKORO_AVAILABLE:
                    valid_engines.append('kokoro')
                if EDGE_TTS_AVAILABLE:
                    valid_engines.append('edge')
                # XTTS support removed
                    valid_engines.append('xtts')
                
                if engine_name in valid_engines:
                    # Set TTS engine and update speaker to first available voice
                    voices = set_tts_engine(engine_name)
                    save_config()  # Save settings
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success', 'engine': engine_name, 'voices': voices}).encode())
                else:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'error', 'message': 'Engine not available'}).encode())
            elif self.path == '/api/get_ollama_models':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                models = get_ollama_models()
                current = config.ollama_model
                self.wfile.write(json.dumps({'models': models, 'current': current}).encode())
            elif self.path.startswith('/api/set_ollama_model/'):
                import urllib.parse
                model_name = urllib.parse.unquote(self.path.split('/')[-1])
                config.ollama_model = model_name
                save_config()  # Save settings
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'model': model_name}).encode())
            elif self.path == '/api/get_visualization':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                viz_data = {
                    'show_tracking': config.show_tracking,
                    'enable_ball_trail': config.enable_ball_trail,
                    'trail_length': config.trail_length,
                    'trail_style': config.trail_style
                }
                self.wfile.write(json.dumps(viz_data).encode())
            elif self.path.startswith('/api/set_visualization/'):
                import urllib.parse
                params = self.path.split('/')[-1]
                param_dict = dict(urllib.parse.parse_qsl(params))
                
                if 'show_tracking' in param_dict:
                    config.show_tracking = param_dict['show_tracking'].lower() == 'true'
                if 'enable_ball_trail' in param_dict:
                    config.enable_ball_trail = param_dict['enable_ball_trail'].lower() == 'true'
                if 'trail_length' in param_dict:
                    config.trail_length = int(param_dict['trail_length'])
                if 'trail_style' in param_dict:
                    config.trail_style = param_dict['trail_style']
                
                save_config()  # Save settings
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode())
            elif self.path == '/api/get_ai_provider':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'ai_provider': config.ai_provider}).encode())
            elif self.path.startswith('/api/set_ai_provider/'):
                provider = self.path.split('/')[-1]
                print(f"[API] set_ai_provider called with provider: '{provider}'")
                if provider in ['echo_ai', 'pollinations']:
                    config.ai_provider = provider
                    
                    # Reinitialize echo_ai with the new provider
                    if provider == 'pollinations':
                        echo_ai = PollinationsAI(
                            model=config.pollinations_model,
                            voice=config.pollinations_voice,
                            max_tokens=config.echo_max_tokens
                        )
                        print(f"[API] Initialized PollinationsAI with model: {config.pollinations_model}")
                    else:  # echo_ai
                        echo_ai = EchoAI(
                            enabled=True,
                            temperature=0.8,
                            max_tokens=config.echo_max_tokens,
                            ollama_model=config.ollama_model,
                            personality=config.echo_personality
                        )
                        print(f"[API] Initialized EchoAI with model: {config.ollama_model}")
                    
                    # Set personality for both providers
                    if config.custom_character_prompt:
                        echo_ai.system_prompt = config.custom_character_prompt
                        print(f"[API] Using custom character prompt")
                    else:
                        if hasattr(echo_ai, 'set_personality'):
                            echo_ai.set_personality(config.echo_personality, game_mode=getattr(config, 'game_mode', 'rocket_league'))
                            print(f"[API] Set personality to: {config.echo_personality}")
                    
                    save_config()  # Save settings
                    print(f"[API] AI provider changed to: {provider}")
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success', 'ai_provider': provider}).encode())
                else:
                    print(f"[API] Invalid AI provider: '{provider}'")
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Invalid AI provider'}).encode())
            elif self.path == '/api/get_pollinations_models':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'models': POLLINATIONS_MODELS}).encode())
            elif self.path.startswith('/api/set_pollinations_model/'):
                import urllib.parse
                model_name = urllib.parse.unquote(self.path.split('/')[-1])
                model_names = [model["name"] for model in POLLINATIONS_MODELS]
                if model_name in model_names:
                    config.pollinations_model = model_name
                    save_config()  # Save settings
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success', 'model': model_name}).encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Invalid Pollinations model'}).encode())
            elif self.path == '/api/get_pollinations_voices':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'voices': POLLINATIONS_VOICES}).encode())
            elif self.path.startswith('/api/set_pollinations_voice/'):
                import urllib.parse
                voice_name = urllib.parse.unquote(self.path.split('/')[-1])
                if voice_name in POLLINATIONS_VOICES:
                    config.pollinations_voice = voice_name
                    save_config()  # Save settings
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success', 'voice': voice_name}).encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Invalid Pollinations voice'}).encode())
            elif self.path == '/api/get_nhl_teams':
                # Return list of NHL teams
                nhl_teams = sorted(list(TEAM_ABBR_MAP.keys()))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'teams': nhl_teams}).encode())
            
            # ============================================================
            #                   DISPLAY SOURCE API
            # ============================================================
            elif self.path.startswith('/api/set_display_source/'):
                import urllib.parse
                source = urllib.parse.unquote(self.path.split('/')[-1])
                config.display_source = source
                save_config()
                print(f"[API] Display source changed to: {source}")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'source': source}).encode())
            
            elif self.path == '/api/get_display_source':
                source = getattr(config, 'display_source', 'screen')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'source': source}).encode())
            
            # ============================================================
            #                   IMAGE MODE API
            # ============================================================
            elif self.path.startswith('/api/set_image_mode/'):
                import urllib.parse
                enabled = self.path.split('/')[-1] == 'true'
                config.image_mode_enabled = enabled
                save_config()
                print(f"[API] Image mode: {'ENABLED' if enabled else 'DISABLED'}")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'enabled': enabled}).encode())
            
            elif self.path == '/api/get_image_mode':
                enabled = getattr(config, 'image_mode_enabled', False)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'enabled': enabled}).encode())
            
            else:
                self.send_response(404)
                self.end_headers()
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.end_headers()
        
        def do_POST(self):
            """Handle POST requests for text chat"""
            print(f"[API DEBUG] POST request received: {self.path}", flush=True)
            global echo_ai, ollama_lock, overlay_caption_text, overlay_state, overlay_last_update
            
            if self.path == '/api/open_highlights':
                # Open highlights folder in file explorer
                try:
                    highlights_dir = os.path.abspath('highlights')
                    if not os.path.exists(highlights_dir):
                        os.makedirs(highlights_dir)
                    
                    # Open folder based on OS
                    import platform
                    if platform.system() == 'Windows':
                        os.startfile(highlights_dir)
                    elif platform.system() == 'Darwin':  # macOS
                        os.system(f'open "{highlights_dir}"')
                    else:  # Linux
                        os.system(f'xdg-open "{highlights_dir}"')
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'success'}).encode())
                except Exception as e:
                    print(f"[API] Error opening highlights folder: {e}")
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
                return
            
            elif self.path == '/api/text_chat':
                print("[API DEBUG] Matched /api/text_chat endpoint", flush=True)
                try:
                    # Read POST data
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    
                    message = data.get('message', '').strip()
                    print(f"[API DEBUG] Received message: '{message}'", flush=True)
                    
                    if not message:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': 'No message provided'}).encode())
                        return
                    
                    # 🔥 Detect content box commands (show image, play video, open webpage)
                    text_lower = message.lower()
                    handled_command = False
                    
                    # SHOW IMAGE / PICTURE
                    if ("show" in text_lower or "display" in text_lower or "let me see" in text_lower) and ("image" in text_lower or "picture" in text_lower or "photo" in text_lower or "pic" in text_lower):
                        # Extract subject - try multiple patterns
                        subject = "a cool image"
                        if " of " in text_lower:
                            subject = message.split(" of ", 1)[-1].strip()
                        elif " about " in text_lower:
                            subject = message.split(" about ", 1)[-1].strip()
                        else:
                            # Extract words between command and image/pic/picture/photo
                            # e.g., "show me a cat pic" -> extract "a cat"
                            import re
                            match = re.search(r'(?:show|display|let me see)(?:\s+me)?\s+(?:a|an)?\s+(.+?)(?:\s+(?:image|picture|photo|pic))', text_lower)
                            if match:
                                subject = match.group(1).strip()
                        
                        # Use Pollinations AI image generation (same format as image mode)
                        # Open in main window to bypass Electron security restrictions
                        import urllib.parse
                        encoded_prompt = urllib.parse.quote(subject)
                        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
                        show_external_url(image_url, f"🖼️ {subject.capitalize()}")
                        print(f"[CONTENT BOX] Opening image in browser: {subject}")
                        handled_command = True
                    
                    # PLAY VIDEO
                    elif ("play" in text_lower or "show" in text_lower or "watch" in text_lower) and "video" in text_lower:
                        # Extract topic after "about" or "of"
                        topic = "space"
                        if " about " in text_lower:
                            topic = message.split(" about ", 1)[-1].strip()
                        elif " of " in text_lower:
                            topic = message.split(" of ", 1)[-1].strip()
                        
                        # Use YouTube embed with search list to auto-play first result
                        video_url = f"https://www.youtube.com/embed?listType=search&list={topic.replace(' ', '+')}&autoplay=1"
                        show_external_url(video_url, f"🎥 {topic.capitalize()}")
                        print(f"[CONTENT BOX] Auto-playing video: {topic}")
                        handled_command = True
                    
                    # OPEN WEBSITE
                    elif "open" in text_lower or "go to" in text_lower or "launch" in text_lower:
                        # Map common site names
                        site_map = {
                            "youtube": "https://youtube.com",
                            "google": "https://google.com",
                            "twitter": "https://twitter.com",
                            "reddit": "https://reddit.com",
                            "twitch": "https://twitch.tv",
                            "github": "https://github.com",
                            "discord": "https://discord.com",
                            "spotify": "https://spotify.com"
                        }
                        
                        for site_name, site_url in site_map.items():
                            if site_name in text_lower:
                                show_webpage(site_url, f"🌐 {site_name.capitalize()}")
                                print(f"[CONTENT BOX] Opening {site_name}")
                                handled_command = True
                                break
                    
                    # SEARCH / GOOGLE (only for explicit "search google for X" or "google X")
                    elif ("search google" in text_lower or (text_lower.startswith("google ") and len(text_lower) > 7)):
                        # Extract query
                        query = message
                        if "search google for" in text_lower:
                            query = message.split("search google for", 1)[-1].strip()
                        elif text_lower.startswith("google "):
                            query = message[7:].strip()
                        
                        # Open Google search in external browser
                        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                        show_external_url(search_url, f"🔍 Search: {query}")
                        print(f"[CONTENT BOX] Searching Google: {query}")
                        handled_command = True
                    
                    # If a command was handled, the AI should acknowledge it
                    if handled_command:
                        print(f"[TEXT CHAT] Content box command detected and executed")
                    
                    # AUTO-SEARCH for questions needing real-time data
                    if not handled_command:
                        print(f"[AUTO SEARCH] Checking for real-time data needs...", flush=True)
                        needs_search = False
                        search_triggers = [
                            ("who won", ""), ("what's the score", ""), ("who's winning", ""),
                            ("what's the weather", ""), ("how's the weather", ""),
                            ("what time", ""), ("when is", ""), ("when does", ""), ("when do", ""), ("when will", ""),
                            ("who is playing", ""), ("who are playing", ""), ("who plays", ""),
                            ("latest", "news"), ("recent", "news"), ("current", ""),
                            ("stock price", ""), ("how much is", ""),
                            ("what is the score", ""), ("did", "win")
                        ]
                        
                        for trigger1, trigger2 in search_triggers:
                            if trigger1 in text_lower:
                                if not trigger2 or trigger2 in text_lower:
                                    needs_search = True
                                    print(f"[AUTO SEARCH] Trigger matched: '{trigger1}' + '{trigger2}'", flush=True)
                                    break
                        
                        if needs_search:
                            print(f"[AUTO SEARCH] Fetching answer for: {message}", flush=True)
                            # Fetch answer from Google and give to AI as context
                            try:
                                import requests
                                from bs4 import BeautifulSoup
                                
                                search_url = f"https://www.google.com/search?q={message.replace(' ', '+')}"
                                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                                print(f"[AUTO SEARCH] Fetching from Google...", flush=True)
                                response = requests.get(search_url, headers=headers, timeout=5)
                                soup = BeautifulSoup(response.text, 'html.parser')
                                
                                # Try to find answer - look for multiple indicators
                                answer = None
                                candidates = []
                                
                                print(f"[AUTO SEARCH DEBUG] Searching for answers in HTML...", flush=True)
                                
                                # Detect question type and use appropriate keywords
                                is_schedule_question = any(q in text_lower for q in ['when do', 'when is', 'when does', 'when will', 'what time'])
                                is_score_question = any(q in text_lower for q in ['who won', 'what score', 'who\'s winning', 'did', 'win'])
                                
                                if is_schedule_question:
                                    print(f"[AUTO SEARCH DEBUG] Schedule question detected - looking for dates/times", flush=True)
                                    
                                    # Look for schedule info: dates, times, "tonight", "tomorrow", etc.
                                    schedule_keywords = [
                                        'tonight', 'tomorrow', 'today', 'next', 'monday', 'tuesday', 'wednesday', 
                                        'thursday', 'friday', 'saturday', 'sunday', 'jan', 'feb', 'mar', 'apr', 
                                        'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'p.m.', 'pm', 'a.m.', 'am',
                                        ':', 'est', 'et', 'pst', 'pt', 'mst', 'mt', 'cst', 'ct'
                                    ]
                                    
                                    elements_checked = 0
                                    for element in soup.find_all(['div', 'span', 'td', 'li', 'h3', 'p']):
                                        text = element.get_text(strip=True)
                                        elements_checked += 1
                                        if text and 5 < len(text) < 200:
                                            text_lower_check = text.lower()
                                            # Check for date/time patterns
                                            if any(keyword in text_lower_check for keyword in schedule_keywords):
                                                # Must have numbers (dates/times)
                                                if any(char.isdigit() for char in text):
                                                    candidates.append(text)
                                                    if len(candidates) <= 3:
                                                        print(f"[AUTO SEARCH DEBUG] Found schedule candidate: {text[:100]}", flush=True)
                                    
                                    print(f"[AUTO SEARCH DEBUG] Checked {elements_checked} HTML elements", flush=True)
                                else:
                                    print(f"[AUTO SEARCH DEBUG] Score/result question detected - looking for game results", flush=True)
                                    # Look for score info: "won", "score", "final", etc.
                                    for element in soup.find_all(['div', 'span', 'td', 'li']):
                                        text = element.get_text(strip=True)
                                        if text and 10 < len(text) < 300:
                                            text_lower_check = text.lower()
                                            # Score patterns: "5-3", "5 - 3", "scored 5", "won 7-2"
                                            if any(keyword in text_lower_check for keyword in [
                                                'won', 'win', 'score', 'final', 'defeated', 'beat', 'victory',
                                                '-', 'to', 'vs', 'points', 'goals', 'game'
                                            ]):
                                                # Check if it has numbers (scores)
                                                if any(char.isdigit() for char in text):
                                                    candidates.append(text)
                                
                                print(f"[AUTO SEARCH DEBUG] Found {len(candidates)} candidate answers", flush=True)
                                if len(candidates) > 0 and len(candidates) <= 5:
                                    # Print first few candidates to see what we're getting
                                    for i, cand in enumerate(candidates[:5]):
                                        print(f"[AUTO SEARCH DEBUG]   Candidate {i+1}: {cand[:80]}...", flush=True)
                                
                                # Take the best candidate (shortest one with numbers)
                                if candidates:
                                    answer = min(candidates, key=len)
                                    print(f"[AUTO SEARCH] ✓ Found answer: {answer}", flush=True)
                                    print(f"[AUTO SEARCH] ✓ Using answer in AI prompt (NOT opening browser)", flush=True)
                                    # Give answer to AI as context - NO browser window!
                                    original_message = message
                                    message = f"The user asked: '{original_message}'\n\nHere's what I found: {answer}\n\nTell them the answer in your personality (1-2 sentences, include the score if present)."
                                else:
                                    print(f"[AUTO SEARCH] ✗ No answer found, opening browser", flush=True)
                                    search_url = f"https://www.google.com/search?q={message.replace(' ', '+')}"
                                    show_external_url(search_url, f"🔍 {message}")
                                    message = f"{message} (I've opened a search for you - I couldn't extract the answer automatically)"
                            except Exception as e:
                                print(f"[AUTO SEARCH] ✗ Error: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                                # Don't open browser on error, just let AI respond
                        else:
                            print(f"[AUTO SEARCH] No trigger matched - regular AI response", flush=True)
                    
                    # Use the AI provider stored in echo_ai (which could be PollinationsAI or EchoAI)
                    response = None
                    speaker = config.echo_personality or "AI"
                    
                    print(f"[API DEBUG] echo_ai exists: {echo_ai is not None}, ai_provider: {config.ai_provider}")
                    print(f"[API DEBUG] echo_ai type: {type(echo_ai).__name__}")
                    
                    if echo_ai:
                        # Check the actual type of echo_ai object
                        is_pollinations = type(echo_ai).__name__ == 'PollinationsAI'

                        # Update AI settings before each response (in case user changed it in UI)
                        if is_pollinations:
                            # PollinationsAI has set_model and set_voice methods
                            if hasattr(echo_ai, 'set_model'):
                                echo_ai.set_model(config.pollinations_model)
                            if hasattr(echo_ai, 'set_voice'):
                                echo_ai.set_voice(config.pollinations_voice)
                            print(f"[TEXT CHAT] Using Pollinations ({config.pollinations_model})")
                            # Apply personality to Pollinations too!
                            if config.custom_character_prompt:
                                print(f"[TEXT CHAT] Using custom character")
                                echo_ai.system_prompt = config.custom_character_prompt
                            else:
                                print(f"[TEXT CHAT] Loading personality: {config.echo_personality}")
                                if hasattr(echo_ai, 'set_personality'):
                                    echo_ai.set_personality(config.echo_personality, game_mode=getattr(config, 'game_mode', 'rocket_league'))

                            # --- Chat memory for PollinationsAI ---
                            # Use a simple in-memory buffer for last 5 exchanges (user/AI)
                            if not hasattr(echo_ai, '_chat_history'):
                                echo_ai._chat_history = []
                            # Append user message
                            echo_ai._chat_history.append({'role': 'user', 'content': message})
                            # Only keep last 10 (5 exchanges)
                            echo_ai._chat_history = echo_ai._chat_history[-10:]
                            # Build history string
                            history_str = ''
                            for entry in echo_ai._chat_history[:-1]:
                                who = 'User' if entry['role'] == 'user' else 'AI'
                                history_str += f"{who}: {entry['content']}\n"

                            # Get AI response with history context
                            with ollama_lock:
                                response = echo_ai.generate(message, context={'history': history_str} if history_str else None)
                            # Append AI response to history
                            echo_ai._chat_history.append({'role': 'ai', 'content': response})
                            # Only keep last 10
                            echo_ai._chat_history = echo_ai._chat_history[-10:]
                        else:
                            # Use Echo AI (Ollama)
                            if config.custom_character_prompt:
                                print(f"[TEXT CHAT] Using custom character")
                                echo_ai.system_prompt = config.custom_character_prompt
                            else:
                                print(f"[TEXT CHAT] Loading personality: {config.echo_personality}")
                                if hasattr(echo_ai, 'set_personality'):
                                    echo_ai.set_personality(config.echo_personality, game_mode=getattr(config, 'game_mode', 'rocket_league'))
                            # Get AI response with lock
                            with ollama_lock:
                                response = echo_ai.chat(message)
                    else:
                        response = "AI is not initialized."
                        speaker = "System"
                    
                    if response:
                        global overlay_image_url, overlay_last_update
                        print(f"[TEXT CHAT] User: {message}")
                        print(f"[TEXT CHAT] {speaker}: {response}")
                        # Start non-blocking TTS playback for chat responses (if TTS available)
                        try:
                            if response and (KOKORO_AVAILABLE or EDGE_TTS_AVAILABLE or getattr(config, 'tts_engine', '') == 'pollinations'):
                                _threading.Thread(target=lambda: speak(response, is_chat_response=True), daemon=True).start()
                                print('[API] TTS playback started in background for chat response')
                        except Exception as tts_err:
                            print(f"[API] Failed to start TTS for chat response: {tts_err}")

                        # --- Art Generation for Text Chat (background thread) ---
                        if getattr(config, 'image_mode_enabled', False):
                            def generate_image_async(user_message, ai_response):
                                try:
                                    import urllib.parse
                                    extract_prompt = f"""Based on this conversation:\nUser: {user_message}\nAI: {ai_response}\n\nGenerate a creative, vivid image prompt (1-2 sentences) for an AI image generator. Focus on visual concepts, scenes, art styles, colors, and mood. Be specific and artistic.\n\nImage prompt:"""
                                    image_prompt = None
                                    if echo_ai:
                                        with ollama_lock:
                                            image_prompt = echo_ai.chat(extract_prompt)
                                        if not image_prompt:
                                            image_prompt = f"Abstract art inspired by: {user_message[:100]}"
                                        image_prompt = image_prompt.strip('"\'').split('\n')[0].strip()
                                        print(f"[TEXT CHAT] [IMAGE] Auto-generated prompt: {image_prompt}")
                                        encoded_prompt = urllib.parse.quote(image_prompt)
                                        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
                                        global overlay_image_url, overlay_last_update
                                        overlay_image_url = image_url
                                        overlay_last_update = time.time()
                                        print(f"[TEXT CHAT] [IMAGE] ✓ Image sent to overlay: {image_url}")
                                except Exception as img_error:
                                    print(f"[TEXT CHAT] [IMAGE] Error generating image: {img_error}")

                            import threading
                            threading.Thread(target=generate_image_async, args=(message, response), daemon=True).start()
                    else:
                        response = "No response from AI."
                        speaker = "System"
                    
                    # Include sentence timings if available (for sentence-by-sentence display)
                    sentences = None
                    if overlay_caption_text and isinstance(overlay_caption_text, dict):
                        sentences = overlay_caption_text.get('sentences')
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'response': response,
                        'speaker': speaker,
                        'sentences': sentences  # Include sentence timing for sequential display
                    }).encode())
                    
                except Exception as e:
                    print(f"[API] Error in text_chat: {e}")
                    import traceback
                    traceback.print_exc()
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
            
            # ============================================================
            #                   IMAGE GENERATION API
            # ============================================================
            elif self.path == '/api/generate_image':
                print("[API] Image generation requested")
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    
                    prompt = data.get('prompt', '').strip()
                    
                    if not prompt:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': 'No prompt provided'}).encode())
                        return
                    
                    print(f"[IMAGE] Generating with prompt: {prompt}")
                    
                    # Generate image using Pollinations
                    import urllib.parse
                    encoded_prompt = urllib.parse.quote(prompt)
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1920&height=1080&nologo=true&enhance=true&private=true"
                    
                    print(f"[IMAGE] ✓ Image URL: {image_url}")
                    
                    # Save image to disk
                    saved_path = save_generated_image(image_url, prompt)
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'success',
                        'image_url': image_url,
                        'prompt': prompt,
                        'saved_path': saved_path
                    }).encode())
                    
                except Exception as e:
                    print(f"[API] Error in generate_image: {e}")
                    import traceback
                    traceback.print_exc()
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
            
            elif self.path == '/api/auto_generate_image':
                print("[API] Auto-generating image from conversation")
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    
                    user_message = data.get('user_message', '').strip()
                    ai_response = data.get('ai_response', '').strip()
                    
                    # Use AI to extract visual concepts and generate image prompt
                    # Create a prompt that asks the AI to describe a visual scene
                    extract_prompt = f"""Based on this conversation:
User: {user_message}
AI: {ai_response}

Generate a creative, vivid image prompt (1-2 sentences) for an AI image generator. Focus on visual concepts, scenes, art styles, colors, and mood. Be specific and artistic.

Image prompt:"""
                    
                    # Get AI to generate the image prompt
                    image_prompt = None
                    if echo_ai:
                        with ollama_lock:
                            image_prompt = echo_ai.chat(extract_prompt)
                    
                    if not image_prompt:
                        # Fallback: use keywords from conversation
                        image_prompt = f"Abstract art inspired by: {user_message[:100]}"
                    
                    # Clean up the prompt (remove quotes, extra text)
                    image_prompt = image_prompt.strip('"\'').split('\n')[0].strip()
                    
                    print(f"[IMAGE] Auto-generated prompt: {image_prompt}")
                    
                    # Generate image using Pollinations
                    import urllib.parse
                    encoded_prompt = urllib.parse.quote(image_prompt)
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1920&height=1080&nologo=true&enhance=true&private=true"
                    
                    print(f"[IMAGE] ✓ Auto-generated image URL: {image_url}")
                    
                    # Save image to disk
                    saved_path = save_generated_image(image_url, image_prompt)
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'success',
                        'image_url': image_url,
                        'prompt': image_prompt,
                        'saved_path': saved_path
                    }).encode())
                    
                except Exception as e:
                    print(f"[API] Error in auto_generate_image: {e}")
                    import traceback
                    traceback.print_exc()
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
            
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())
        
        def log_message(self, format, *args):
            # Suppress default logging
            pass
    
    # ============ TWITCH CHAT INTEGRATION ============
    twitch_monitor = None
    # ============ TIKTOK CHAT INTEGRATION ============
    tiktok_monitor = None
    tiktok_user_last_roast = {}  # track last roast timestamp per username

    def handle_tiktok_event(event):
        """Handle TikTok chat messages and events"""
        global echo_ai, overlay_caption_text, overlay_last_update, tiktok_user_last_roast

        event_type = event.get('type')
        username = event.get('username')
        message = event.get('message', '')

        if not username or not message:
            return

        # Only handle chat events
        if event_type == 'chat' and getattr(config, 'tiktok_enabled', False):
            # Only react to mentions or questions
            if event.get('is_mention') and getattr(config, 'tiktok_roast_on_mention', True):
                # Rate limit per user
                try:
                    last = tiktok_user_last_roast.get(username, 0)
                    cooldown = 60.0 / max(1, int(getattr(config, 'tiktok_max_roasts_per_minute', 3)))
                    if time.time() - last < cooldown:
                        print(f"[TIKTOK] Rate limit: skipping roast for {username}")
                        return
                    tiktok_user_last_roast[username] = time.time()
                except Exception:
                    tiktok_user_last_roast[username] = time.time()

                # Show caption first
                user_message_display = f"💬 @{username}: {message}"
                overlay_caption_text = {"text": user_message_display, "speaker": "TikTok Chat"}
                overlay_last_update = time.time()
                print(f"[TIKTOK] 💬 CHAT: {username}: {message}")
                time.sleep(1.5)

                clean_message = message.lstrip('?').strip()
                if not clean_message:
                    clean_message = "said hello"

                # Build roast prompt - instruct model to avoid hateful/abusive content
                if echo_ai:
                    prompt = (f"{username} asks for a roast: {clean_message}. You are the 'Late Night Roaster' personality. "
                              "Deliver a short, playful, witty roast addressed to the viewer by name (do NOT use slurs or hate speech, avoid threats, sexual/violent content). "
                              "Keep it under 35 words and stay in-character.")
                    try:
                        response = echo_ai.generate(prompt, {"event": "tiktok_roast", "username": username})
                    except Exception as e:
                        print(f"[TIKTOK] AI generation failed: {e}")
                        response = None
                else:
                    response = f"Alright @{username}, court ordered roast: you're the human version of a participation trophy."

                if response:
                    print(f"[TIKTOK] 📣 Roast for @{username}: {response}")
                    speak(response, is_chat_response=True)

            elif event.get('is_question'):
                # Handle questions (answer briefly in character)
                user_message_display = f"💬 @{username}: {message}"
                overlay_caption_text = {"text": user_message_display, "speaker": "TikTok Chat"}
                overlay_last_update = time.time()
                print(f"[TIKTOK] ❓ QUESTION: {username}: {message}")
                time.sleep(1.2)

                clean_message = message.lstrip('?').strip()
                if not clean_message:
                    clean_message = "asked something"

                if echo_ai:
                    prompt = f"{username} asks: {clean_message}. Respond briefly in character (Late Night Roaster)."
                    try:
                        response = echo_ai.generate(prompt, {"event": "tiktok_question", "username": username})
                    except Exception as e:
                        print(f"[TIKTOK] AI generation failed: {e}")
                        response = None
                else:
                    response = f"@{username}: {clean_message}"

                if response:
                    print(f"[TIKTOK] 📣 Response: {response}")
                    speak(response, is_chat_response=True)
    
    def handle_twitch_event(event):
        """Handle Twitch chat messages and events"""
        global echo_ai, overlay_caption_text, overlay_last_update, overlay_image_url
        
        event_type = event['type']
        username = event['username']
        message = event.get('message', '')
        is_priority = event.get('priority', False)
        
        # Generate response based on event type
        response = None
        
        if event_type == 'subscription':
            if getattr(config, 'twitch_respond_to_subs', True):
                response = f"Thanks for subscribing, {username}! You're awesome!"
                print(f"[TWITCH] 🎁 SUB: {username}")
        
        elif event_type == 'gift_sub':
            if getattr(config, 'twitch_respond_to_subs', True):
                response = message  # Already formatted
                print(f"[TWITCH] 🎁 GIFT: {username}")
        
        elif event_type == 'raid':
            response = message  # Already formatted
            print(f"[TWITCH] 🚀 RAID: {username}")
        
        elif event_type == 'chat':
            if getattr(config, 'twitch_respond_to_chat', True):
                # Only respond to questions or mentions
                if event['is_question'] or event['is_mention']:
                    # Show user's message in captions first (caption only, no speech)
                    user_message_display = f"💬 {username}: {message}"
                    overlay_caption_text = {"text": user_message_display, "speaker": "Twitch Chat"}
                    overlay_last_update = time.time()
                    print(f"[TWITCH] 💬 CHAT: {username}: {message}")
                    time.sleep(1.5)  # Brief pause to show user message
                    
                    # Clean the message (remove command prefixes like ?)
                    clean_message = message.lstrip('?').strip()
                    if not clean_message:
                        clean_message = "said hello"
                    
                    # Use EchoAI to generate personality-driven response
                    if echo_ai:
                        prompt = f"{username} says: {clean_message}. Respond briefly in character."
                        response = echo_ai.generate(prompt, {"event": "chat_question", "username": username})
                    else:
                        response = f"Hey {username}! {clean_message}"
                    print(f"[TWITCH] 📣 Response: {response}")
                    
                    # Auto-generate image if image mode is enabled (check config)
                    if getattr(config, 'image_mode_enabled', False):
                        print(f"[TWITCH] 🎨 Image mode enabled - generating art for conversation")
                        try:
                            import urllib.parse
                            # Create image generation prompt from conversation
                            extract_prompt = f"""Based on this Twitch chat conversation:
{username}: {clean_message}
AI: {response}

Generate a creative, vivid image prompt (1-2 sentences) for an AI image generator. Focus on visual concepts, scenes, art styles, colors, and mood. Be specific and artistic.

Image prompt:"""
                            
                            # Get AI to generate the image prompt
                            image_prompt = None
                            if echo_ai:
                                with ollama_lock:
                                    image_prompt = echo_ai.chat(extract_prompt)
                            
                            if not image_prompt:
                                # Fallback: use keywords from conversation
                                image_prompt = f"Abstract art inspired by: {clean_message[:100]}"
                            
                            # Clean up the prompt
                            image_prompt = image_prompt.strip('"\'').split('\n')[0].strip()
                            
                            print(f"[TWITCH] [IMAGE] Auto-generated prompt: {image_prompt}")
                            
                            # Generate image URL (Pollinations)
                            encoded_prompt = urllib.parse.quote(image_prompt)
                            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1920&height=1080&nologo=true&enhance=true&private=true"
                            
                            print(f"[TWITCH] [IMAGE] ✓ Image URL: {image_url}")
                            
                            # Save image to disk
                            saved_path = save_generated_image(image_url, image_prompt)
                            if saved_path:
                                print(f"[TWITCH] [IMAGE] ✓ Saved: {saved_path}")
                            
                            # Store in global so overlay can fetch it via polling
                            # (overlay_image_url, overlay_last_update already declared as global above)
                            overlay_image_url = image_url
                            overlay_last_update = time.time()  # Trigger overlay update
                            
                        except Exception as img_error:
                            print(f"[TWITCH] [IMAGE] Failed to generate: {img_error}")
        
        # Speak the response with priority and mark as chat response - disabled to prevent errors
        channel = getattr(config, 'twitch_channel', '').strip()
        oauth = getattr(config, 'twitch_oauth', '').strip()
        
        if channel and oauth:
            try:
                from twitch_chat import TwitchChatMonitor
                twitch_monitor = TwitchChatMonitor(channel, oauth, handle_twitch_event)
                # Set personality name for mention detection
                twitch_monitor.personality_name = config.echo_personality or "plaix"
                twitch_monitor.start()
                print(f"[TWITCH] ✅ Monitoring chat for #{channel} (personality: {twitch_monitor.personality_name})")
            except Exception as e:
                print(f"[TWITCH ERROR] Failed to start: {e}")
        else:
            print("[TWITCH] ⚠️ Enabled but missing channel or OAuth token")

        # ================= TIKTOK CHAT INTEGRATION =================
        try:
            if getattr(config, 'tiktok_enabled', False):
                t_username = getattr(config, 'tiktok_username', '').strip()
                t_cookie = getattr(config, 'tiktok_cookie', '').strip()
                if t_username:
                    try:
                        from tiktok_chat import TikTokChatMonitor
                        # Create monitor and start
                        tiktok_monitor = TikTokChatMonitor(t_username, t_cookie, handle_tiktok_event)
                        tiktok_monitor.personality_name = config.echo_personality or "plaix"
                        started = tiktok_monitor.start()
                        if started:
                            print(f"[TIKTOK] ✅ Monitoring chat for @{t_username} (personality: {tiktok_monitor.personality_name})")
                    except Exception as e:
                        print(f"[TIKTOK ERROR] Failed to start: {e}")
                else:
                    print("[TIKTOK] ⚠️ Enabled but missing username or credentials")
        except Exception as e:
            print(f"[TIKTOK] Startup check failed: {e}")
    
    # ============ END TWITCH INTEGRATION ============
    
    # Start overlay API server on port 7862 in background thread
    # If the Flask-based API was already started earlier, skip starting the simple HTTP server
    def run_overlay_api():
        try:
            api_server = HTTPServer(('127.0.0.1', 7862), OverlayAPIHandler)
            print("[API] Overlay status server running on http://127.0.0.1:7862/api/overlay_status")
            api_server.serve_forever()
        except OSError as e:
            print(f"[API] Could not start simple HTTP server (port may be in use): {e}")
        except Exception as e:
            print(f"[API] Unexpected error starting overlay API: {e}")
            import traceback
            traceback.print_exc()
    
    if getattr(globals(), '_api_thread_started', False):
        print("[API] Flask API already started on port 7862 - skipping simple HTTP server to avoid conflicts")
    else:
        api_thread = threading.Thread(target=run_overlay_api, daemon=True)
        api_thread.start()
    
    # Auto-trigger overlay window creation
    print("[STARTUP] 🎭 Creating avatar overlay window...")
    try:
        with open('overlay.flag', 'w') as f:
            f.write('create_overlay')
        print("[STARTUP] ✓ Overlay flag created - Electron will create window")
    except Exception as e:
        print(f"[STARTUP] Warning: Could not create overlay flag: {e}")
    
    print(f"[STARTUP] OVERLAY API: http://127.0.0.1:7862/api/overlay_status")
    print(f"[STARTUP] 🎭 Avatar overlay starting...")
    print(f"[STARTUP] All settings can be adjusted in the overlay's settings panel")
    
    # Keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Shutting down PLAIX...")
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Shutting down PLAIX...")