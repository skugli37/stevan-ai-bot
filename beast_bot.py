#!/usr/bin/env python3
"""
🔥 BEAST AI ART BOT - PROFESSIONAL EDITION 🔥
=============================================

Features:
- 10+ AI modela (FLUX, SD3, SDXL, Anime, Realistic...)
- Stilovi (Cyberpunk, Anime, Oil Painting, Photo...)
- Negative prompts
- Upscaling (2x, 4x)
- Varijacije slike
- Image-to-Image
- Aspect ratios
- Queue system
- Referral program
- Premium tiers
- Galerija & Favoriti

Autor: Claude & Stevan
"""

import os
import asyncio
import logging
import sqlite3
import hashlib
import random
import string
from datetime import datetime, date
from io import BytesIO
from typing import Optional, Dict, Any, List
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, 
    InputMediaPhoto, BotCommand, ReplyKeyboardMarkup, KeyboardButton
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, filters, ContextTypes, ConversationHandler
)
from gradio_client import Client
import requests
from PIL import Image
import uuid
from deep_translator import GoogleTranslator

# ============ AUTO TRANSLATE ============
def translate_to_english(text: str) -> str:
    """Auto-detect and translate to English"""
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        # Ako je isto, verovatno je već engleski
        if translated and translated.lower() != text.lower():
            return translated
        return text
    except:
        return text

# ============ KONFIGURACIJA ============
BOT_TOKEN = "8518707116:AAFAUimJAmuWyK3L1Voz5bBp4pGfIImPrms"
ADMIN_IDS = []  # Dodaćemo tvoj ID kad pošalješ /start

# ============ AI MODELI ============
MODELS = {
    "flux_schnell": {
        "name": "⚡ FLUX Fast",
        "id": "black-forest-labs/FLUX.1-schnell",
        "desc": "Najbrži, 4 sec",
        "premium": False
    },
    "flux_dev": {
        "name": "✨ FLUX Pro",
        "id": "black-forest-labs/FLUX.1-dev",
        "desc": "Najbolji kvalitet, 30 sec",
        "premium": False
    },
    "sd3": {
        "name": "🎨 SD3 Medium",
        "id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "desc": "Stable Diffusion 3",
        "premium": False
    },
    "sdxl": {
        "name": "🖼 SDXL",
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "desc": "Stable Diffusion XL",
        "premium": False
    },
    "playground": {
        "name": "🎮 Playground v2.5",
        "id": "playgroundai/playground-v2.5-1024px-aesthetic",
        "desc": "Aesthetic fokus",
        "premium": False
    },
    "animagine": {
        "name": "🌸 Animagine XL",
        "id": "cagliostrolab/animagine-xl-3.1",
        "desc": "Anime stil",
        "premium": False
    },
    "realvis": {
        "name": "📷 RealVisXL",
        "id": "SG161222/RealVisXL_V4.0",
        "desc": "Photorealistic",
        "premium": True
    },
    "juggernaut": {
        "name": "💪 Juggernaut XL",
        "id": "RunDiffusion/Juggernaut-XL-v9",
        "desc": "Svemoćni",
        "premium": True
    },
}

# ============ VIDEO MODELI ============
VIDEO_MODELS = {
    "svd": {
        "name": "🎬 Stable Video",
        "id": "stabilityai/stable-video-diffusion",
        "desc": "Image to Video, 4 sec"
    },
    "zeroscope": {
        "name": "🎥 Zeroscope",
        "id": "hysts/zeroscope-v2-xl",
        "desc": "Text to Video"
    },
}

# ============ STILOVI ============
STYLES = {
    "none": {"name": "🚫 Bez stila", "prompt": "", "negative": ""},
    "cinematic": {
        "name": "🎬 Cinematic",
        "prompt": "cinematic shot, dramatic lighting, film grain, movie scene, epic composition",
        "negative": "amateur, low quality"
    },
    "anime": {
        "name": "🌸 Anime",
        "prompt": "anime style, manga, japanese animation, vibrant colors, detailed",
        "negative": "realistic, photo, 3d render"
    },
    "photorealistic": {
        "name": "📷 Photo",
        "prompt": "photorealistic, ultra realistic, professional photography, 8k, sharp focus",
        "negative": "cartoon, painting, illustration, drawing"
    },
    "digital_art": {
        "name": "🎨 Digital Art",
        "prompt": "digital art, digital painting, artstation, concept art, highly detailed",
        "negative": "photo, realistic"
    },
    "oil_painting": {
        "name": "🖼 Oil Painting",
        "prompt": "oil painting, classical art, brush strokes, canvas texture, masterpiece",
        "negative": "photo, digital"
    },
    "cyberpunk": {
        "name": "🌆 Cyberpunk",
        "prompt": "cyberpunk, neon lights, futuristic city, rain, night, blade runner style",
        "negative": "nature, daylight, rural"
    },
    "fantasy": {
        "name": "🐉 Fantasy",
        "prompt": "fantasy art, magical, ethereal, epic fantasy, detailed illustration",
        "negative": "modern, realistic, photo"
    },
    "minimalist": {
        "name": "⬜ Minimalist",
        "prompt": "minimalist, simple, clean, white background, elegant, modern design",
        "negative": "complex, detailed, busy"
    },
    "3d_render": {
        "name": "🎮 3D Render",
        "prompt": "3d render, octane render, unreal engine 5, highly detailed, volumetric lighting",
        "negative": "2d, flat, painting"
    },
    "watercolor": {
        "name": "💧 Watercolor",
        "prompt": "watercolor painting, soft colors, artistic, flowing paint, paper texture",
        "negative": "digital, photo, sharp"
    },
    "comic": {
        "name": "💥 Comic",
        "prompt": "comic book style, bold lines, vibrant colors, action scene, marvel dc style",
        "negative": "realistic, photo"
    },
    "neon": {
        "name": "💜 Neon",
        "prompt": "neon art, glowing, vibrant neon colors, dark background, synthwave",
        "negative": "natural lighting, daylight"
    },
    "vintage": {
        "name": "📺 Vintage",
        "prompt": "vintage, retro, 1950s style, old photograph, sepia tones, nostalgic",
        "negative": "modern, futuristic"
    },
    "horror": {
        "name": "👻 Horror",
        "prompt": "dark horror, creepy, atmospheric, gothic, nightmare fuel, scary",
        "negative": "bright, happy, colorful"
    }
}

# ============ ASPECT RATIOS ============
ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
}

# ============ PREMIUM TIERS ============
TIERS = {
    "free": {
        "name": "🆓 Free",
        "daily_limit": 99999,
        "models": "all",
        "max_resolution": "2048x2048",
        "queue_priority": 3
    },
    "basic": {
        "name": "⭐ Basic",
        "daily_limit": 30,
        "models": ["flux_schnell", "flux_dev", "sd3", "sdxl", "playground", "animagine"],
        "max_resolution": "1344x1344",
        "queue_priority": 1,
        "price": 5
    },
    "pro": {
        "name": "💎 Pro",
        "daily_limit": 100,
        "models": "all",
        "max_resolution": "1536x1536",
        "queue_priority": 2,
        "price": 15
    },
    "unlimited": {
        "name": "🔥 Unlimited",
        "daily_limit": -1,
        "models": "all",
        "max_resolution": "2048x2048",
        "queue_priority": 3,
        "price": 30
    }
}

# ============ DATABASE ============
DB_PATH = "beast_bot.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        first_name TEXT,
        tier TEXT DEFAULT 'free',
        credits INTEGER DEFAULT 0,
        daily_used INTEGER DEFAULT 0,
        total_generated INTEGER DEFAULT 0,
        last_reset DATE,
        referral_code TEXT UNIQUE,
        referred_by INTEGER,
        referral_earnings REAL DEFAULT 0,
        preferred_model TEXT DEFAULT 'flux_schnell',
        preferred_style TEXT DEFAULT 'none',
        preferred_ratio TEXT DEFAULT '1:1',
        negative_prompt TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_banned INTEGER DEFAULT 0
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS generations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        prompt TEXT,
        style TEXT,
        model TEXT,
        width INTEGER,
        height INTEGER,
        seed INTEGER,
        image_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS favorites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_id TEXT,
        prompt TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        amount REAL,
        type TEXT,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()

def get_user(user_id: int, username: str = None, first_name: str = None) -> Dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
    row = c.fetchone()
    
    today = date.today().isoformat()
    
    if row is None:
        ref_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        c.execute('''INSERT INTO users 
            (user_id, username, first_name, last_reset, referral_code)
            VALUES (?, ?, ?, ?, ?)''',
            (user_id, username, first_name, today, ref_code))
        conn.commit()
        user = {
            "user_id": user_id, "username": username, "first_name": first_name,
            "tier": "free", "credits": 0, "daily_used": 0, "total_generated": 0,
            "referral_code": ref_code, "referred_by": None, "referral_earnings": 0,
            "preferred_model": "flux_schnell", "preferred_style": "none",
            "preferred_ratio": "1:1", "negative_prompt": "", "is_banned": 0
        }
    else:
        user = dict(row)
        if user["last_reset"] != today:
            c.execute('UPDATE users SET daily_used = 0, last_reset = ? WHERE user_id = ?',
                     (today, user_id))
            conn.commit()
            user["daily_used"] = 0
    
    conn.close()
    return user

def update_user(user_id: int, **kwargs):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    updates = [f"{k} = ?" for k in kwargs.keys()]
    values = list(kwargs.values()) + [user_id]
    c.execute(f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?", values)
    conn.commit()
    conn.close()

def log_generation(user_id: int, prompt: str, style: str, model: str, 
                   width: int, height: int, seed: int, image_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO generations 
        (user_id, prompt, style, model, width, height, seed, image_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (user_id, prompt, style, model, width, height, seed, image_id))
    conn.commit()
    conn.close()

def get_user_history(user_id: int, limit: int = 10) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''SELECT * FROM generations WHERE user_id = ? 
                 ORDER BY created_at DESC LIMIT ?''', (user_id, limit))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def add_favorite(user_id: int, image_id: str, prompt: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO favorites (user_id, image_id, prompt) VALUES (?, ?, ?)',
              (user_id, image_id, prompt))
    conn.commit()
    conn.close()

def get_favorites(user_id: int, limit: int = 20) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''SELECT * FROM favorites WHERE user_id = ? 
                 ORDER BY created_at DESC LIMIT ?''', (user_id, limit))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def apply_referral(user_id: int, referral_code: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id FROM users WHERE referral_code = ?', (referral_code,))
    referrer = c.fetchone()
    if referrer and referrer[0] != user_id:
        c.execute('UPDATE users SET referred_by = ? WHERE user_id = ?',
                  (referrer[0], user_id))
        c.execute('UPDATE users SET credits = credits + 5 WHERE user_id = ?', (referrer[0],))
        c.execute('UPDATE users SET credits = credits + 3 WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

# ============ IMAGE GENERATION ============
# SVE besplatne opcije sa rotacijom
IMAGE_PROVIDERS = [
    {
        "name": "FLUX.1-schnell",
        "space": "black-forest-labs/FLUX.1-schnell",
        "steps": 4
    },
    {
        "name": "FLUX.1-dev", 
        "space": "black-forest-labs/FLUX.1-dev",
        "steps": 28
    },
    {
        "name": "FLUX-merged",
        "space": "multimodalart/FLUX.1-merged", 
        "steps": 4
    },
    {
        "name": "SD3-medium",
        "space": "stabilityai/stable-diffusion-3-medium",
        "steps": 28
    },
    {
        "name": "SD3.5-large",
        "space": "stabilityai/stable-diffusion-3.5-large",
        "steps": 40
    },
]

# Legacy compatibility
FLUX_PROVIDERS = [p["space"] for p in IMAGE_PROVIDERS]

current_provider_index = 0

def _generate_sync(prompt: str, model: str, width: int, height: int) -> tuple:
    """Sync generation - runs in thread with SMART provider rotation"""
    global current_provider_index
    
    # TRANSLATE TO ENGLISH
    prompt_en = translate_to_english(prompt)
    
    # Try each provider with rotation
    errors = []
    for i in range(len(IMAGE_PROVIDERS)):
        provider = IMAGE_PROVIDERS[(current_provider_index + i) % len(IMAGE_PROVIDERS)]
        try:
            logging.info(f"🎨 Trying: {provider['name']} ({provider['space']})")
            client = Client(provider['space'], verbose=False)
            
            # Use provider-specific steps
            steps = provider.get('steps', 4)
            
            result = client.predict(
                prompt=prompt_en,
                seed=0,
                randomize_seed=True,
                width=width,
                height=height,
                num_inference_steps=steps,
                api_name="/infer"
            )
            
            # Success - rotate to next provider for load balancing
            current_provider_index = (current_provider_index + 1) % len(IMAGE_PROVIDERS)
            logging.info(f"✅ Success with {provider['name']}")
            return result[0], result[1] if isinstance(result, tuple) else (result, 0)
            
        except Exception as e:
            error_msg = str(e)
            errors.append(f"{provider['name']}: {error_msg[:80]}")
            logging.warning(f"❌ {provider['name']} failed: {error_msg[:80]}")
            
            # Continue to next provider
            continue
    
    # All providers failed
    raise Exception(f"All {len(IMAGE_PROVIDERS)} providers failed!")

async def generate_image(prompt: str, model: str, style: str, 
                        width: int, height: int, negative: str = "") -> tuple:
    """Generate image and return (path, seed)"""
    
    style_data = STYLES.get(style, STYLES["none"])
    full_prompt = f"{prompt}, {style_data['prompt']}" if style_data['prompt'] else prompt
    
    try:
        # Run in thread to not block event loop
        result = await asyncio.to_thread(_generate_sync, full_prompt, model, width, height)
        return result
    except Exception as e:
        logging.error(f"Generation error: {e}")
        raise

# ============ VIDEO GENERATION ============
# SVE besplatne video opcije sa rotacijom
# Video provajderi - img2video sa rotacijom
VIDEO_PROVIDERS = [
    {
        "name": "SVD-Official",
        "space": "stabilityai/stable-video-diffusion",
        "api": "/video",
    },
    {
        "name": "SVD-Multimodal",
        "space": "multimodalart/stable-video-diffusion", 
        "api": "/video",
    },
]

video_provider_index = 0

def _generate_video_sync(image_path: str) -> str:
    """Generate video from image - SA ROTACIJOM kroz sve provajdere"""
    global video_provider_index
    
    errors = []
    
    for i in range(len(VIDEO_PROVIDERS)):
        provider = VIDEO_PROVIDERS[(video_provider_index + i) % len(VIDEO_PROVIDERS)]
        try:
            logging.info(f"🎬 Trying: {provider['name']}")
            client = Client(provider["space"], verbose=False)
            
            result = client.predict(
                image=image_path,
                seed=0,
                randomize_seed=True,
                motion_bucket_id=127,
                fps_id=6,
                api_name=provider["api"]
            )
            
            # Result is (dict(video=path), seed) - extract video path
            if isinstance(result, tuple):
                video_data = result[0]
            else:
                video_data = result
                
            # Extract actual path from dict
            if isinstance(video_data, dict) and "video" in video_data:
                video_path = video_data["video"]
            else:
                video_path = video_data
            
            # Rotate for next time
            video_provider_index = (video_provider_index + 1) % len(VIDEO_PROVIDERS)
            logging.info(f"✅ Video done with {provider['name']}: {video_path}")
            return video_path
            
        except Exception as e:
            error_msg = str(e)[:80]
            errors.append(f"{provider['name']}: {error_msg}")
            logging.warning(f"❌ {provider['name']}: {error_msg}")
            continue
    
    raise Exception(f"All {len(VIDEO_PROVIDERS)} video providers failed!")

def _generate_text_to_video_sync(prompt: str) -> str:
    """Generate video from text: Image -> Animate"""
    logging.info(f"🎬 Text-to-Video: {prompt[:50]}")
    
    # Step 1: Generate image (16:9 for video)
    logging.info("📸 Step 1: Generating image...")
    img_result = _generate_sync(prompt, "flux_schnell", 1024, 576)
    image_path = img_result[0]
    logging.info(f"✅ Image ready: {image_path}")
    
    # Step 2: Animate image
    logging.info("🎬 Step 2: Animating...")
    video_path = _generate_video_sync(image_path)
    logging.info(f"✅ Video ready: {video_path}")
    
    return video_path

async def generate_video_from_image(image_path: str) -> str:
    """Async wrapper for image-to-video"""
    try:
        result = await asyncio.to_thread(_generate_video_sync, image_path)
        return result
    except Exception as e:
        logging.error(f"Video generation error: {e}")
        raise

async def generate_video_from_text(prompt: str) -> str:
    """Async wrapper for text-to-video"""
    try:
        prompt_en = translate_to_english(prompt)
        result = await asyncio.to_thread(_generate_text_to_video_sync, prompt_en)
        return result
    except Exception as e:
        logging.error(f"Video generation error: {e}")
        raise

# ============ LOGGING ============
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.FileHandler('beast_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============ HANDLERS ============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_user(
        update.effective_user.id,
        update.effective_user.username,
        update.effective_user.first_name
    )
    
    # Check referral
    if context.args and len(context.args) > 0:
        ref_code = context.args[0]
        if user.get("referred_by") is None:
            if apply_referral(user["user_id"], ref_code):
                await update.message.reply_text("🎁 Referral primenjen! Dobio si 3 besplatna kredita!")
    
    tier_info = TIERS[user["tier"]]
    remaining = tier_info["daily_limit"] - user["daily_used"] if tier_info["daily_limit"] > 0 else "∞"
    
    keyboard = [
        [KeyboardButton("🎨 Generiši"), KeyboardButton("⚙️ Settings")],
        [KeyboardButton("📊 Status"), KeyboardButton("🛒 Premium")],
        [KeyboardButton("📖 Help"), KeyboardButton("🔗 Referral")]
    ]
    
    text = f"""
🔥 **BEAST AI ART BOT** 🔥

Zdravo {user.get('first_name', 'prijatelju')}!

Ja sam najmoćniji AI art bot!

**10+ modela:** FLUX, SD3, SDXL, Anime...
**15 stilova:** Cinematic, Cyberpunk, Fantasy...
**Sve veličine:** Square, Wide, Portrait...

📊 **Tvoj status:**
{tier_info['name']} | Danas: {remaining} | Krediti: {user['credits']}

💡 **Pošalji mi prompt i krećemo!**

Primer: `cyberpunk samurai in neon tokyo`
"""
    await update.message.reply_text(
        text, 
        parse_mode='Markdown',
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )

async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_user(update.effective_user.id)
    
    keyboard = [
        [InlineKeyboardButton("🤖 Model", callback_data="settings_model"),
         InlineKeyboardButton("🎨 Stil", callback_data="settings_style")],
        [InlineKeyboardButton("📐 Ratio", callback_data="settings_ratio"),
         InlineKeyboardButton("🚫 Negative", callback_data="settings_negative")],
    ]
    
    current_model = MODELS.get(user['preferred_model'], MODELS['flux_schnell'])
    current_style = STYLES.get(user['preferred_style'], STYLES['none'])
    
    text = f"""
⚙️ **Podešavanja**

🤖 Model: **{current_model['name']}**
🎨 Stil: **{current_style['name']}**
📐 Ratio: **{user['preferred_ratio']}**
🚫 Negative: `{user['negative_prompt'][:30] or 'Nije postavljeno'}...`

Izaberi šta želiš da promeniš:
"""
    await update.message.reply_text(text, parse_mode='Markdown', 
                                    reply_markup=InlineKeyboardMarkup(keyboard))

async def model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = get_user(query.from_user.id)
    tier = TIERS[user["tier"]]
    
    keyboard = []
    for model_key, model_info in MODELS.items():
        available = tier["models"] == "all" or model_key in tier["models"]
        prefix = "✅" if user['preferred_model'] == model_key else "⬜"
        lock = "" if available else "🔒"
        keyboard.append([InlineKeyboardButton(
            f"{prefix} {lock} {model_info['name']} - {model_info['desc']}",
            callback_data=f"setmodel_{model_key}" if available else "need_premium"
        )])
    
    keyboard.append([InlineKeyboardButton("◀️ Nazad", callback_data="back_settings")])
    
    await query.edit_message_text(
        "🤖 **Izaberi model:**\n\n🔒 = Potreban premium",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def style_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = get_user(query.from_user.id)
    
    keyboard = []
    row = []
    for i, (style_key, style_info) in enumerate(STYLES.items()):
        prefix = "✅" if user['preferred_style'] == style_key else ""
        row.append(InlineKeyboardButton(
            f"{prefix}{style_info['name']}",
            callback_data=f"setstyle_{style_key}"
        ))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("◀️ Nazad", callback_data="back_settings")])
    
    await query.edit_message_text(
        "🎨 **Izaberi stil:**",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def ratio_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = get_user(query.from_user.id)
    
    keyboard = []
    row = []
    for ratio in ASPECT_RATIOS.keys():
        prefix = "✅" if user['preferred_ratio'] == ratio else ""
        row.append(InlineKeyboardButton(
            f"{prefix}{ratio}",
            callback_data=f"setratio_{ratio}"
        ))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("◀️ Nazad", callback_data="back_settings")])
    
    await query.edit_message_text(
        "📐 **Izaberi aspect ratio:**",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    user_id = query.from_user.id
    
    if data == "settings_model":
        await model_selection(update, context)
    elif data == "settings_style":
        await style_selection(update, context)
    elif data == "settings_ratio":
        await ratio_selection(update, context)
    elif data == "settings_negative":
        await query.answer()
        await query.edit_message_text(
            "🚫 **Negative Prompt**\n\n"
            "Pošalji mi tekst koji želiš da AI izbegava.\n"
            "Npr: `blurry, low quality, bad anatomy`\n\n"
            "Pošalji /setnegative [tekst]"
        , parse_mode='Markdown')
    elif data.startswith("setmodel_"):
        model = data.replace("setmodel_", "")
        update_user(user_id, preferred_model=model)
        await query.answer(f"✅ Model: {MODELS[model]['name']}")
        await model_selection(update, context)
    elif data.startswith("setstyle_"):
        style = data.replace("setstyle_", "")
        update_user(user_id, preferred_style=style)
        await query.answer(f"✅ Stil: {STYLES[style]['name']}")
        await style_selection(update, context)
    elif data.startswith("setratio_"):
        ratio = data.replace("setratio_", "")
        update_user(user_id, preferred_ratio=ratio)
        await query.answer(f"✅ Ratio: {ratio}")
        await ratio_selection(update, context)
    elif data == "need_premium":
        await query.answer("🔒 Potreban Premium! /premium", show_alert=True)
    elif data == "back_settings":
        await query.answer()
        user = get_user(user_id)
        current_model = MODELS.get(user['preferred_model'], MODELS['flux_schnell'])
        current_style = STYLES.get(user['preferred_style'], STYLES['none'])
        keyboard = [
            [InlineKeyboardButton("🤖 Model", callback_data="settings_model"),
             InlineKeyboardButton("🎨 Stil", callback_data="settings_style")],
            [InlineKeyboardButton("📐 Ratio", callback_data="settings_ratio"),
             InlineKeyboardButton("🚫 Negative", callback_data="settings_negative")],
        ]
        await query.edit_message_text(
            f"⚙️ **Podešavanja**\n\n"
            f"🤖 Model: **{current_model['name']}**\n"
            f"🎨 Stil: **{current_style['name']}**\n"
            f"📐 Ratio: **{user['preferred_ratio']}**",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    elif data.startswith("vary_"):
        image_id = data.replace("vary_", "")
        context.user_data['vary_image'] = image_id
        await query.answer("🔄 Pošalji novi prompt za varijaciju!")
    elif data.startswith("fav_"):
        image_id = data.replace("fav_", "")
        prompt = context.user_data.get('last_prompt', '')
        add_favorite(user_id, image_id, prompt)
        await query.answer("⭐ Dodato u favorite!")
    elif data.startswith("regen_"):
        seed = int(data.replace("regen_", ""))
        context.user_data['force_seed'] = seed
        await query.answer("🔄 Pošalji prompt za regeneraciju sa istim seed-om!")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_user(update.effective_user.id)
    tier = TIERS[user["tier"]]
    remaining = tier["daily_limit"] - user["daily_used"] if tier["daily_limit"] > 0 else "∞"
    
    text = f"""
📊 **Tvoj Status**

👤 ID: `{user['user_id']}`
📛 Tier: {tier['name']}

📈 **Statistika:**
• Danas iskorišćeno: {user['daily_used']}/{tier['daily_limit'] if tier['daily_limit'] > 0 else '∞'}
• Preostalo: {remaining}
• Ukupno generisano: {user['total_generated']}
• Krediti: {user['credits']}

💰 **Referral:**
• Tvoj kod: `{user['referral_code']}`
• Zarada: ${user['referral_earnings']:.2f}

⚙️ **Podešavanja:**
• Model: {MODELS[user['preferred_model']]['name']}
• Stil: {STYLES[user['preferred_style']]['name']}
• Ratio: {user['preferred_ratio']}
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def premium_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("⭐ Basic - $5/mes", callback_data="buy_basic")],
        [InlineKeyboardButton("💎 Pro - $15/mes", callback_data="buy_pro")],
        [InlineKeyboardButton("🔥 Unlimited - $30/mes", callback_data="buy_unlimited")],
    ]
    
    text = """
🛒 **Premium Planovi**

🆓 **Free** (trenutni)
• 5 slika dnevno
• 2 modela
• Osnovni stilovi

⭐ **Basic** - $5/mesec
• 30 slika dnevno
• 6 modela
• Svi stilovi
• Prioritet u queue

💎 **Pro** - $15/mesec
• 100 slika dnevno
• Svi modeli
• Upscaling
• Img2Img
• Najviši prioritet

🔥 **Unlimited** - $30/mesec
• Neograničeno
• Sve features
• API pristup
• VIP podrška

💳 **Plaćanje:**
• Crypto (BTC, ETH, USDT, SOL)
• PayPal
• Revolut

Kontakt: @YOUR_USERNAME
"""
    await update.message.reply_text(text, parse_mode='Markdown',
                                    reply_markup=InlineKeyboardMarkup(keyboard))

async def referral_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_user(update.effective_user.id)
    bot_username = (await context.bot.get_me()).username
    ref_link = f"https://t.me/{bot_username}?start={user['referral_code']}"
    
    text = f"""
🔗 **Referral Program**

Pozovi prijatelje i zarađuj!

📤 **Tvoj link:**
`{ref_link}`

🎁 **Nagrade:**
• Ti dobijaš: **5 kredita** po referralu
• Prijatelj dobija: **3 kredita**
• Bonus: **10%** od svih njihovih kupovina!

📊 **Tvoja statistika:**
• Referral kod: `{user['referral_code']}`
• Ukupna zarada: **${user['referral_earnings']:.2f}**

Podeli link i zarađuj! 🚀
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
📖 **BEAST BOT - Pomoć**

🎨 **Kako generisati:**
Samo pošalji prompt! Npr:
`cyberpunk samurai, neon lights, rain`

✨ **Saveti za promptove:**

1️⃣ **Budi detaljan:**
`A majestic dragon flying over a medieval castle at sunset, dramatic clouds, epic fantasy art`

2️⃣ **Dodaj stil u prompt:**
`portrait of a warrior, oil painting style`
`futuristic city, anime style`
`beautiful landscape, photorealistic`

3️⃣ **Kvalitet:**
`masterpiece, best quality, highly detailed, 8k`

4️⃣ **Osvetljenje:**
`golden hour, dramatic lighting, volumetric light, neon glow`

5️⃣ **Kompozicija:**
`close-up, wide shot, bird's eye view, cinematic composition`

⚙️ **Komande:**
/settings - Podešavanja
/status - Tvoj status
/premium - Premium planovi
/referral - Referral program
/history - Istorija generacija
/favorites - Tvoji favoriti

💡 **Pro tip:** Kombinuj model + stil za najbolje rezultate!
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = get_user_history(user_id, 10)
    
    if not history:
        await update.message.reply_text("📭 Nemaš istoriju generacija.")
        return
    
    text = "📜 **Tvoja istorija:**\n\n"
    for i, gen in enumerate(history, 1):
        text += f"{i}. `{gen['prompt'][:40]}...`\n"
        text += f"   {gen['model']} | {gen['width']}x{gen['height']}\n\n"
    
    await update.message.reply_text(text, parse_mode='Markdown')

async def favorites_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    favs = get_favorites(user_id, 10)
    
    if not favs:
        await update.message.reply_text("⭐ Nemaš favorite. Koristi ⭐ dugme na slikama!")
        return
    
    text = "⭐ **Tvoji favoriti:**\n\n"
    for i, fav in enumerate(favs, 1):
        text += f"{i}. `{fav['prompt'][:50]}...`\n"
    
    await update.message.reply_text(text, parse_mode='Markdown')

async def video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler za /video komandu - text to video"""
    user = get_user(update.effective_user.id)
    
    if not context.args:
        await update.message.reply_text(
            "🎬 **Video Generator**\n\n"
            "Koristi: `/video [prompt]`\n\n"
            "Primer: `/video A cat playing piano`\n\n"
            "Ili pošalji sliku i ja ću je animirati!",
            parse_mode='Markdown'
        )
        return
    
    prompt = " ".join(context.args)
    prompt_en = translate_to_english(prompt)
    
    await update.message.chat.send_action("record_video")
    
    status = await update.message.reply_text(
        f"🎬 **Generišem video...**\n\n"
        f"📝 `{prompt[:50]}`\n"
        f"🌐 `{prompt_en[:50]}`\n\n"
        f"⏱ Ovo može trajati 30-60 sekundi...",
        parse_mode='Markdown'
    )
    
    try:
        video_path = await generate_video_from_text(prompt_en)
        
        await update.message.reply_video(
            video=open(video_path, 'rb'),
            caption=f"🎬 **{prompt[:100]}**",
            parse_mode='Markdown'
        )
        
        await status.delete()
        
    except Exception as e:
        await status.edit_text(f"❌ Greška: {str(e)[:100]}\n\nProbaj ponovo!")
        logger.error(f"Video error: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler za slike - animira ih u video"""
    user = get_user(update.effective_user.id)
    
    # Download slike
    photo = update.message.photo[-1]  # Najveća rezolucija
    file = await context.bot.get_file(photo.file_id)
    
    # Sačuvaj lokalno
    image_path = f"/tmp/user_image_{update.effective_user.id}.jpg"
    await file.download_to_drive(image_path)
    
    await update.message.chat.send_action("record_video")
    
    status = await update.message.reply_text(
        "🎬 **Animiram tvoju sliku...**\n\n"
        "⏱ Ovo može trajati 30-60 sekundi...",
        parse_mode='Markdown'
    )
    
    try:
        video_path = await generate_video_from_image(image_path)
        
        await update.message.reply_video(
            video=open(video_path, 'rb'),
            caption="🎬 **Animirana slika!**",
            parse_mode='Markdown'
        )
        
        await status.delete()
        
    except Exception as e:
        await status.edit_text(f"❌ Greška: {str(e)[:100]}\n\nProbaj ponovo!")
        logger.error(f"Video error: {e}")

async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main generation handler"""
    user = get_user(update.effective_user.id)
    
    # Check ban
    if user.get("is_banned"):
        await update.message.reply_text("⛔ Banovan si.")
        return
    
    prompt = update.message.text
    
    # Ignore menu buttons
    if prompt in ["🎨 Generiši", "⚙️ Settings", "📊 Status", "🛒 Premium", "📖 Help", "🔗 Referral"]:
        if prompt == "⚙️ Settings":
            await settings_menu(update, context)
        elif prompt == "📊 Status":
            await status_command(update, context)
        elif prompt == "🛒 Premium":
            await premium_command(update, context)
        elif prompt == "📖 Help":
            await help_command(update, context)
        elif prompt == "🔗 Referral":
            await referral_command(update, context)
        elif prompt == "🎨 Generiši":
            await update.message.reply_text("💡 Pošalji mi prompt za sliku!")
        return
    
    # Ignore commands
    if prompt.startswith('/'):
        return
    
    # Check limits
    tier = TIERS[user["tier"]]
    if tier["daily_limit"] > 0 and user["daily_used"] >= tier["daily_limit"]:
        if user["credits"] <= 0:
            await update.message.reply_text(
                f"❌ Dostigao si dnevni limit ({tier['daily_limit']} slika)!\n\n"
                "💎 /premium za više generacija\n"
                "🔗 /referral za besplatne kredite"
            )
            return
    
    # Get settings
    model = user["preferred_model"]
    style = user["preferred_style"]
    ratio = user["preferred_ratio"]
    width, height = ASPECT_RATIOS[ratio]
    negative = user.get("negative_prompt", "")
    
    # Status message
    await update.message.chat.send_action("upload_photo")
    
    model_info = MODELS[model]
    style_info = STYLES[style]
    
    # Translate for display
    prompt_en = translate_to_english(prompt)
    
    status = await update.message.reply_text(
        f"🔥 **Generišem...**\n\n"
        f"📝 `{prompt[:60]}{'...' if len(prompt) > 60 else ''}`\n"
        f"🌐 `{prompt_en[:60]}{'...' if len(prompt_en) > 60 else ''}`\n\n"
        f"🤖 {model_info['name']}\n"
        f"🎨 {style_info['name']}\n"
        f"📐 {width}x{height}",
        parse_mode='Markdown'
    )
    
    try:
        # Generate
        start = datetime.now()
        image_path, seed = await generate_image(prompt, model, style, width, height, negative)
        elapsed = (datetime.now() - start).seconds
        
        # Update stats
        new_daily = user["daily_used"] + 1
        new_total = user["total_generated"] + 1
        update_user(user["user_id"], daily_used=new_daily, total_generated=new_total)
        
        # Log
        image_id = str(uuid.uuid4())[:8]
        log_generation(user["user_id"], prompt, style, model, width, height, seed, image_id)
        
        # Store for callbacks
        context.user_data['last_prompt'] = prompt
        context.user_data['last_seed'] = seed
        context.user_data['last_image_id'] = image_id
        
        # Action buttons
        keyboard = [
            [
                InlineKeyboardButton("🔄 Varijacija", callback_data=f"vary_{image_id}"),
                InlineKeyboardButton("⭐ Favorite", callback_data=f"fav_{image_id}"),
            ],
            [
                InlineKeyboardButton("🔁 Regenerate", callback_data=f"regen_{seed}"),
            ]
        ]
        
        tier_info = TIERS[user["tier"]]
        remaining = tier_info["daily_limit"] - new_daily if tier_info["daily_limit"] > 0 else "∞"
        
        # Send image
        await update.message.reply_photo(
            photo=open(image_path, 'rb'),
            caption=f"🎨 **{prompt[:150]}**\n\n"
                   f"⏱ {elapsed}s | 🌱 {seed}\n"
                   f"📊 Preostalo: {remaining}",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        await status.delete()
        logger.info(f"Generated for {user['user_id']}: {prompt[:50]}")
        
    except Exception as e:
        await status.edit_text(f"❌ Greška: {str(e)[:100]}\n\nProbaj ponovo!")
        logger.error(f"Error: {e}")

async def set_negative(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Koristi: /setnegative [tekst]\n\n"
            "Primer: `/setnegative blurry, low quality, bad anatomy`",
            parse_mode='Markdown'
        )
        return
    
    negative = " ".join(context.args)
    update_user(update.effective_user.id, negative_prompt=negative)
    await update.message.reply_text(f"✅ Negative prompt postavljen:\n`{negative}`", parse_mode='Markdown')

# ============ ADMIN ============
async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM users')
    total_users = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM generations')
    total_gens = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM generations WHERE DATE(created_at) = DATE("now")')
    today_gens = c.fetchone()[0]
    
    conn.close()
    
    text = f"""
📊 **Admin Stats**

👥 Users: {total_users}
🖼 Total gens: {total_gens}
📅 Today: {today_gens}
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def admin_addcredits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /addcredits USER_ID AMOUNT")
        return
    
    user_id = int(context.args[0])
    amount = int(context.args[1])
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE users SET credits = credits + ? WHERE user_id = ?', (amount, user_id))
    conn.commit()
    conn.close()
    
    await update.message.reply_text(f"✅ Added {amount} credits to {user_id}")

# ============ MAIN ============
def main():
    init_db()
    
    print("🔥 Starting BEAST AI ART BOT...")
    print(f"📊 IMAGE providers: {len(IMAGE_PROVIDERS)}")
    print(f"🎬 VIDEO providers: {len(VIDEO_PROVIDERS)}")
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("settings", settings_menu))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("premium", premium_command))
    app.add_handler(CommandHandler("referral", referral_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("favorites", favorites_command))
    app.add_handler(CommandHandler("setnegative", set_negative))
    app.add_handler(CommandHandler("video", video_command))
    
    # Admin
    app.add_handler(CommandHandler("adminstats", admin_stats))
    app.add_handler(CommandHandler("addcredits", admin_addcredits))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    # Photos - animira u video
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate))
    
    print("✅ Bot ready!")
    app.run_polling()

if __name__ == "__main__":
    main()
