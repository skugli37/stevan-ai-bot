#!/usr/bin/env python3
"""
🔥🔥🔥 STEVAN BEAST AI BOT 🔥🔥🔥
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIKE: 8 FLUX/SD providera sa rotacijom
VIDEO: 6 Video providera sa rotacijom
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Za 100k+ korisnika - Production Ready
"""

import os
import sys
import logging
import asyncio
import random
import time
import json
import tempfile
from datetime import datetime
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, filters, ContextTypes
)
from deep_translator import GoogleTranslator

# ═══════════════════════════════════════════════════════════
# KONFIGURACIJA
# ═══════════════════════════════════════════════════════════

BOT_TOKEN = "8518707116:AAFAUimJAmuWyK3L1Voz5bBp4pGfIImPrms"

# Logging setup
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Thread pool za sync operacije
executor = ThreadPoolExecutor(max_workers=10)

# ═══════════════════════════════════════════════════════════
# IMAGE PROVIDERS - 8 BESPLATNIH IZVORA
# ═══════════════════════════════════════════════════════════

IMAGE_PROVIDERS = [
    {
        "name": "FLUX.1-schnell",
        "space": "black-forest-labs/FLUX.1-schnell",
        "api": "/infer",
        "params": lambda p, w, h: {
            "prompt": p, "seed": 0, "randomize_seed": True,
            "width": w, "height": h, "num_inference_steps": 4
        },
        "result_index": 0,
        "emoji": "⚡"
    },
    {
        "name": "FLUX.1-dev",
        "space": "black-forest-labs/FLUX.1-dev",
        "api": "/infer",
        "params": lambda p, w, h: {
            "prompt": p, "seed": 0, "randomize_seed": True,
            "width": w, "height": h, "guidance_scale": 3.5, 
            "num_inference_steps": 28
        },
        "result_index": 0,
        "emoji": "🎨"
    },
    {
        "name": "SD3-Medium",
        "space": "stabilityai/stable-diffusion-3-medium",
        "api": "/infer",
        "params": lambda p, w, h: {
            "prompt": p, "negative_prompt": "", "seed": 0,
            "randomize_seed": True, "width": w, "height": h,
            "guidance_scale": 5, "num_inference_steps": 28
        },
        "result_index": 0,
        "emoji": "🖼️"
    },
    {
        "name": "FLUX-Merged",
        "space": "multimodalart/FLUX.1-merged",
        "api": "/infer",
        "params": lambda p, w, h: {
            "prompt": p, "seed": 0, "randomize_seed": True,
            "width": w, "height": h, "num_inference_steps": 4
        },
        "result_index": 0,
        "emoji": "🔥"
    },
    {
        "name": "Animagine-XL",
        "space": "Linaqruf/animagine-xl-3.1",
        "api": "/run",
        "params": lambda p, w, h: {
            "prompt": p + ", masterpiece, best quality",
            "negative_prompt": "lowres, bad anatomy",
            "seed": 0, "width": w, "height": h,
            "guidance_scale": 7, "num_inference_steps": 28
        },
        "result_index": 0,
        "emoji": "🎌"
    },
    {
        "name": "RealVisXL",
        "space": "SG161222/RealVisXL_V4.0",
        "api": "/run",
        "params": lambda p, w, h: {
            "prompt": p, "negative_prompt": "cartoon, anime",
            "width": w, "height": h, "guidance_scale": 5,
            "num_inference_steps": 25
        },
        "result_index": 0,
        "emoji": "📸"
    },
    {
        "name": "Playground-v2.5",
        "space": "playgroundai/playground-v2.5",
        "api": "/run",
        "params": lambda p, w, h: {
            "prompt": p, "width": w, "height": h,
            "guidance_scale": 3, "num_inference_steps": 25
        },
        "result_index": 0,
        "emoji": "🎮"
    },
    {
        "name": "PixArt-Sigma",
        "space": "PixArt-alpha/PixArt-Sigma",
        "api": "/infer",
        "params": lambda p, w, h: {
            "prompt": p, "width": w, "height": h,
            "guidance_scale": 4.5, "num_inference_steps": 14
        },
        "result_index": 0,
        "emoji": "✨"
    }
]

# ═══════════════════════════════════════════════════════════
# VIDEO PROVIDERS - 6 BESPLATNIH IZVORA
# ═══════════════════════════════════════════════════════════

VIDEO_PROVIDERS = [
    {
        "name": "Stable Video Diffusion",
        "space": "stabilityai/stable-video-diffusion",
        "type": "img2video",
        "api": "/video",
        "emoji": "🎥"
    }
]

# Statistike providera - Updated 1765122305
provider_stats = {
    "image": {p["name"]: {"success": 0, "fail": 0, "last_fail": 0} for p in IMAGE_PROVIDERS},
    "video": {p["name"]: {"success": 0, "fail": 0, "last_fail": 0} for p in VIDEO_PROVIDERS}
}

# ═══════════════════════════════════════════════════════════
# STILOVI ZA SLIKE
# ═══════════════════════════════════════════════════════════

STYLES = {
    "none": {"name": "🎨 Originalno", "prefix": "", "suffix": ""},
    "cinematic": {
        "name": "🎬 Cinematic",
        "prefix": "cinematic shot, ",
        "suffix": ", dramatic lighting, film grain, anamorphic lens, 8k"
    },
    "anime": {
        "name": "🎌 Anime",
        "prefix": "anime style, ",
        "suffix": ", vibrant colors, detailed anime art, studio ghibli quality"
    },
    "realistic": {
        "name": "📸 Realistic",
        "prefix": "photorealistic, ",
        "suffix": ", ultra detailed, 8k uhd, professional photography, natural lighting"
    },
    "cyberpunk": {
        "name": "🌃 Cyberpunk",
        "prefix": "cyberpunk style, ",
        "suffix": ", neon lights, rain, futuristic city, blade runner aesthetic"
    },
    "fantasy": {
        "name": "🧙 Fantasy",
        "prefix": "fantasy art, ",
        "suffix": ", magical atmosphere, ethereal lighting, epic scale, digital painting"
    },
    "oil_painting": {
        "name": "🖼️ Oil Painting",
        "prefix": "oil painting style, ",
        "suffix": ", classical art, museum quality, brush strokes visible, masterpiece"
    },
    "watercolor": {
        "name": "💧 Watercolor",
        "prefix": "watercolor painting, ",
        "suffix": ", soft colors, artistic, delicate brush strokes, paper texture"
    },
    "3d_render": {
        "name": "🎮 3D Render",
        "prefix": "3d render, ",
        "suffix": ", octane render, unreal engine 5, photorealistic, detailed textures"
    },
    "pixel_art": {
        "name": "👾 Pixel Art",
        "prefix": "pixel art, ",
        "suffix": ", retro game style, 16-bit, detailed pixels, nostalgic"
    },
    "horror": {
        "name": "👻 Horror",
        "prefix": "horror style, ",
        "suffix": ", dark atmosphere, unsettling, creepy lighting, nightmare fuel"
    },
    "comic": {
        "name": "💥 Comic",
        "prefix": "comic book style, ",
        "suffix": ", bold lines, vibrant colors, dynamic composition, marvel dc style"
    },
    "minimalist": {
        "name": "⬜ Minimalist",
        "prefix": "minimalist style, ",
        "suffix": ", simple, clean lines, negative space, modern design"
    },
    "vintage": {
        "name": "📷 Vintage",
        "prefix": "vintage photograph, ",
        "suffix": ", sepia tones, old photo effect, nostalgic, film grain"
    },
    "neon": {
        "name": "💜 Neon",
        "prefix": "neon style, ",
        "suffix": ", glowing neon lights, dark background, synthwave, retrowave"
    }
}

# ═══════════════════════════════════════════════════════════
# ASPECT RATIOS
# ═══════════════════════════════════════════════════════════

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "21:9": (1536, 640),
}

# ═══════════════════════════════════════════════════════════
# USER STATE
# ═══════════════════════════════════════════════════════════

user_state = {}  # user_id -> {style, ratio, last_prompt, ...}

def get_user_state(user_id: int) -> dict:
    if user_id not in user_state:
        user_state[user_id] = {
            "style": "none",
            "ratio": "1:1",
            "last_prompt": None,
            "last_image": None,
            "generation_count": 0
        }
    return user_state[user_id]

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def translate_to_english(text: str) -> str:
    """Prevodi tekst na engleski ako nije"""
    try:
        # Detektuj i prevedi
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated if translated else text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def apply_style(prompt: str, style_key: str) -> str:
    """Primeni stil na prompt"""
    if style_key not in STYLES or style_key == "none":
        return prompt
    
    style = STYLES[style_key]
    return f"{style['prefix']}{prompt}{style['suffix']}"

def get_best_provider(provider_type: str) -> list:
    """Vrati providere sortirane po uspešnosti"""
    providers = IMAGE_PROVIDERS if provider_type == "image" else VIDEO_PROVIDERS
    stats = provider_stats[provider_type]
    
    def score(p):
        s = stats[p["name"]]
        total = s["success"] + s["fail"]
        if total == 0:
            return 1.0  # Novi provider - daj šansu
        
        success_rate = s["success"] / total
        # Penalizuj ako je nedavno failovao
        time_penalty = 1.0
        if s["last_fail"] > 0:
            seconds_since_fail = time.time() - s["last_fail"]
            if seconds_since_fail < 300:  # 5 minuta cooldown
                time_penalty = seconds_since_fail / 300
        
        return success_rate * time_penalty
    
    # Sortiraj po score-u, ali dodaj malo random za load balancing
    sorted_providers = sorted(providers, key=lambda p: score(p) + random.uniform(0, 0.1), reverse=True)
    return sorted_providers

# ═══════════════════════════════════════════════════════════
# IMAGE GENERATION SA ROTACIJOM
# ═══════════════════════════════════════════════════════════

def generate_image_sync(prompt: str, width: int, height: int) -> Tuple[Optional[str], str, str]:
    """
    Generiši sliku sa rotacijom kroz providere
    Returns: (image_path, provider_name, error_message)
    """
    from gradio_client import Client
    
    providers = get_best_provider("image")
    errors = []
    
    for provider in providers:
        try:
            logger.info(f"🎨 Trying {provider['name']}...")
            
            client = Client(provider["space"], verbose=False)
            params = provider["params"](prompt, width, height)
            
            result = client.predict(**params, api_name=provider["api"])
            
            # Izvuci putanju slike
            if isinstance(result, tuple):
                image_path = result[provider.get("result_index", 0)]
            else:
                image_path = result
            
            if image_path and os.path.exists(str(image_path)):
                # Uspeh!
                provider_stats["image"][provider["name"]]["success"] += 1
                logger.info(f"✅ {provider['name']} SUCCESS!")
                return str(image_path), provider["name"], ""
            
        except Exception as e:
            error_msg = str(e)[:100]
            errors.append(f"{provider['name']}: {error_msg}")
            provider_stats["image"][provider["name"]]["fail"] += 1
            provider_stats["image"][provider["name"]]["last_fail"] = time.time()
            logger.warning(f"❌ {provider['name']} failed: {error_msg}")
            continue
    
    return None, "", "\n".join(errors)

async def generate_image(prompt: str, width: int, height: int) -> Tuple[Optional[str], str, str]:
    """Async wrapper za image generation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_image_sync, prompt, width, height)

# ═══════════════════════════════════════════════════════════
# VIDEO GENERATION SA ROTACIJOM
# ═══════════════════════════════════════════════════════════

def generate_video_text2video_sync(prompt: str) -> Tuple[Optional[str], str, str]:
    """Text to Video - Generiše sliku pa animira sa SVD"""
    from gradio_client import Client
    
    errors = []
    
    try:
        # STEP 1: Generiši sliku iz prompta
        logger.info(f"🎨 Step 1: Generating image from prompt...")
        
        image_path, img_provider, img_error = generate_image_sync(prompt, 1024, 576)
        
        if not image_path:
            return None, "", f"Image generation failed: {img_error}"
        
        logger.info(f"✅ Image generated with {img_provider}")
        
        # STEP 2: Animiraj sliku sa SVD
        logger.info(f"🎬 Step 2: Animating with Stable Video Diffusion...")
        
        svd_client = Client("stabilityai/stable-video-diffusion", verbose=False)
        
        result = svd_client.predict(
            image=image_path,
            seed=0,
            randomize_seed=True,
            motion_bucket_id=127,
            fps_id=6,
            api_name="/video"
        )
        
        video_path = result[0] if isinstance(result, (list, tuple)) else result
        
        if video_path and (os.path.exists(str(video_path)) or str(video_path).startswith("http")):
            logger.info(f"✅ Video generated successfully!")
            return str(video_path), f"{img_provider} + SVD", ""
        
        return None, "", "SVD returned no video"
        
    except Exception as e:
        error_msg = str(e)[:200]
        logger.warning(f"❌ T2V failed: {error_msg}")
        return None, "", error_msg

def generate_video_img2video_sync(image_path: str) -> Tuple[Optional[str], str, str]:
    """Image to Video sa SVD"""
    from gradio_client import Client
    
    try:
        logger.info(f"🎥 Animating image with Stable Video Diffusion...")
        
        client = Client("stabilityai/stable-video-diffusion", verbose=False)
        
        result = client.predict(
            image=image_path,
            seed=0,
            randomize_seed=True,
            motion_bucket_id=127,
            fps_id=6,
            api_name="/video"
        )
        
        video_path = result[0] if isinstance(result, (list, tuple)) else result
        
        if video_path and (os.path.exists(str(video_path)) or str(video_path).startswith("http")):
            logger.info(f"✅ SVD I2V SUCCESS!")
            return str(video_path), "Stable Video Diffusion", ""
        
        return None, "", "SVD returned no video"
            
    except Exception as e:
        error_msg = str(e)[:200]
        logger.warning(f"❌ SVD I2V failed: {error_msg}")
        return None, "", error_msg

async def generate_video_t2v(prompt: str) -> Tuple[Optional[str], str, str]:
    """Async Text to Video"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_video_text2video_sync, prompt)

async def generate_video_i2v(image_path: str) -> Tuple[Optional[str], str, str]:
    """Async Image to Video"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_video_img2video_sync, image_path)

# ═══════════════════════════════════════════════════════════
# TELEGRAM HANDLERS
# ═══════════════════════════════════════════════════════════

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    user = update.effective_user
    state = get_user_state(user.id)
    
    welcome = f"""
🔥 **STEVAN BEAST AI BOT** 🔥
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pozdrav {user.first_name}! 👋

🎨 **SLIKE** - Samo pošalji prompt
🎬 **VIDEO** - `/video [prompt]`
🎥 **ANIMIRAJ** - `/animate` (za poslednju sliku)

📐 **Komande:**
• `/style` - Izaberi stil (15 stilova!)
• `/ratio` - Promeni format slike
• `/video [tekst]` - Text to Video
• `/animate` - Animiraj zadnju sliku
• `/stats` - Statistika providera
• `/help` - Pomoć

💡 **Primeri:**
• `Cyberpunk samurai u Tokiju`
• `/video Mačka pleše na kiši`
• `Epic dragon breathing fire`

🌍 Piši na srpskom - automatski prevodim!

**Trenutni stil:** {STYLES[state['style']]['name']}
**Format:** {state['ratio']}
"""
    
    keyboard = [
        [InlineKeyboardButton("🎨 Stilovi", callback_data="menu_styles"),
         InlineKeyboardButton("📐 Format", callback_data="menu_ratio")],
        [InlineKeyboardButton("🎬 Video Help", callback_data="menu_video")]
    ]
    
    await update.message.reply_text(
        welcome,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    help_text = """
📖 **DETALJNO UPUTSTVO**
━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎨 **GENERISANJE SLIKA:**
Samo napiši šta želiš da vidiš!
• `Sunset over mountains, golden hour`
• `Portrait of a cyberpunk girl`
• `Stara kuća na selu` (prevodi se automatski)

🎬 **TEXT TO VIDEO:**
`/video [opis]` - Napravi video od teksta
• `/video A cat dancing in the rain`
• `/video Waves crashing on rocks`

🎥 **IMAGE TO VIDEO:**
`/animate` - Animiraj poslednju generisanu sliku
• Prvo generiši sliku, pa `/animate`

🎨 **STILOVI (15):**
`/style` pa izaberi:
• Cinematic, Anime, Realistic
• Cyberpunk, Fantasy, Horror
• Oil Painting, Watercolor
• 3D Render, Pixel Art, Comic
• Minimalist, Vintage, Neon

📐 **FORMATI:**
`/ratio` pa izaberi:
• 1:1 (Instagram)
• 16:9 (YouTube)
• 9:16 (TikTok/Stories)
• 4:3, 3:4, 21:9

⚡ **TIPS:**
• Budi detaljan u opisu
• Koristi engleski za bolje rezultate
• Probaj razne stilove!
"""
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Prikaži stilove"""
    keyboard = []
    row = []
    for key, style in STYLES.items():
        row.append(InlineKeyboardButton(style["name"], callback_data=f"style_{key}"))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    
    await update.message.reply_text(
        "🎨 **Izaberi stil:**",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def ratio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Prikaži aspect ratio opcije"""
    keyboard = []
    row = []
    for ratio in ASPECT_RATIOS.keys():
        row.append(InlineKeyboardButton(ratio, callback_data=f"ratio_{ratio}"))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    
    await update.message.reply_text(
        "📐 **Izaberi format:**",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Prikaži statistiku providera"""
    text = "📊 **STATISTIKA PROVIDERA**\n\n"
    
    text += "**🎨 IMAGE:**\n"
    for name, stats in provider_stats["image"].items():
        total = stats["success"] + stats["fail"]
        rate = (stats["success"] / total * 100) if total > 0 else 0
        status = "🟢" if rate > 70 else "🟡" if rate > 30 else "🔴"
        text += f"{status} {name}: {stats['success']}/{total} ({rate:.0f}%)\n"
    
    text += "\n**🎬 VIDEO:**\n"
    for name, stats in provider_stats["video"].items():
        total = stats["success"] + stats["fail"]
        rate = (stats["success"] / total * 100) if total > 0 else 0
        status = "🟢" if rate > 70 else "🟡" if rate > 30 else "🔴"
        text += f"{status} {name}: {stats['success']}/{total} ({rate:.0f}%)\n"
    
    await update.message.reply_text(text, parse_mode="Markdown")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    state = get_user_state(user_id)
    data = query.data
    
    if data.startswith("style_"):
        style_key = data.replace("style_", "")
        if style_key in STYLES:
            state["style"] = style_key
            style_info = STYLES[style_key]
            await query.edit_message_text(
                f"✅ **Stil promenjen: {style_info['name']}**\n\n"
                f"Sada pošalji prompt i slika će biti generisana u ovom stilu!\n\n"
                f"📐 Trenutni format: {state['ratio']}",
                parse_mode="Markdown"
            )
    
    elif data.startswith("ratio_"):
        ratio = data.replace("ratio_", "")
        if ratio in ASPECT_RATIOS:
            state["ratio"] = ratio
            await query.edit_message_text(
                f"✅ Format promenjen: **{ratio}**",
                parse_mode="Markdown"
            )
    
    elif data == "menu_styles":
        await style_command(query, context)
    
    elif data == "menu_ratio":
        await ratio_command(query, context)
    
    elif data == "menu_video":
        await query.edit_message_text(
            "🎬 **VIDEO KOMANDE:**\n\n"
            "• `/video [prompt]` - Text to Video\n"
            "• `/animate` - Animiraj poslednju sliku\n\n"
            "**Primeri:**\n"
            "• `/video A tiger running through forest`\n"
            "• `/video Talasi na moru`",
            parse_mode="Markdown"
        )

async def video_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Text to Video command"""
    if not context.args:
        await update.message.reply_text(
            "❓ Korišćenje: `/video [opis videa]`\n\n"
            "Primer: `/video A cat playing piano`",
            parse_mode="Markdown"
        )
        return
    
    prompt = " ".join(context.args)
    user = update.effective_user
    
    # Prevedi ako treba
    translated_prompt = translate_to_english(prompt)
    
    # Status poruka
    status_msg = await update.message.reply_text(
        f"🎬 **Generišem video...**\n\n"
        f"📝 Prompt: _{prompt}_\n"
        f"🔄 Rotiram kroz 4 video providera...\n\n"
        f"⏳ Ovo može potrajati 2-5 minuta...",
        parse_mode="Markdown"
    )
    
    try:
        video_path, provider_name, error = await generate_video_t2v(translated_prompt)
        
        if video_path:
            # Pošalji video
            if video_path.startswith("http"):
                await update.message.reply_video(
                    video=video_path,
                    caption=f"🎬 **Video generisan!**\n\n"
                           f"📝 _{prompt}_\n"
                           f"⚙️ Provider: {provider_name}",
                    parse_mode="Markdown"
                )
            else:
                with open(video_path, 'rb') as f:
                    await update.message.reply_video(
                        video=f,
                        caption=f"🎬 **Video generisan!**\n\n"
                               f"📝 _{prompt}_\n"
                               f"⚙️ Provider: {provider_name}",
                        parse_mode="Markdown"
                    )
            await status_msg.delete()
        else:
            await status_msg.edit_text(
                f"❌ **Video generisanje nije uspelo**\n\n"
                f"Svi provideri su zauzeti ili imaju GPU kvote.\n\n"
                f"💡 Probaj ponovo za par minuta!\n\n"
                f"Greške:\n```\n{error[:500]}\n```",
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Video generation error: {e}")
        await status_msg.edit_text(f"❌ Greška: {str(e)[:200]}")

async def animate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Animiraj poslednju sliku"""
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    
    if not state.get("last_image"):
        await update.message.reply_text(
            "❓ Nema slike za animaciju!\n\n"
            "Prvo generiši sliku, pa onda `/animate`",
            parse_mode="Markdown"
        )
        return
    
    status_msg = await update.message.reply_text(
        "🎥 **Animiram sliku...**\n\n"
        "⏳ Ovo može potrajati 1-3 minuta...",
        parse_mode="Markdown"
    )
    
    try:
        video_path, provider_name, error = await generate_video_i2v(state["last_image"])
        
        if video_path:
            if video_path.startswith("http"):
                await update.message.reply_video(
                    video=video_path,
                    caption=f"🎥 **Slika animirana!**\n⚙️ Provider: {provider_name}",
                    parse_mode="Markdown"
                )
            else:
                with open(video_path, 'rb') as f:
                    await update.message.reply_video(
                        video=f,
                        caption=f"🎥 **Slika animirana!**\n⚙️ Provider: {provider_name}",
                        parse_mode="Markdown"
                    )
            await status_msg.delete()
        else:
            await status_msg.edit_text(
                f"❌ Animacija nije uspela\n\nGreške:\n```\n{error[:300]}\n```",
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Animate error: {e}")
        await status_msg.edit_text(f"❌ Greška: {str(e)[:200]}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages - generisi sliku"""
    if not update.message or not update.message.text:
        return
    
    prompt = update.message.text.strip()
    if not prompt or prompt.startswith("/"):
        return
    
    user = update.effective_user
    state = get_user_state(user.id)
    
    # Prevedi prompt
    translated_prompt = translate_to_english(prompt)
    
    # Primeni stil
    styled_prompt = apply_style(translated_prompt, state["style"])
    
    # Dobij dimenzije
    width, height = ASPECT_RATIOS.get(state["ratio"], (1024, 1024))
    
    # Status poruka
    style_name = STYLES[state["style"]]["name"]
    status_msg = await update.message.reply_text(
        f"🎨 **Generišem sliku...**\n\n"
        f"📝 Prompt: _{prompt}_\n"
        f"🎭 Stil: {style_name}\n"
        f"📐 Format: {state['ratio']}\n"
        f"🔄 Rotiram kroz 8 providera...",
        parse_mode="Markdown"
    )
    
    try:
        image_path, provider_name, error = await generate_image(styled_prompt, width, height)
        
        if image_path:
            # Sačuvaj za animaciju
            state["last_image"] = image_path
            state["last_prompt"] = prompt
            state["generation_count"] += 1
            
            # Keyboard za akcije
            keyboard = [
                [InlineKeyboardButton("🎥 Animiraj", callback_data="action_animate"),
                 InlineKeyboardButton("🔄 Regeneriši", callback_data="action_regen")]
            ]
            
            with open(image_path, 'rb') as f:
                await update.message.reply_photo(
                    photo=f,
                    caption=f"🖼️ **Slika generisana!**\n\n"
                           f"📝 _{prompt}_\n"
                           f"🎭 Stil: {style_name}\n"
                           f"⚙️ Provider: {provider_name}",
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
            await status_msg.delete()
        else:
            await status_msg.edit_text(
                f"❌ **Generisanje nije uspelo**\n\n"
                f"Svi provideri su trenutno zauzeti.\n"
                f"💡 Probaj ponovo za 30 sekundi!\n\n"
                f"Greške:\n```\n{error[:500]}\n```",
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        await status_msg.edit_text(f"❌ Greška: {str(e)[:200]}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads za Image to Video"""
    photo = update.message.photo[-1]  # Najveća rezolucija
    
    file = await context.bot.get_file(photo.file_id)
    
    # Download sliku
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        await file.download_to_drive(f.name)
        image_path = f.name
    
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    state["last_image"] = image_path
    
    keyboard = [
        [InlineKeyboardButton("🎥 Animiraj ovu sliku", callback_data="action_animate")]
    ]
    
    await update.message.reply_text(
        "📸 **Slika primljena!**\n\n"
        "Klikni dugme ili pošalji `/animate` da je animiraš!",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def action_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle action callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    state = get_user_state(user_id)
    
    if query.data == "action_animate":
        if state.get("last_image"):
            await query.message.reply_text("🎥 Pokrećem animaciju...")
            # Simuliraj animate command
            update.message = query.message
            await animate_command(update, context)
    
    elif query.data == "action_regen":
        if state.get("last_prompt"):
            prompt = state["last_prompt"]
            
            # Prevedi i primeni stil
            translated_prompt = translate_to_english(prompt)
            styled_prompt = apply_style(translated_prompt, state["style"])
            
            # Dobij dimenzije
            width, height = ASPECT_RATIOS.get(state["ratio"], (1024, 1024))
            style_name = STYLES[state["style"]]["name"]
            
            # Status poruka
            status_msg = await query.message.reply_text(
                f"🔄 **Regenerišem sliku...**\n\n"
                f"📝 Prompt: _{prompt}_\n"
                f"🎭 Stil: {style_name}\n"
                f"📐 Format: {state['ratio']}",
                parse_mode="Markdown"
            )
            
            try:
                image_path, provider_name, error = await generate_image(styled_prompt, width, height)
                
                if image_path:
                    state["last_image"] = image_path
                    state["generation_count"] += 1
                    
                    keyboard = [
                        [InlineKeyboardButton("🎥 Animiraj", callback_data="action_animate"),
                         InlineKeyboardButton("🔄 Regeneriši", callback_data="action_regen")]
                    ]
                    
                    with open(image_path, 'rb') as f:
                        await query.message.reply_photo(
                            photo=f,
                            caption=f"🔄 **Regenerisano!**\n\n"
                                   f"📝 _{prompt}_\n"
                                   f"🎭 Stil: {style_name}\n"
                                   f"⚙️ Provider: {provider_name}",
                            parse_mode="Markdown",
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                    await status_msg.delete()
                else:
                    await status_msg.edit_text(f"❌ Regenerisanje nije uspelo:\n{error[:200]}")
            except Exception as e:
                await status_msg.edit_text(f"❌ Greška: {str(e)[:200]}")
        else:
            await query.message.reply_text("❌ Nema prethodnog prompta za regeneraciju!")

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    """Start the bot"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║  🔥🔥🔥 STEVAN BEAST AI BOT 🔥🔥🔥                        ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║  🎨 8 Image Providers (FLUX, SD3, Animagine...)           ║
║  🎬 6 Video Providers (CogVideoX, SVD, Wan2.1...)         ║
║  ⚡ Smart Rotation & Load Balancing                       ║
║  🌍 Auto Translation                                      ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║  Starting...                                              ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Build application
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("style", style_command))
    app.add_handler(CommandHandler("ratio", ratio_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("video", video_command))
    app.add_handler(CommandHandler("animate", animate_command))
    
    # Callback handlers
    app.add_handler(CallbackQueryHandler(action_callback, pattern="^action_"))
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    # Message handlers
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Run
    logger.info("🚀 Bot is starting...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
