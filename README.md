# 🔥 STEVAN BEAST AI BOT 🔥

Telegram bot za AI generisanje slika i videa sa rotacijom kroz multiple besplatne providere.

## ✨ Features

### 🎨 IMAGE GENERATION (8 Providers)
- FLUX.1-schnell (⚡ Najbrži)
- FLUX.1-dev (🎨 Najviši kvalitet)
- Stable Diffusion 3 Medium
- FLUX-Merged
- Animagine XL (🎌 Anime)
- RealVisXL (📸 Realistic)
- Playground v2.5
- PixArt-Sigma

### 🎬 VIDEO GENERATION (6 Providers)
- CogVideoX-5B (Text to Video)
- Stable Video Diffusion (Image to Video)
- AnimateDiff-Lightning (Text to Video)
- Wan2.1-T2V (Text to Video)
- Open-Sora (Text to Video)
- I2VGen-XL (Image to Video)

### 🎭 15 Artistic Styles
Cinematic, Anime, Realistic, Cyberpunk, Fantasy, Horror, Oil Painting, Watercolor, 3D Render, Pixel Art, Comic, Minimalist, Vintage, Neon

### 📐 6 Aspect Ratios
1:1, 16:9, 9:16, 4:3, 3:4, 21:9

### ⚡ Smart Features
- Auto-rotation through providers
- Load balancing based on success rate
- Automatic Serbian → English translation
- Provider statistics tracking

## 🚀 Deploy

### Option 1: Render.com (FREE)
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. New → Background Worker
4. Connect this repo
5. Build: `pip install -r requirements.txt`
6. Start: `python beast_video_bot.py`
7. Deploy!

### Option 2: Koyeb.com (FREE)
1. Go to [koyeb.com](https://koyeb.com)
2. Import from GitHub
3. Select this repo
4. Deploy!

## 📱 Commands

- `/start` - Start bot
- `/help` - Help
- `/style` - Change style
- `/ratio` - Change aspect ratio
- `/video [prompt]` - Generate video from text
- `/animate` - Animate last image
- `/stats` - Provider statistics

## 📝 Usage

Just send any text to generate an image!
- `Cyberpunk samurai in Tokyo`
- `Mačka pleše na kiši` (Serbian - auto translates)
- `/video A dragon flying over mountains`

## 🔧 Environment

No API keys needed - uses free HuggingFace Spaces!

## 📄 License

MIT
