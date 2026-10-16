# -*- coding: utf-8 -*-
import os
import json
import uuid
import time
import base64
import textwrap
import threading
import subprocess
import io

from flask import (Flask, render_template, request, jsonify,
                   send_file, session, url_for)
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import requests
import numpy as np
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

# ── Uygulama kurulumu ──────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder='templates_standup',
    static_folder='static_standup',
    static_url_path='/static_standup'
)
app.secret_key = os.environ.get('STANDUP_SECRET', 'standup-secret-2024-xYz')

OPENAI_KEY   = os.environ.get('OPENAI_API_KEY', '')
ELEVEN_KEY   = os.environ.get('ELEVENLABS_API_KEY', '')

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
# Render.com'da kalici disk /opt/render/project/src/standup_data konumuna mount edilir.
# Yerel gelistirmede BASE_DIR/standup_data kullanilir.
DATA_DIR     = os.environ.get('DATA_DIR', os.path.join(BASE_DIR, 'standup_data'))

CHAR_DIR     = os.path.join(DATA_DIR, 'characters')
OUTPUT_DIR   = os.path.join(DATA_DIR, 'outputs')
AUDIO_DIR    = os.path.join(DATA_DIR, 'audio')
VENUE_DIR    = os.path.join(DATA_DIR, 'venues')

# Statik karakterler ve mekanlar ayni zamanda static_standup/ dizinine de kopyalanir (serve icin)
STATIC_CHAR_DIR  = os.path.join(BASE_DIR, 'static_standup', 'characters')
STATIC_VENUE_DIR = os.path.join(BASE_DIR, 'static_standup', 'venues')
FONT_PATH        = os.path.join(BASE_DIR, 'static_standup', 'fonts', 'DejaVuSans-Bold.ttf')

for d in [CHAR_DIR, OUTPUT_DIR, AUDIO_DIR, VENUE_DIR,
          STATIC_CHAR_DIR, STATIC_VENUE_DIR,
          os.path.join(BASE_DIR, 'static_standup', 'fonts')]:
    os.makedirs(d, exist_ok=True)

VIDEO_W, VIDEO_H = 1280, 720
FPS = 24

# ── Renk paleti ve mekan temaları ───────────────────────────────────────────
STAGE_THEMES = {
    'bar':     {'bg': (24, 14, 10),    'spot': (255, 190, 60),  'name': '🍺 Bar & Pub Ortamı'},
    'club':    {'bg': (15, 10, 30),    'spot': (255, 220, 80),  'name': '🎙️ Tuğla Duvarlı Kulüp'},
    'theatre': {'bg': (35, 12, 18),    'spot': (255, 180, 60),  'name': '🎭 Kırmızı Perdeli Tiyatro'},
    'street':  {'bg': (18, 25, 35),    'spot': (200, 240, 255), 'name': '🌆 Açık Hava / Sokak'},
    'studio':  {'bg': (10, 10, 40),    'spot': (150, 180, 255), 'name': '🎬 Modern Talk-Show Stüdyosu'},
}

JOB_STORE: dict[str, dict] = {}  # job_id -> {status, progress, result}

# ═══════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════

def _font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def generate_character_image(prompt: str, role: str, color: str) -> str:
    """Karakter görseli üret — DALL·E, Pollinations AI veya gelişmiş yerel çizim."""
    cid = str(uuid.uuid4())[:8]
    out_path = os.path.join(CHAR_DIR, f'{cid}.png')

    # 1. Öncelik: OpenAI DALL-E 3 (eğer anahtar varsa)
    if OPENAI_KEY:
        try:
            headers = {'Authorization': f'Bearer {OPENAI_KEY}',
                       'Content-Type': 'application/json'}
            full_prompt = (
                f"3D Pixar style cartoon character, single full-body illustration isolated on white background. "
                f"Role: {role}. Description: {prompt}. "
                f"Comedic exaggerated features, stand-up comedy club setting holding a microphone."
            )
            body = {'model': 'dall-e-3', 'prompt': full_prompt,
                    'n': 1, 'size': '1024x1024', 'response_format': 'b64_json'}
            r = requests.post('https://api.openai.com/v1/images/generations',
                              headers=headers, json=body, timeout=40)
            data = r.json()
            if 'data' in data and len(data['data']) > 0:
                img_b64 = data['data'][0]['b64_json']
                img_bytes = base64.b64decode(img_b64)
                with open(out_path, 'wb') as f:
                    f.write(img_bytes)
                _make_transparent(out_path)
                _copy_to_static(cid)
                return cid
        except Exception as e:
            print(f'DALL-E hata: {e}')

    # 2. Öncelik: Pollinations.ai (Tamamen ÜCRETSİZ, API anahtarı istemez, gerçek 3D animasyon üretir)
    try:
        en_prompt = _translate_prompt_to_en(prompt, role)
        full_ai_prompt = f"3d pixar disney style animated funny standup comedian {en_prompt}, holding microphone, full body on stage, vibrant colors, isolated on white background"
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(full_ai_prompt)}?width=512&height=512&nologo=true&seed={int(time.time()) % 10000}"
        r = requests.get(url, timeout=18)
        if r.status_code == 200 and len(r.content) > 5000:
            with open(out_path, 'wb') as f:
                f.write(r.content)
            _make_transparent(out_path)
            _copy_to_static(cid)
            return cid
    except Exception as e:
        print(f'Pollinations AI hata: {e}')

    # 3. Öncelik: Prompt anahtar kelimelerini okuyan akıllı yerel çizim
    _draw_local_character(out_path, role, color, prompt)
    _copy_to_static(cid)
    return cid


def _copy_to_static(cid: str):
    """Karakter PNG'yi static klasörüne kopyala (Flask'ın serve etmesi için)."""
    import shutil
    src = os.path.join(CHAR_DIR, f'{cid}.png')
    dst = os.path.join(STATIC_CHAR_DIR, f'{cid}.png')
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)


def _translate_prompt_to_en(p: str, role: str) -> str:
    """Türkçe prompt anahtar kelimelerini İngilizceye çevirir."""
    p_lower = (p or '').lower()
    traits = []
    if 'kırmızı bere' in p_lower or 'kirmizi bere' in p_lower or 'bere' in p_lower:
        traits.append('wearing a red french beret')
    if 'şapka' in p_lower or 'sapka' in p_lower:
        traits.append('wearing a comedy hat')
    if 'bıyık' in p_lower or 'biyik' in p_lower or 'bıyıklı' in p_lower:
        traits.append('with thick mustache')
    if 'sakal' in p_lower or 'sakallı' in p_lower:
        traits.append('with beard')
    if 'gözlük' in p_lower or 'gozluk' in p_lower or 'gözlüklü' in p_lower:
        traits.append('wearing eyeglasses')
    if 'şişman' in p_lower or 'tombul' in p_lower or 'kilolu' in p_lower:
        traits.append('chubby overweight body')
    if 'genç' in p_lower or 'genc' in p_lower:
        traits.append('young')
    if 'yaşlı' in p_lower or 'yasli' in p_lower:
        traits.append('elderly with gray hair')
    if 'kadın' in p_lower or 'kadin' in p_lower:
        traits.append('female')
    else:
        traits.append('male')

    base = ', '.join(traits) if traits else p
    return f"{base}, role of {role}"


def _make_transparent(img_path: str):
    """Beyaz arka plan piksellerini şeffaflaştırır."""
    try:
        img = Image.open(img_path).convert('RGBA')
        datas = img.getdata()
        new_data = []
        for item in datas:
            # Beyaza çok yakın arka plan piksellerini şeffaf yap
            if item[0] > 235 and item[1] > 235 and item[2] > 235:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        img.putdata(new_data)
        img.save(img_path, 'PNG')
    except Exception as e:
        print(f'Transparan yapma hata: {e}')


def _draw_local_character(path: str, role: str, body_color: str, prompt: str):
    """Prompt'taki kırmızı bere, bıyık, gözlük vb. özellikleri okuyarak çizen akıllı çizici."""
    W, H = 400, 600
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    p_lower = (prompt or '').lower()
    has_red_beret = ('kırmızı bere' in p_lower or 'kirmizi bere' in p_lower or 'bere' in p_lower)
    has_hat       = ('şapka' in p_lower or 'sapka' in p_lower)
    has_mustache  = ('bıyık' in p_lower or 'biyik' in p_lower or 'bıyıklı' in p_lower)
    has_beard     = ('sakal' in p_lower or 'sakallı' in p_lower or 'sakalli' in p_lower)
    has_glasses   = ('gözlük' in p_lower or 'gozluk' in p_lower or 'gözlüklü' in p_lower)
    is_chubby     = ('şişman' in p_lower or 'sisman' in p_lower or 'tombul' in p_lower or 'kilolu' in p_lower)
    is_female     = ('kadın' in p_lower or 'kadin' in p_lower or 'kız' in p_lower)
    is_elderly    = ('yaşlı' in p_lower or 'yasli' in p_lower or 'dede' in p_lower)

    try:
        r, g, b = int(body_color[1:3], 16), int(body_color[3:5], 16), int(body_color[5:7], 16)
    except Exception:
        r, g, b = 45, 65, 95

    skin = (255, 215, 175) if not is_elderly else (240, 205, 175)

    # 1. Gövde
    torso_w = 170 if is_chubby else 140
    tx1, tx2 = 200 - torso_w // 2, 200 + torso_w // 2
    draw.ellipse([tx1, 220, tx2, 430], fill=(r, g, b))

    # Gömlek & Kravat
    draw.polygon([(170, 220), (230, 220), (200, 290)], fill=(245, 245, 250))
    draw.polygon([(194, 230), (206, 230), (209, 340), (200, 355), (191, 340)], fill=(210, 40, 40))

    # Kollar
    draw.ellipse([tx1 - 25, 230, tx1 + 35, 360], fill=(r, g, b))
    draw.ellipse([tx2 - 35, 230, tx2 + 25, 360], fill=(r, g, b))

    # Bacaklar & Ayakkabılar
    draw.rectangle([140, 420, 185, 560], fill=(max(0, r - 25), max(0, g - 25), max(0, b - 25)))
    draw.rectangle([215, 420, 260, 560], fill=(max(0, r - 25), max(0, g - 25), max(0, b - 25)))
    draw.ellipse([115, 545, 190, 580], fill=(20, 20, 25))
    draw.ellipse([210, 545, 285, 580], fill=(20, 20, 25))

    # Kadın saçı (arka)
    if is_female:
        draw.ellipse([120, 70, 280, 270], fill=(110, 60, 30))

    # 2. Baş
    draw.ellipse([140, 70, 260, 220], fill=skin)

    # Gözler
    draw.ellipse([162, 125, 186, 149], fill=(30, 30, 30))
    draw.ellipse([214, 125, 238, 149], fill=(30, 30, 30))
    draw.ellipse([170, 129, 178, 137], fill=(255, 255, 255))
    draw.ellipse([222, 129, 230, 137], fill=(255, 255, 255))

    # 3. Kırmızı Bere veya Şapka
    if has_red_beret:
        # Kırmızı Fransız beresi
        draw.ellipse([115, 42, 285, 118], fill=(215, 30, 40))
        draw.ellipse([130, 78, 270, 120], fill=(175, 20, 30))
        draw.rectangle([197, 34, 203, 48], fill=(150, 15, 25)) # Bere sapı
    elif has_hat:
        # Şapka
        draw.ellipse([105, 75, 295, 105], fill=(35, 35, 40))
        draw.rectangle([135, 40, 265, 85], fill=(35, 35, 40))
    else:
        # Saç
        hair_color = (180, 180, 180) if is_elderly else (50, 35, 25)
        draw.arc([135, 60, 265, 150], start=180, end=360, fill=hair_color, width=12)

    # 4. Gözlük
    if has_glasses:
        draw.ellipse([153, 117, 193, 157], outline=(30, 30, 35), width=4)
        draw.ellipse([207, 117, 247, 157], outline=(30, 30, 35), width=4)
        draw.line([(193, 137), (207, 137)], fill=(30, 30, 35), width=4)

    # 5. Bıyık & Sakal
    if has_mustache:
        draw.chord([162, 155, 200, 185], start=0, end=180, fill=(35, 25, 20))
        draw.chord([200, 155, 238, 185], start=0, end=180, fill=(35, 25, 20))

    if has_beard:
        draw.chord([155, 175, 245, 235], start=0, end=180, fill=(45, 35, 30))

    # 6. Gülümseyen Ağız
    draw.arc([168, 168, 232, 204], start=10, end=170, fill=(175, 35, 40), width=4)

    # 7. Stand-up Mikrofonu
    draw.line([(255, 240), (255, 460)], fill=(180, 180, 190), width=6)
    draw.ellipse([(242, 210), (268, 245)], fill=(75, 75, 85))
    draw.ellipse([(246, 215), (264, 240)], fill=(145, 145, 155))

    img.save(path, 'PNG')


def gpt_generate_script(topic: str, duration_min: int, style: str) -> list[dict]:
    """GPT-4 ile senaryo üret, yoksa örnek döndür."""
    if OPENAI_KEY:
        try:
            headers = {'Authorization': f'Bearer {OPENAI_KEY}',
                       'Content-Type': 'application/json'}
            system = (
                "Sen profesyonel bir stand-up komedyenisin. "
                "Verilen konuya göre Türkçe stand-up metni yaz. "
                "JSON listesi döndür: [{\"speaker\": \"comedian\", \"text\": \"...\"}]. "
                "speaker değerleri: comedian, comedian2, host, audience_laugh, audience_clap"
            )
            user_msg = (
                f"Konu: {topic}\n"
                f"Süre: ~{duration_min} dakika\n"
                f"Stil: {style}\n"
                "Stand-up senaryo yaz. Sadece JSON döndür."
            )
            body = {
                'model': 'gpt-4o',
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': user_msg}],
                'temperature': 0.9,
                'response_format': {'type': 'json_object'}
            }
            r = requests.post('https://api.openai.com/v1/chat/completions',
                              headers=headers, json=body, timeout=60)
            data = r.json()
            raw = data['choices'][0]['message']['content']
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            # GPT bazen {"lines": [...]} döndürür
            for v in parsed.values():
                if isinstance(v, list):
                    return v
        except Exception as e:
            print(f'GPT hata: {e}')

    # ── Fallback örnek senaryo ──────────────────────────────────────────────
    return _sample_script(topic)


def _sample_script(topic: str) -> list[dict]:
    lines = [
        {"speaker": "host",      "text": f"Merhaba sevgili seyirciler! Bu akşamki konumuz: {topic}. Hazır mısınız?"},
        {"speaker": "audience_clap", "text": ""},
        {"speaker": "comedian",  "text": f"Arkadaşlar, bugün {topic} hakkında konuşacağız. Benim de bu konuda çok şey söylemek istiyorum!"},
        {"speaker": "comedian",  "text": "Bir keresinde annem bana 'Oğlum, sen hiç ciddiye alınmazsın' dedi. Ben de dedim ki: 'Anne, tam da bu yüzden stand-up yapıyorum!'"},
        {"speaker": "audience_laugh", "text": ""},
        {"speaker": "comedian",  "text": f"{topic} deyince aklıma hep şu geliyor. Sabah kalktım, baktım ki... her şey aynı. Değişen hiçbir şey yok. Ama bizim beklentilerimiz değişiyor ya! İşte asıl mesele bu!"},
        {"speaker": "comedian",  "text": "Karım dedi ki 'Sen hiç dinlemiyorsun beni.' Ben de dedim ki 'Ne dedin?' Ama gerçekten duymadım ya!"},
        {"speaker": "audience_laugh", "text": ""},
        {"speaker": "comedian",  "text": "Teknoloji de iyice sapıttı. Telefona 'OK Google, hayatımı düzelt' dedim. 'Üzgünüm, bu konuda yardımcı olamam' dedi. Yani telefon bile benden ümidini kesti!"},
        {"speaker": "audience_laugh", "text": ""},
        {"speaker": "comedian",  "text": "Doktora gittim, 'Doktor bey, her sabah kalktığımda yorgun hissediyorum' dedim. 'Kaç saatte uyuyorsunuz?' dedi. '10 saat' dedim. 'Bu çok fazla' dedi. Yani uyumak da sağlıksız oldu mu? Ne yapacağız?"},
        {"speaker": "comedian",  "text": "Arkadaşlar, hayat zor ama gülmesek daha da zor olur. Teşekkür ederim, iyi geceler!"},
        {"speaker": "audience_clap", "text": ""},
    ]
    return lines


def text_to_speech(text: str, speaker: str, lang: str = 'tr') -> str | None:
    """TTS -> Doğal erkek/kadın sesi veya OpenAI/ElevenLabs."""
    if not text.strip():
        return None
    aid = str(uuid.uuid4())[:8]
    out = os.path.join(AUDIO_DIR, f'{aid}.mp3')

    # 1. Öncelik: OpenAI TTS (Varsa en doğal insansı ses)
    if OPENAI_KEY:
        try:
            # Komedyen için kalın doğal erkek sesi 'onyx' veya 'echo'
            voice = 'onyx' if speaker in ('comedian', 'host') else 'nova'
            headers = {'Authorization': f'Bearer {OPENAI_KEY}', 'Content-Type': 'application/json'}
            body = {'model': 'tts-1', 'input': text, 'voice': voice}
            r = requests.post('https://api.openai.com/v1/audio/speech', headers=headers, json=body, timeout=30)
            if r.status_code == 200:
                with open(out, 'wb') as f:
                    f.write(r.content)
                return out
        except Exception as e:
            print(f'OpenAI TTS hata: {e}')

    # 2. Öncelik: ElevenLabs (Varsa)
    if ELEVEN_KEY:
        voice_map = {
            'comedian':  'pNInz6obpgDQGcFmaJgB',  # Adam (Erkek)
            'comedian2': 'EXAVITQu4vr4xnSDxMaL',  # Bella (Kadın)
            'host':      'VR6AewLTigWG4xSOukaG',  # Arnold (Erkek)
        }
        voice_id = voice_map.get(speaker, voice_map['comedian'])
        try:
            headers = {'xi-api-key': ELEVEN_KEY, 'Content-Type': 'application/json'}
            body = {'text': text, 'model_id': 'eleven_multilingual_v2',
                    'voice_settings': {'stability': 0.5, 'similarity_boost': 0.75}}
            r = requests.post(
                f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
                headers=headers, json=body, timeout=30)
            if r.status_code == 200:
                with open(out, 'wb') as f:
                    f.write(r.content)
                return out
        except Exception as e:
            print(f'ElevenLabs hata: {e}')

    # 3. Öncelik: gTTS + ffmpeg Akıllı Ses Tonlama (Tamamen ÜCRETSİZ, Erkek / Kadın ayrımı)
    try:
        raw_out = os.path.join(AUDIO_DIR, f'raw_{aid}.mp3')
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(raw_out)

        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # Konuşmacıya göre ses filtresi belirle:
        # Komedyen (Erkek): Frekansı %22 düşürerek tok, doğal ve erkeksi bir tona dönüştürür
        if speaker == 'comedian':
            af_filter = 'asetrate=24000*0.78,aresample=24000,atempo=1.28'
        elif speaker == 'host':
            # Sunucu: Tok, güçlü radyo spikeri sesi
            af_filter = 'asetrate=24000*0.74,aresample=24000,atempo=1.35'
        else:
            # 2. Komedyen (Kadın): Canlı kadın tonu
            af_filter = 'asetrate=24000*1.04,aresample=24000,atempo=0.96'

        subprocess.run([
            ffmpeg_exe, '-y', '-i', raw_out,
            '-filter:a', af_filter,
            out
        ], capture_output=True, timeout=15)

        if os.path.exists(raw_out):
            try:
                os.remove(raw_out)
            except Exception:
                pass

        if os.path.exists(out):
            return out
        return raw_out
    except Exception as e:
        print(f'gTTS hata: {e}')
        return None


def get_audio_duration(path: str) -> float:
    """imageio_ffmpeg ile ses süresi al."""
    try:
        import imageio_ffmpeg, re
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [ffmpeg_exe, '-i', path, '-f', 'null', '-'],
            capture_output=True, text=True, timeout=10
        )
        m = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.?\d*)', result.stderr)
        if m:
            hours, mins, secs = m.groups()
            return int(hours) * 3600 + int(mins) * 60 + float(secs)
    except Exception:
        pass
    return 3.0  # varsayılan 3 saniye


# ═══════════════════════════════════════════════════════════════════════════
# VİDEO ÜRETİM SİSTEMİ
# ═══════════════════════════════════════════════════════════════════════════

def _draw_stage_bg(frame: Image.Image, theme_key: str, custom_bg_path: str = None):
    """Sahne ve mekan arka planı çizer (Bar, Kulüp, Tiyatro veya özel resim)."""
    # 1. Kullanıcı kendi sahne/bar resmini yüklemiş veya AI ile üretmişse
    if custom_bg_path and os.path.exists(custom_bg_path):
        try:
            bg_img = Image.open(custom_bg_path).convert('RGB').resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
            frame.paste(bg_img, (0, 0))
            draw = ImageDraw.Draw(frame)
            _draw_audience(draw)
            return
        except Exception as e:
            print(f'Özel sahne yükleme hata: {e}')

    draw = ImageDraw.Draw(frame)
    theme = STAGE_THEMES.get(theme_key, STAGE_THEMES['bar'])
    bg = theme['bg']
    spot = theme['spot']

    if theme_key == 'bar':
        # 🍺 Bar & Pub Ortamı Çizimi
        # Sıcak loş ahşap bar degradesi
        for y in range(VIDEO_H):
            ratio = y / VIDEO_H
            r = int(32 - ratio * 18)
            g = int(18 - ratio * 10)
            b = int(14 - ratio * 8)
            draw.line([(0, y), (VIDEO_W, y)], fill=(r, g, b))

        # Ahşap içki rafları
        draw.rectangle([(60, 160), (VIDEO_W - 60, 172)], fill=(65, 40, 25))
        draw.rectangle([(60, 280), (VIDEO_W - 60, 292)], fill=(65, 40, 25))

        # Raflardaki renkli içki şişeleri ve bardaklar
        bottle_cols = [(190, 85, 25), (45, 155, 75), (170, 35, 45), (220, 165, 45), (85, 45, 115)]
        for i, sx in enumerate(range(100, VIDEO_W - 100, 48)):
            c1 = bottle_cols[i % len(bottle_cols)]
            bh1 = 50 + (i % 3) * 10
            draw.rectangle([(sx, 160 - bh1), (sx + 18, 160)], fill=c1)
            draw.rectangle([(sx + 5, 160 - bh1 - 10), (sx + 13, 160 - bh1)], fill=c1)

            c2 = bottle_cols[(i + 2) % len(bottle_cols)]
            bh2 = 45 + ((i * 2) % 3) * 12
            draw.rectangle([(sx, 280 - bh2), (sx + 18, 280)], fill=c2)
            draw.rectangle([(sx + 5, 280 - bh2 - 10), (sx + 13, 280 - bh2)], fill=c2)

        # Parlayan Neon Tabela
        draw.text((VIDEO_W // 2, 75), '★ STAND-UP COMEDY PUB & BAR ★',
                  fill=(255, 70, 110), anchor='mm', font=_font(26))

        # Sıcak amber sahne spotu
        cx, cy = VIDEO_W // 2, 0
        for radius in range(400, 0, -8):
            alpha = max(0, int(35 * (1 - radius / 400)))
            draw.ellipse([cx - radius, cy - radius // 2, cx + radius, cy + radius // 2],
                         fill=(min(255, spot[0] + alpha), min(255, spot[1] + alpha), spot[2]))

        # Ahşap zemin
        draw.polygon([(0, VIDEO_H - 140), (VIDEO_W, VIDEO_H - 140), (VIDEO_W, VIDEO_H), (0, VIDEO_H)],
                     fill=(48, 28, 18))
        for fy in range(VIDEO_H - 140, VIDEO_H, 22):
            draw.line([(0, fy), (VIDEO_W, fy)], fill=(32, 18, 10), width=2)

    elif theme_key == 'theatre':
        # 🎭 Kırmızı Kadife Perdeli Tiyatro
        for y in range(VIDEO_H):
            ratio = y / VIDEO_H
            draw.line([(0, y), (VIDEO_W, y)], fill=(int(35 - ratio * 20), 10, 15))

        # Yan ve üst kadife perdeler
        draw.polygon([(0, 0), (140, 0), (90, VIDEO_H), (0, VIDEO_H)], fill=(120, 15, 25))
        draw.polygon([(VIDEO_W - 140, 0), (VIDEO_W, 0), (VIDEO_W, VIDEO_H), (VIDEO_W - 90, VIDEO_H)], fill=(120, 15, 25))
        draw.rectangle([(0, 0), (VIDEO_W, 70)], fill=(140, 20, 30))

        # Altın spot
        cx, cy = VIDEO_W // 2, 0
        for radius in range(380, 0, -6):
            draw.ellipse([cx - radius, cy - radius // 2, cx + radius, cy + radius // 2], fill=(255, 200, 80))

        # Cilalı sahne
        draw.polygon([(0, VIDEO_H - 140), (VIDEO_W, VIDEO_H - 140), (VIDEO_W, VIDEO_H), (0, VIDEO_H)], fill=(60, 30, 20))

    else:
        # 🎙️ Kulüp / Stüdyo / Diğer
        for y in range(VIDEO_H):
            ratio = y / VIDEO_H
            r = int(bg[0] * (1 - ratio * 0.5))
            g = int(bg[1] * (1 - ratio * 0.5))
            b = int(bg[2] * (1 - ratio * 0.5))
            draw.line([(0, y), (VIDEO_W, y)], fill=(r, g, b))

        cx, cy = VIDEO_W // 2, 0
        for radius in range(380, 0, -5):
            draw.ellipse([cx - radius, cy - radius // 2, cx + radius, cy + radius // 2], fill=spot)

        draw.polygon([(0, VIDEO_H - 140), (VIDEO_W, VIDEO_H - 140), (VIDEO_W, VIDEO_H), (0, VIDEO_H)],
                     fill=(int(bg[0] * 1.5), int(bg[1] * 1.5), int(bg[2] * 1.5)))

    _draw_audience(draw)


def _draw_audience(draw: ImageDraw.ImageDraw):
    silhouette = (20, 15, 35)
    positions = [60, 160, 260, 360, 460, 560, 660, 760, 860, 960, 1060, 1160]
    for i, x in enumerate(positions):
        h = 55 + (i % 3) * 10
        y_base = VIDEO_H - 145
        # Gövde
        draw.ellipse([x - 18, y_base - h, x + 18, y_base - h + 50],
                     fill=silhouette)
        # Kafa
        draw.ellipse([x - 14, y_base - h - 28, x + 14, y_base - h + 2],
                     fill=silhouette)


def create_custom_photo_character(image_file, role: str = 'comedian', mode: str = 'bobblehead', body_color: str = '#2c3e50') -> str:
    """Kullanıcının yüklediği kendi fotoğrafından animasyon karakteri üretir."""
    cid = str(uuid.uuid4())[:8]
    out_path = os.path.join(CHAR_DIR, f'{cid}.png')

    img = Image.open(image_file).convert('RGBA')
    W, H = 400, 600
    canvas = Image.new('RGBA', (W, H), (0, 0, 0, 0))

    if mode == 'bobblehead':
        # 1. Çizgi Stand-Up Kıyafeti & Gövde
        draw = ImageDraw.Draw(canvas)
        try:
            r, g, b = int(body_color[1:3], 16), int(body_color[3:5], 16), int(body_color[5:7], 16)
        except Exception:
            r, g, b = 44, 62, 80

        # Ceket & Omuzlar
        draw.polygon([(100, 230), (300, 230), (330, 430), (70, 430)], fill=(r, g, b))
        # Beyaz gömlek yakası
        draw.polygon([(165, 230), (235, 230), (200, 310)], fill=(245, 245, 250))
        # Kırmızı kravat
        draw.polygon([(194, 240), (206, 240), (210, 360), (200, 375), (190, 360)], fill=(210, 45, 45))
        # Pantolon
        pant_color = (max(0, r - 25), max(0, g - 25), max(0, b - 25))
        draw.rectangle([(120, 430), (185, 565)], fill=pant_color)
        draw.rectangle([(215, 430), (280, 565)], fill=pant_color)
        # Ayakkabılar
        draw.ellipse([(105, 545), (190, 580)], fill=(15, 15, 20))
        draw.ellipse([(210, 545), (295, 580)], fill=(15, 15, 20))

        # 2. Kullanıcının fotoğrafından kafa/yüz kesimi (210x230)
        head_w, head_h = 210, 230
        min_dim = min(img.width, img.height)
        left = (img.width - min_dim) // 2
        top = max(0, (img.height - min_dim) // 4)
        cropped = img.crop((left, top, left + min_dim, top + min_dim)).resize((head_w, head_h), Image.LANCZOS)

        # Oval maske
        mask = Image.new('L', (head_w, head_h), 0)
        m_draw = ImageDraw.Draw(mask)
        m_draw.ellipse([4, 4, head_w - 4, head_h - 4], fill=255)

        # Çizgi film kenar çizgisi (altın sarısı aksan)
        outline_layer = Image.new('RGBA', (head_w, head_h), (0, 0, 0, 0))
        o_draw = ImageDraw.Draw(outline_layer)
        o_draw.ellipse([2, 2, head_w - 2, head_h - 2], outline=(245, 200, 66, 230), width=4)

        head_x = (W - head_w) // 2
        head_y = 30
        canvas.paste(cropped, (head_x, head_y), mask)
        canvas.paste(outline_layer, (head_x, head_y), outline_layer)

        # 3. Stand-up Mikrofonu
        draw.line([(240, 260), (240, 480)], fill=(180, 180, 190), width=6)
        draw.ellipse([(227, 230), (253, 265)], fill=(80, 80, 90))
        draw.ellipse([(230, 235), (250, 260)], fill=(150, 150, 160))

    else:
        # Doğrudan fotoğraf modu
        img_copy = img.copy()
        img_copy.thumbnail((360, 520), Image.LANCZOS)
        px = (W - img_copy.width) // 2
        py = (H - img_copy.height) - 40
        canvas.paste(img_copy, (px, py), img_copy if img_copy.mode == 'RGBA' else None)

        draw = ImageDraw.Draw(canvas)
        draw.line([(px + img_copy.width - 35, py + img_copy.height // 2), (px + img_copy.width - 35, H - 30)], fill=(180, 180, 190), width=6)
        draw.ellipse([(px + img_copy.width - 48, py + img_copy.height // 2 - 25), (px + img_copy.width - 22, py + img_copy.height // 2 + 10)], fill=(80, 80, 90))

    canvas.save(out_path, 'PNG')
    _copy_to_static(cid)
    return cid


def _paste_character(frame: Image.Image, char_path: str,
                     x: int, y: int, scale: float = 1.0, mouth_open: bool = False, bounce: int = 0, tilt: float = 0.0):
    """Karakter PNG'yi frame'e yapıştırır, konuşurken ağzını açar ve sahnede canlı animasyon uygular."""
    try:
        char = Image.open(char_path).convert('RGBA')
        new_w = int(char.width * scale)
        new_h = int(char.height * scale)
        char = char.resize((new_w, new_h), Image.LANCZOS)

        if mouth_open:
            # Gerçek ağız açılma animasyonu (Çene aşağı iner, ağız içi ve dişler görünür)
            split_y = int(new_h * 0.38)
            top = char.crop((0, 0, new_w, split_y))
            bot = char.crop((0, split_y, new_w, new_h))
            gap = 12
            combined = Image.new('RGBA', (new_w, new_h + gap), (0, 0, 0, 0))

            # Ağız boşluğu ve diş çizimi
            c_draw = ImageDraw.Draw(combined)
            mw = max(14, int(new_w * 0.18))
            cx = new_w // 2
            # Ağız içi (koyu kırmızı / bordo)
            c_draw.ellipse([cx - mw, split_y - 2, cx + mw, split_y + gap + 3], fill=(55, 12, 18))
            # Üst dişler (beyaz çizgi)
            c_draw.rectangle([cx - int(mw * 0.7), split_y, cx + int(mw * 0.7), split_y + 3], fill=(250, 250, 250))

            combined.paste(top, (0, 0), top)
            combined.paste(bot, (0, split_y + gap), bot)
            char = combined

        if abs(tilt) > 0.4:
            char = char.rotate(tilt, resample=Image.BICUBIC, expand=True)

        paste_x = x - char.width // 2
        paste_y = y - char.height - bounce
        frame.paste(char, (paste_x, paste_y), char)
    except Exception as e:
        print(f'Karakter yapıştırma hata: {e}')


def _draw_subtitle(draw: ImageDraw.ImageDraw, text: str, speaker_name: str):
    """Altyazı bandı çiz."""
    if not text.strip():
        return
    fnt_big = _font(28)
    fnt_sm  = _font(20)
    band_y  = VIDEO_H - 125

    # Altyazı arka plan (y0: 595, y1: 710 -> Pillow hata vermez)
    draw.rectangle([(0, band_y), (VIDEO_W, VIDEO_H - 10)],
                   fill=(0, 0, 0, 200))

    # Konuşmacı adı
    draw.text((VIDEO_W // 2, band_y + 10), speaker_name,
              font=fnt_sm, fill=(255, 200, 50), anchor='mt')

    # Metin (max 2 satır)
    wrapped = textwrap.fill(text, width=70)
    draw.text((VIDEO_W // 2, band_y + 38), wrapped,
              font=fnt_big, fill=(255, 255, 255), anchor='mt',
              align='center')


def resolve_line_audio(line: dict, lang: str = 'tr'):
    """Her satır için ses dosyasını (kullanıcı sesi, efekt veya TTS) ve süresini belirler."""
    speaker = line.get('speaker', 'comedian')
    text    = line.get('text', '')
    custom_audio = line.get('custom_audio')

    # 1. Kullanıcının kendi yüklediği/kaydettiği ses
    if custom_audio:
        fname = os.path.basename(custom_audio)
        cand = os.path.join(AUDIO_DIR, fname)
        if os.path.exists(cand):
            dur = get_audio_duration(cand)
            return cand, max(1.5, dur)

    # 2. Efekt sesleri
    if speaker in ('audience_laugh', 'audience_clap'):
        eff = 'laugh.mp3' if speaker == 'audience_laugh' else 'clap.mp3'
        eff_path = os.path.join(BASE_DIR, 'static_standup', 'sounds', eff)
        if os.path.exists(eff_path):
            return eff_path, 2.5
        return None, 2.5

    # 3. Metin varsa TTS üretimi
    if text.strip():
        audio_path = text_to_speech(text, speaker, lang)
        if audio_path:
            dur = get_audio_duration(audio_path)
            return audio_path, max(1.5, dur)

    return None, 3.0


def _generate_video_worker(job_id: str, project: dict):
    """Arka planda video üret."""
    JOB_STORE[job_id] = {'status': 'running', 'progress': 0, 'result': None, 'error': None}

    try:
        import imageio, math
        theme_key = project.get('theme', 'bar')
        lines     = project.get('lines', [])
        characters = project.get('characters', {})  # role -> cid
        lang      = project.get('lang', 'tr')
        custom_bg = project.get('custom_bg')

        custom_bg_path = None
        if custom_bg:
            cand = os.path.join(VENUE_DIR, os.path.basename(custom_bg))
            if os.path.exists(cand):
                custom_bg_path = cand

        char_paths = {}
        for role, cid in characters.items():
            p = os.path.join(CHAR_DIR, f'{cid}.png')
            if os.path.exists(p):
                char_paths[role] = p

        CHAR_POS = {
            'comedian':  (VIDEO_W // 2, VIDEO_H - 155),
            'comedian2': (VIDEO_W // 3, VIDEO_H - 155),
            'host':      (VIDEO_W // 4, VIDEO_H - 155),
        }

        video_id = str(uuid.uuid4())[:8]
        video_path = os.path.join(OUTPUT_DIR, f'{video_id}.mp4')
        writer = imageio.get_writer(video_path, fps=FPS,
                                    codec='libx264', quality=7,
                                    macro_block_size=None,
                                    ffmpeg_params=['-pix_fmt', 'yuv420p'])

        total = len(lines)
        speaker_names = {
            'comedian': '🎤 Komedyen', 'comedian2': '🎤 Komedyen 2',
            'host': '🎙️ Sunucu', 'audience_laugh': '😂 Kahkaha',
            'audience_clap': '👏 Alkış'
        }

        for idx, line in enumerate(lines):
            speaker = line.get('speaker', 'comedian')
            text    = line.get('text', '')

            audio_path, duration = resolve_line_audio(line, lang)
            n_frames = max(int(duration * FPS), FPS)

            for f_idx in range(n_frames):
                frame = Image.new('RGB', (VIDEO_W, VIDEO_H), (0, 0, 0))
                _draw_stage_bg(frame, theme_key, custom_bg_path=custom_bg_path)

                # Tüm karakterleri çiz
                for role, pos in CHAR_POS.items():
                    cp = char_paths.get(role)
                    if not cp:
                        continue
                    is_speaking = (role == speaker)
                    scale = 0.55 if is_speaking else 0.40
                    mouth_open = is_speaking and ((f_idx // 3) % 2 == 1)
                    bounce = int(abs(math.sin(f_idx * 0.55)) * 6) if is_speaking else int(math.sin(f_idx * 0.12) * 2)
                    tilt = math.sin(f_idx * 0.35) * 3.0 if is_speaking else 0.0
                    _paste_character(frame, cp, pos[0], pos[1], scale, mouth_open=mouth_open, bounce=bounce, tilt=tilt)

                # Altyazı
                draw2 = ImageDraw.Draw(frame)
                _draw_subtitle(draw2, text if speaker not in
                               ('audience_laugh', 'audience_clap') else
                               ('😂😂😂' if speaker == 'audience_laugh' else '👏👏👏'),
                               speaker_names.get(speaker, speaker))

                writer.append_data(np.array(frame))

            JOB_STORE[job_id]['progress'] = int((idx + 1) / total * 90)

        writer.close()
        _mux_audio_video(job_id, project, lines, video_path, char_paths, theme_key, lang, speaker_names, custom_bg_path=custom_bg_path)

    except Exception as e:
        JOB_STORE[job_id]['status'] = 'error'
        JOB_STORE[job_id]['error']  = str(e)
        print(f'Video üretim hata: {e}')


def _mux_audio_video(job_id, project, lines, silent_video, char_paths, theme_key, lang, speaker_names, custom_bg_path=None):
    """Ses dosyalarını video ile birleştir (ffmpeg concat)."""
    try:
        import imageio, math
        CHAR_POS  = {
            'comedian':  (VIDEO_W // 2, VIDEO_H - 155),
            'comedian2': (VIDEO_W // 3, VIDEO_H - 155),
            'host':      (VIDEO_W // 4, VIDEO_H - 155),
        }

        video_id   = str(uuid.uuid4())[:8]
        final_path = os.path.join(OUTPUT_DIR, f'final_{video_id}.mp4')
        writer     = imageio.get_writer(final_path, fps=FPS, codec='libx264',
                                        quality=7, macro_block_size=None,
                                        ffmpeg_params=['-pix_fmt', 'yuv420p'])

        all_audio_segments = []

        for idx, line in enumerate(lines):
            speaker = line.get('speaker', 'comedian')
            text    = line.get('text', '')

            audio_path, duration = resolve_line_audio(line, lang)
            all_audio_segments.append(audio_path)

            n_frames = max(int(duration * FPS), FPS)
            for f_idx in range(n_frames):
                frame = Image.new('RGB', (VIDEO_W, VIDEO_H), (0, 0, 0))
                _draw_stage_bg(frame, theme_key, custom_bg_path=custom_bg_path)

                for role, pos in CHAR_POS.items():
                    cp = char_paths.get(role)
                    if not cp:
                        continue
                    is_speaking = (role == speaker)
                    scale = 0.55 if is_speaking else 0.40
                    mouth_open = is_speaking and ((f_idx // 3) % 2 == 1)
                    bounce = int(abs(math.sin(f_idx * 0.55)) * 6) if is_speaking else int(math.sin(f_idx * 0.12) * 2)
                    tilt = math.sin(f_idx * 0.35) * 3.0 if is_speaking else 0.0
                    _paste_character(frame, cp, pos[0], pos[1], scale, mouth_open=mouth_open, bounce=bounce, tilt=tilt)

                draw2 = ImageDraw.Draw(frame)
                _draw_subtitle(draw2,
                               text if speaker not in ('audience_laugh', 'audience_clap')
                               else ('😂😂😂' if speaker == 'audience_laugh' else '👏👏👏'),
                               speaker_names.get(speaker, speaker))

                writer.append_data(np.array(frame))

        writer.close()

        # Ses birleştirme
        valid_segs = [s for s in all_audio_segments if s and os.path.exists(s)]
        if valid_segs:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            concat_list = os.path.join(OUTPUT_DIR, f'concat_{video_id}.txt')
            with open(concat_list, 'w') as f:
                for s in valid_segs:
                    f.write(f"file '{s}'\n")

            merged_audio = os.path.join(OUTPUT_DIR, f'audio_{video_id}.mp3')
            subprocess.run([
                ffmpeg_exe, '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_list, merged_audio
            ], capture_output=True, timeout=120)

            if os.path.exists(merged_audio):
                muxed = os.path.join(OUTPUT_DIR, f'muxed_{video_id}.mp4')
                subprocess.run([
                    ffmpeg_exe, '-y',
                    '-i', final_path,
                    '-i', merged_audio,
                    '-c:v', 'copy', '-c:a', 'aac',
                    '-shortest', muxed
                ], capture_output=True, timeout=120)
                if os.path.exists(muxed):
                    os.replace(muxed, final_path)

        JOB_STORE[job_id]['status']   = 'done'
        JOB_STORE[job_id]['progress'] = 100
        JOB_STORE[job_id]['result']   = os.path.basename(final_path)

    except Exception as e:
        JOB_STORE[job_id]['status'] = 'error'
        JOB_STORE[job_id]['error']  = str(e)
        print(f'Mux hata: {e}')


# ═══════════════════════════════════════════════════════════════════════════
# FLASK ROUTE'LARI
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ── Karakter ────────────────────────────────────────────────────────────────

@app.route('/character')
def character_page():
    return render_template('character.html', themes=STAGE_THEMES)


@app.route('/api/create-character', methods=['POST'])
def api_create_character():
    data  = request.json
    prompt = data.get('prompt', 'Komik bir komedyen')
    role   = data.get('role', 'comedian')
    color  = data.get('color', '#5060d0')
    cid     = generate_character_image(prompt, role, color)
    img_url = f'/static_standup/characters/{cid}.png'
    return jsonify({'cid': cid, 'img_url': img_url})


@app.route('/api/upload-character', methods=['POST'])
def api_upload_character():
    """Kullanıcının kendi fotoğrafından animasyon karakteri oluşturur."""
    if 'image' not in request.files:
        return jsonify({'error': 'Fotoğraf seçilmedi'}), 400
    img_file = request.files['image']
    if not img_file or img_file.filename == '':
        return jsonify({'error': 'Geçersiz dosya'}), 400

    role  = request.form.get('role', 'comedian')
    mode  = request.form.get('mode', 'bobblehead')
    color = request.form.get('color', '#2c3e50')

    try:
        cid = create_custom_photo_character(img_file, role=role, mode=mode, body_color=color)
        img_url = f'/static_standup/characters/{cid}.png'
        return jsonify({'cid': cid, 'img_url': img_url})
    except Exception as e:
        return jsonify({'error': f'Fotoğraf işlenemedi: {str(e)}'}), 500


@app.route('/api/upload-audio', methods=['POST'])
def api_upload_audio():
    """Kullanıcının kendi ses kaydını (mikrofon veya dosya) sisteme kaydeder."""
    if 'audio' not in request.files:
        return jsonify({'error': 'Ses dosyası seçilmedi'}), 400
    f = request.files['audio']
    if not f or f.filename == '':
        return jsonify({'error': 'Geçersiz dosya'}), 400

    aid = str(uuid.uuid4())[:8]
    ext = os.path.splitext(f.filename)[1].lower() or '.mp3'
    temp_path = os.path.join(AUDIO_DIR, f'temp_{aid}{ext}')
    f.save(temp_path)

    out_mp3 = os.path.join(AUDIO_DIR, f'{aid}.mp3')
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        subprocess.run([
            ffmpeg_exe, '-y', '-i', temp_path,
            '-c:a', 'libmp3lame', '-b:a', '128k', out_mp3
        ], capture_output=True, timeout=30)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
    except Exception:
        if not os.path.exists(out_mp3):
            os.rename(temp_path, out_mp3)

    dur = get_audio_duration(out_mp3)
    return jsonify({
        'aid': aid,
        'audio_url': f'/audio/{aid}.mp3',
        'duration': round(dur, 2)
    })



# ── Sahne / Senaryo / Mekan ──────────────────────────────────────────────────

@app.route('/api/generate-venue', methods=['POST'])
def api_generate_venue():
    """AI ile özel bar/sahne/mekan ortamı görseli üretir."""
    data = request.json or {}
    prompt = data.get('prompt', 'Loş ışıklı ahşap bar, raflarında içkiler, neon ışıklar')
    vid = str(uuid.uuid4())[:8]
    out_path = os.path.join(VENUE_DIR, f'{vid}.png')
    static_path = os.path.join(STATIC_VENUE_DIR, f'{vid}.png')

    try:
        en_prompt = f"cinematic photorealistic interior background of a standup comedy {prompt}, comedy club bar setting, empty stage center, 8k resolution, highly detailed"
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(en_prompt)}?width=1280&height=720&nologo=true&seed={int(time.time()) % 10000}"
        r = requests.get(url, timeout=25)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(out_path, 'wb') as f:
                f.write(r.content)
            import shutil
            shutil.copy2(out_path, static_path)
            return jsonify({'vid': vid, 'img_url': f'/static_standup/venues/{vid}.png'})
    except Exception as e:
        print(f'AI mekan üretim hata: {e}')

    return jsonify({'error': 'Mekan görseli üretilemedi, lütfen tekrar deneyin'}), 500


@app.route('/api/upload-venue', methods=['POST'])
def api_upload_venue():
    """Kullanıcının yüklediği bar veya mekan görselini kaydeder."""
    if 'image' not in request.files:
        return jsonify({'error': 'Görsel seçilmedi'}), 400
    img_file = request.files['image']
    if not img_file or img_file.filename == '':
        return jsonify({'error': 'Geçersiz dosya'}), 400

    vid = str(uuid.uuid4())[:8]
    out_path = os.path.join(VENUE_DIR, f'{vid}.png')
    static_path = os.path.join(STATIC_VENUE_DIR, f'{vid}.png')

    try:
        img = Image.open(img_file).convert('RGB')
        img = img.resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
        img.save(out_path, 'PNG')
        import shutil
        shutil.copy2(out_path, static_path)
        return jsonify({'vid': vid, 'img_url': f'/static_standup/venues/{vid}.png'})
    except Exception as e:
        return jsonify({'error': f'Mekan görseli kaydedilemedi: {str(e)}'}), 500


@app.route('/scene')
def scene_page():
    return render_template('scene.html', themes=STAGE_THEMES)


@app.route('/api/generate-script', methods=['POST'])
def api_generate_script():
    data     = request.json
    topic    = data.get('topic', 'Günlük hayat')
    duration = int(data.get('duration', 5))
    style    = data.get('style', 'eğlenceli ve alaycı')
    lines    = gpt_generate_script(topic, duration, style)
    return jsonify({'lines': lines})


# ── Seslendirme ─────────────────────────────────────────────────────────────

@app.route('/voiceover')
def voiceover_page():
    return render_template('voiceover.html')


@app.route('/api/preview-voice', methods=['POST'])
def api_preview_voice():
    data    = request.json
    text    = data.get('text', '')
    speaker = data.get('speaker', 'comedian')
    lang    = data.get('lang', 'tr')
    path    = text_to_speech(text, speaker, lang)
    if not path:
        return jsonify({'error': 'Ses üretilemedi'}), 500
    fname   = os.path.basename(path)
    return jsonify({'audio_url': f'/audio/{fname}'})


@app.route('/audio/<fname>')
def serve_audio(fname: str):
    path = os.path.join(AUDIO_DIR, fname)
    if not os.path.exists(path):
        return 'Bulunamadı', 404
    return send_file(path, mimetype='audio/mpeg')


# ── Video Üretimi ────────────────────────────────────────────────────────────

@app.route('/preview')
def preview_page():
    return render_template('preview.html')


@app.route('/api/start-video', methods=['POST'])
def api_start_video():
    project  = request.json   # {theme, lines, characters, lang}
    job_id   = str(uuid.uuid4())[:8]
    t = threading.Thread(target=_generate_video_worker, args=(job_id, project), daemon=True)
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/api/video-status/<job_id>')
def api_video_status(job_id: str):
    info = JOB_STORE.get(job_id, {'status': 'unknown'})
    return jsonify(info)


@app.route('/download/<fname>')
def download_video(fname: str):
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        return 'Video bulunamadı', 404
    return send_file(path, as_attachment=True,
                     download_name='standup_video.mp4',
                     mimetype='video/mp4')


# ── Session yardımcı ─────────────────────────────────────────────────────────

@app.route('/api/save-project', methods=['POST'])
def api_save_project():
    session['project'] = request.json
    return jsonify({'ok': True})


@app.route('/api/get-project')
def api_get_project():
    return jsonify(session.get('project', {}))


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, io
    # Windows konsolunda UTF-8 zorla
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 50)
    print("  Stand-Up Studyosu baslatiliyor...")
    print("  http://localhost:5001 adresini ac")
    print("=" * 50)
    has_openai = bool(OPENAI_KEY)
    has_eleven = bool(ELEVEN_KEY)
    print(f"  OpenAI  API : {'AKTIF' if has_openai else 'YOK (yerel fallback)'}")
    print(f"  ElevenLabs  : {'AKTIF' if has_eleven else 'YOK (gTTS fallback)'}")
    print()
    app.run(debug=True, port=5001, use_reloader=False)

