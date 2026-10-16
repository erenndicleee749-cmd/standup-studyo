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
# Statik karakterler ayni zamanda static_standup/characters'a da kopyalanir (serve icin)
STATIC_CHAR_DIR = os.path.join(BASE_DIR, 'static_standup', 'characters')
FONT_PATH    = os.path.join(BASE_DIR, 'static_standup', 'fonts', 'DejaVuSans-Bold.ttf')

for d in [CHAR_DIR, OUTPUT_DIR, AUDIO_DIR, STATIC_CHAR_DIR,
          os.path.join(BASE_DIR, 'static_standup', 'fonts')]:
    os.makedirs(d, exist_ok=True)

VIDEO_W, VIDEO_H = 1280, 720
FPS = 24

# ── Renk paleti ────────────────────────────────────────────────────────────
STAGE_THEMES = {
    'club':    {'bg': (15, 10, 30),    'spot': (255, 220, 80),  'name': '🍺 Gece Kulübü'},
    'theatre': {'bg': (30, 15, 15),    'spot': (255, 180, 60),  'name': '🎭 Tiyatro'},
    'street':  {'bg': (20, 40, 20),    'spot': (200, 240, 200), 'name': '🌆 Sokak Sahnesi'},
    'studio':  {'bg': (10, 10, 50),    'spot': (150, 180, 255), 'name': '🎬 Stüdyo'},
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
    """Karakter görseli üret — DALL·E veya yerel fallback."""
    cid = str(uuid.uuid4())[:8]
    out_path = os.path.join(CHAR_DIR, f'{cid}.png')

    if OPENAI_KEY:
        try:
            headers = {'Authorization': f'Bearer {OPENAI_KEY}',
                       'Content-Type': 'application/json'}
            full_prompt = (
                f"2D cartoon character, single full-body illustration on transparent background. "
                f"Role: {role}. Description: {prompt}. "
                f"Style: vibrant flat colors, thick outlines, comedic exaggerated features, "
                f"stand-up comedy club setting. No background, white canvas."
            )
            body = {'model': 'dall-e-3', 'prompt': full_prompt,
                    'n': 1, 'size': '1024x1024', 'response_format': 'b64_json'}
            r = requests.post('https://api.openai.com/v1/images/generations',
                              headers=headers, json=body, timeout=60)
            data = r.json()
            img_b64 = data['data'][0]['b64_json']
            img_bytes = base64.b64decode(img_b64)
            with open(out_path, 'wb') as f:
                f.write(img_bytes)
            _copy_to_static(cid)
            return cid
        except Exception as e:
            print(f'DALL-E hata: {e}')

    # ── Yerel fallback: geometrik karakter çiz ─────────────────────────────
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



def _draw_local_character(path: str, role: str, body_color: str, prompt: str):
    """Basit PIL karakter çizer."""
    W, H = 400, 600
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        r, g, b = int(body_color[1:3], 16), int(body_color[3:5], 16), int(body_color[5:7], 16)
    except Exception:
        r, g, b = 80, 120, 200

    skin = (255, 213, 170)
    # Gövde
    draw.ellipse([120, 220, 280, 420], fill=(r, g, b))
    # Kol sol
    draw.ellipse([60, 230, 130, 360], fill=(r, g, b))
    # Kol sağ
    draw.ellipse([270, 230, 340, 360], fill=(r, g, b))
    # Bacak sol
    draw.rectangle([140, 410, 185, 560], fill=(r//2, g//2, b//2))
    # Bacak sağ
    draw.rectangle([215, 410, 260, 560], fill=(r//2, g//2, b//2))
    # Baş
    draw.ellipse([140, 70, 260, 220], fill=skin)
    # Gözler
    draw.ellipse([160, 120, 185, 145], fill=(30, 30, 30))
    draw.ellipse([215, 120, 240, 145], fill=(30, 30, 30))
    # Göz parıltısı
    draw.ellipse([168, 126, 176, 134], fill=(255, 255, 255))
    draw.ellipse([223, 126, 231, 134], fill=(255, 255, 255))
    # Ağız (gülümseme)
    draw.arc([163, 155, 237, 205], start=10, end=170, fill=(180, 60, 60), width=4)

    # Rol etiketi
    fnt = _font(22)
    role_labels = {'comedian': '🎤 Komedyen', 'host': '🎙️ Sunucu',
                   'audience': '👥 Seyirci', 'comedian2': '🎤 Komedyen 2'}
    label = role_labels.get(role, role)
    draw.text((W // 2, 575), label, font=fnt, fill=(220, 220, 220, 230),
              anchor='mm')

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
    """TTS → mp3 dosyası döndür, boş metinde None."""
    if not text.strip():
        return None
    aid = str(uuid.uuid4())[:8]
    out = os.path.join(AUDIO_DIR, f'{aid}.mp3')

    # ElevenLabs
    if ELEVEN_KEY:
        voice_map = {
            'comedian':  'pNInz6obpgDQGcFmaJgB',  # Adam
            'comedian2': 'EXAVITQu4vr4xnSDxMaL',  # Bella
            'host':      'VR6AewLTigWG4xSOukaG',  # Arnold
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

    # gTTS fallback
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(out)
        return out
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

def _draw_stage_bg(draw: ImageDraw.ImageDraw, theme: dict):
    """Sahne arka planı çiz."""
    bg = theme['bg']
    spot = theme['spot']

    # Arka plan degradesi
    for y in range(VIDEO_H):
        ratio = y / VIDEO_H
        r = int(bg[0] * (1 - ratio * 0.5))
        g = int(bg[1] * (1 - ratio * 0.5))
        b = int(bg[2] * (1 - ratio * 0.5))
        draw.line([(0, y), (VIDEO_W, y)], fill=(r, g, b))

    # Spotlight
    cx, cy = VIDEO_W // 2, 0
    for radius in range(380, 0, -5):
        alpha = max(0, int(40 * (1 - radius / 380)))
        color = (min(255, spot[0] + alpha), min(255, spot[1] + alpha // 2),
                 max(0, spot[2] - alpha), alpha)
        draw.ellipse([cx - radius, cy - radius // 2,
                      cx + radius, cy + radius // 2], fill=color[:3])

    # Sahne zemini
    draw.polygon([(0, VIDEO_H - 140), (VIDEO_W, VIDEO_H - 140),
                  (VIDEO_W, VIDEO_H), (0, VIDEO_H)],
                 fill=(int(bg[0] * 1.5), int(bg[1] * 1.5), int(bg[2] * 1.5)))

    # Seyirci siluetleri
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


def _paste_character(frame: Image.Image, char_path: str,
                     x: int, y: int, scale: float = 1.0, mouth_open: bool = False):
    """Karakter PNG'yi frame'e yapıştır."""
    try:
        char = Image.open(char_path).convert('RGBA')
        new_w = int(char.width * scale)
        new_h = int(char.height * scale)
        char = char.resize((new_w, new_h), Image.LANCZOS)

        if mouth_open:
            # Basit ağız animasyonu: alt yarı hafif kaydır
            top = char.crop((0, 0, new_w, new_h // 2))
            bot = char.crop((0, new_h // 2, new_w, new_h))
            shifted = Image.new('RGBA', (new_w, new_h + 6), (0, 0, 0, 0))
            shifted.paste(top, (0, 0))
            shifted.paste(bot, (0, new_h // 2 + 6))
            char = shifted.crop((0, 0, new_w, new_h))

        paste_x = x - new_w // 2
        paste_y = y - new_h
        frame.paste(char, (paste_x, paste_y), char)
    except Exception as e:
        print(f'Karakter yapıştırma hata: {e}')


def _draw_subtitle(draw: ImageDraw.ImageDraw, text: str, speaker_name: str):
    """Altyazı bandı çiz."""
    if not text.strip():
        return
    fnt_big = _font(28)
    fnt_sm  = _font(20)
    band_y  = VIDEO_H - 100

    # Altyazı arka plan
    draw.rectangle([(0, band_y), (VIDEO_W, VIDEO_H - 145)],
                   fill=(0, 0, 0, 180))

    # Konuşmacı adı
    draw.text((VIDEO_W // 2, band_y + 8), speaker_name,
              font=fnt_sm, fill=(255, 200, 50), anchor='mt')

    # Metin (max 2 satır)
    wrapped = textwrap.fill(text, width=70)
    draw.text((VIDEO_W // 2, band_y + 32), wrapped,
              font=fnt_big, fill=(255, 255, 255), anchor='mt',
              align='center')


def _generate_video_worker(job_id: str, project: dict):
    """Arka planda video üret."""
    JOB_STORE[job_id] = {'status': 'running', 'progress': 0, 'result': None, 'error': None}

    try:
        import imageio
        theme_key = project.get('theme', 'club')
        theme     = STAGE_THEMES.get(theme_key, STAGE_THEMES['club'])
        lines     = project.get('lines', [])
        characters = project.get('characters', {})  # role -> cid
        lang      = project.get('lang', 'tr')

        char_paths = {}
        for role, cid in characters.items():
            p = os.path.join(CHAR_DIR, f'{cid}.png')
            if os.path.exists(p):
                char_paths[role] = p

        # Karakter pozisyonları
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

            # Ses üret
            audio_path = None
            duration   = 3.0
            if text.strip() and speaker not in ('audience_laugh', 'audience_clap'):
                audio_path = text_to_speech(text, speaker, lang)
                if audio_path:
                    duration = get_audio_duration(audio_path)

            elif speaker in ('audience_laugh', 'audience_clap'):
                duration = 2.5

            # Bu satır için frame sayısı
            n_frames = max(int(duration * FPS), FPS)

            for f_idx in range(n_frames):
                frame = Image.new('RGB', (VIDEO_W, VIDEO_H), theme['bg'])
                draw  = ImageDraw.Draw(frame)
                _draw_stage_bg(draw, theme)

                # Tüm karakterleri çiz (konuşan büyük, diğerleri küçük)
                for role, pos in CHAR_POS.items():
                    cp = char_paths.get(role)
                    if not cp:
                        continue
                    is_speaking = (role == speaker)
                    scale = 0.55 if is_speaking else 0.40
                    mouth_open = is_speaking and (f_idx % (FPS // 4) < FPS // 8)
                    _paste_character(frame, cp, pos[0], pos[1], scale, mouth_open)

                # Altyazı
                draw2 = ImageDraw.Draw(frame)
                _draw_subtitle(draw2, text if speaker not in
                               ('audience_laugh', 'audience_clap') else
                               ('😂😂😂' if speaker == 'audience_laugh' else '👏👏👏'),
                               speaker_names.get(speaker, speaker))

                writer.append_data(imageio.core.util.Array(frame))

            JOB_STORE[job_id]['progress'] = int((idx + 1) / total * 90)

        writer.close()

        # Ses dosyalarını video ile birleştir (basit concat yerine ffmpeg ile)
        # Sadece video dosyası dön; ses zaten ayrı üretildi ve frame başına sync edildi
        # Gerçek sync için moviepy kullan:
        _mux_audio_video(job_id, project, lines, video_path, char_paths, theme, lang, speaker_names)

    except Exception as e:
        JOB_STORE[job_id]['status'] = 'error'
        JOB_STORE[job_id]['error']  = str(e)
        print(f'Video üretim hata: {e}')


def _mux_audio_video(job_id, project, lines, silent_video, char_paths, theme, lang, speaker_names):
    """Ses dosyalarını video ile birleştir (ffmpeg concat)."""
    try:
        import imageio
        theme_key = project.get('theme', 'club')
        theme     = STAGE_THEMES.get(theme_key, STAGE_THEMES['club'])
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

            audio_path = None
            duration   = 3.0

            if text.strip() and speaker not in ('audience_laugh', 'audience_clap'):
                audio_path = text_to_speech(text, speaker, lang)
                if audio_path:
                    duration = get_audio_duration(audio_path)
                    all_audio_segments.append(audio_path)
                else:
                    all_audio_segments.append(None)
            else:
                duration = 2.5
                # Efekt sesleri (statik dosya)
                eff = 'laugh.mp3' if speaker == 'audience_laugh' else 'clap.mp3'
                eff_path = os.path.join(
                    BASE_DIR, 'static_standup', 'sounds', eff)
                if os.path.exists(eff_path):
                    all_audio_segments.append(eff_path)
                else:
                    all_audio_segments.append(None)

            n_frames = max(int(duration * FPS), FPS)
            for f_idx in range(n_frames):
                frame = Image.new('RGB', (VIDEO_W, VIDEO_H), theme['bg'])
                draw  = ImageDraw.Draw(frame)
                _draw_stage_bg(draw, theme)

                for role, pos in CHAR_POS.items():
                    cp = char_paths.get(role)
                    if not cp:
                        continue
                    is_speaking = (role == speaker)
                    scale = 0.55 if is_speaking else 0.40
                    mouth_open = is_speaking and (f_idx % (FPS // 4) < FPS // 8)
                    _paste_character(frame, cp, pos[0], pos[1], scale, mouth_open)

                draw2 = ImageDraw.Draw(frame)
                _draw_subtitle(draw2,
                               text if speaker not in ('audience_laugh', 'audience_clap')
                               else ('😂😂😂' if speaker == 'audience_laugh' else '👏👏👏'),
                               speaker_names.get(speaker, speaker))

                writer.append_data(imageio.core.util.Array(frame))

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


# ── Sahne / Senaryo ─────────────────────────────────────────────────────────

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

