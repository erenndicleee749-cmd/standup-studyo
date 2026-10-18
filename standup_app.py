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
import math
import re

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

PIPER_DIR    = os.path.join(DATA_DIR, 'piper_models')

for d in [CHAR_DIR, OUTPUT_DIR, AUDIO_DIR, VENUE_DIR, PIPER_DIR,
          STATIC_CHAR_DIR, STATIC_VENUE_DIR,
          os.path.join(BASE_DIR, 'static_standup', 'fonts')]:
    os.makedirs(d, exist_ok=True)

VIDEO_W, VIDEO_H = 1280, 720
FPS = 15  # Render ücretsiz planın zayıf CPU'su için optimize (çizgi animasyon için yeterince akıcı)

# ── Renk paleti ve mekan temaları ───────────────────────────────────────────
STAGE_THEMES = {
    'bar':       {'bg': (24, 14, 10),    'spot': (255, 190, 60),  'name': '🍺 Otantik Bar & Pub Ortamı'},
    'bar_photo': {'bg': (24, 14, 10),    'spot': (255, 190, 60),  'name': '📸 Gerçek Bar Fotoğrafı'},
    'club':      {'bg': (20, 12, 18),    'spot': (255, 210, 80),  'name': '🎙️ Komedi Kulübü & Bar'},
    'theatre':   {'bg': (35, 12, 18),    'spot': (255, 180, 60),  'name': '🎭 Kırmızı Perdeli Tiyatro'},
    'street':    {'bg': (18, 25, 35),    'spot': (200, 240, 255), 'name': '🌆 Açık Hava / Sokak'},
    'studio':    {'bg': (10, 10, 40),    'spot': (150, 180, 255), 'name': '🎬 Modern Talk-Show Stüdyosu'},
}

JOB_STORE: dict[str, dict] = {}  # job_id -> {status, progress, result}

# ═══════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════

def _font(size: int):
    candidates = [
        FONT_PATH,
        'C:/Windows/Fonts/arialbd.ttf',
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/segoeuib.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _clean_no_emoji(s: str) -> str:
    """Standart TTF fontlarının desteklemediği 4-byte emojileri temizler."""
    return re.sub(r'[\U00010000-\U0010ffff]', '', s or '').strip()


def generate_character_image(prompt: str, role: str, color: str) -> str:
    """Karakter görseli üret — Feride / Emir Sağlam 2D karikatür tarzında orijinal çizim."""
    cid = str(uuid.uuid4())[:8]
    out_path = os.path.join(CHAR_DIR, f'{cid}.png')

    # 1. Öncelik: Kullanıcı özellikle 3D / gerçekçi istemişse ve OpenAI Key varsa DALL-E 3
    p_lower = (prompt or '').lower()
    if OPENAI_KEY and ('3d' in p_lower or 'gerçekçi' in p_lower or 'pixar' in p_lower):
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

    # 2. Öncelik: Emir Sağlam / Feride animasyonu tarzında orijinal 2D vektörel karikatür çizimi
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


def _draw_local_character(path: str, role: str = 'comedian', body_color: str = '#2c3e50', prompt: str = ''):
    """Emir Sağlam / Feride (HBO Max / BluTV) 2D animasyon dizisi stilinde orijinal karakter çizimi."""
    W, H = 400, 600
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    p_lower = (prompt or '').lower()
    has_red_beret = ('kırmızı bere' in p_lower or 'kirmizi bere' in p_lower or 'bere' in p_lower)
    has_mustache  = ('bıyık' in p_lower or 'biyik' in p_lower or 'bıyıklı' in p_lower)
    is_female     = ('kadın' in p_lower or 'kadin' in p_lower or 'feride' in p_lower or 'kız' in p_lower)

    outline = (22, 18, 22)
    skin = (255, 226, 202)
    skin_shadow = (238, 202, 178)

    try:
        r, g, b = int(body_color[1:3], 16), int(body_color[3:5], 16), int(body_color[5:7], 16)
    except Exception:
        r, g, b = 44, 62, 80

    # 1. Bacaklar & Ayakkabılar
    draw.line([(180, 440), (180, 555)], fill=outline, width=8)
    draw.line([(220, 440), (220, 555)], fill=outline, width=8)
    draw.line([(180, 440), (180, 555)], fill=(35, 35, 40), width=4)
    draw.line([(220, 440), (220, 555)], fill=(35, 35, 40), width=4)
    draw.ellipse([158, 545, 192, 570], fill=(25, 25, 30), outline=outline, width=3)
    draw.ellipse([208, 545, 242, 570], fill=(25, 25, 30), outline=outline, width=3)

    # 2. Gövde
    if is_female:
        # Feride Orijinal Tişörtü (V yakalı, pembe şeritli)
        draw.polygon([
            (120, 310), (150, 260), (200, 252), (250, 260), (280, 310),
            (285, 440), (115, 440)
        ], fill=(250, 250, 255), outline=outline, width=4)
        # Pembe V-Yaka şeridi (Feride'nin meşhur tişörtü)
        draw.line([(175, 260), (200, 315)], fill=(235, 60, 95), width=5)
        draw.line([(225, 260), (200, 315)], fill=(235, 60, 95), width=5)
    else:
        # Barmen Tişörtü + Yeleği + Kırmızı Papyonu
        draw.polygon([
            (120, 310), (150, 260), (200, 252), (250, 260), (280, 310),
            (285, 440), (115, 440)
        ], fill=(245, 245, 250), outline=outline, width=4)
        draw.polygon([(120, 310), (150, 260), (180, 260), (170, 440), (115, 440)], fill=(r, g, b), outline=outline, width=3)
        draw.polygon([(280, 310), (250, 260), (220, 260), (230, 440), (285, 440)], fill=(r, g, b), outline=outline, width=3)
        # Kırmızı V-çizgisi ve papyon
        draw.line([(175, 260), (200, 305)], fill=(225, 40, 50), width=4)
        draw.line([(225, 260), (200, 305)], fill=(225, 40, 50), width=4)
        draw.polygon([(191, 280), (178, 270), (178, 292)], fill=(225, 40, 50), outline=outline, width=2)
        draw.polygon([(209, 280), (222, 270), (222, 292)], fill=(225, 40, 50), outline=outline, width=2)
        draw.ellipse([194, 274, 206, 286], fill=(190, 25, 35), outline=outline, width=2)

    # 3. İnce Uzun Zarif Boyun (Feride Dizisi İmzası)
    neck_pts = [
        (182, 195), (218, 195),
        (222, 235), (232, 258),
        (168, 258), (178, 235)
    ]
    draw.polygon(neck_pts, fill=skin, outline=outline, width=4)
    draw.polygon([(178, 242), (222, 242), (232, 258), (168, 258)], fill=skin_shadow)

    # 4. Kollar ve Mikrofon
    # Sol kol (Sarı/kırmızı manşetli, jest yapan kol)
    draw.polygon([(120, 310), (135, 318), (128, 335), (113, 327)], fill=(240, 180, 40), outline=outline, width=2)
    draw.line([(120, 325), (85, 370), (55, 345)], fill=outline, width=9)
    draw.line([(120, 325), (85, 370), (55, 345)], fill=skin, width=5)
    draw.polygon([(55, 345), (40, 332), (36, 344), (45, 356), (56, 355)], fill=skin, outline=outline, width=2)

    # Sağ kol (Mavi manşetli, mikrofon tutan kol)
    draw.polygon([(280, 310), (265, 318), (272, 335), (287, 327)], fill=(40, 150, 225), outline=outline, width=2)
    draw.line([(275, 325), (295, 365), (258, 295)], fill=outline, width=9)
    draw.line([(275, 325), (295, 365), (258, 295)], fill=skin, width=5)
    draw.ellipse([246, 285, 266, 305], fill=skin, outline=outline, width=3)

    # Stand-up Mikrofonu
    draw.line([(257, 300), (266, 350)], fill=(75, 75, 80), width=6)
    draw.ellipse([248, 275, 266, 296], fill=(50, 50, 55), outline=outline, width=2)
    draw.ellipse([251, 278, 263, 293], fill=(160, 160, 170))

    # 5. Kafa Yapısı (Feride'nin armut/damla formu)
    head_pts = []
    for ang in range(160, 380, 10):
        rad = math.radians(ang % 360)
        rx = 200 + int(64 * math.cos(rad))
        ry = 110 + int(56 * math.sin(rad))
        head_pts.append((rx, ry))
    head_pts.extend([
        (254, 135), (242, 165), (226, 192), (212, 208),
        (200, 210), (188, 208), (174, 192), (158, 165), (146, 135)
    ])
    draw.polygon(head_pts, fill=skin, outline=outline, width=4)

    # Kulaklar
    draw.arc([136, 118, 152, 146], start=90, end=270, fill=outline, width=3)
    draw.arc([248, 118, 264, 146], start=270, end=90, fill=outline, width=3)

    # 6. Saç / Şapka / Bere
    if is_female:
        # Feride Orijinal Turuncu Saçı + Beyaz Toka
        # At kuyruğu (sola doğru sarkan)
        draw.chord([95, 60, 175, 220], start=90, end=270, fill=(240, 120, 30), outline=outline, width=4)
        # Beyaz saç tokası (incili bant)
        for i, bx in enumerate([145, 156, 168]):
            draw.ellipse([bx, 60 - i*2, bx + 14, 74 - i*2], fill=(255, 255, 255), outline=outline, width=2)
        # Tepe saçı
        draw.chord([120, 30, 275, 140], start=180, end=360, fill=(240, 120, 30), outline=outline, width=4)
        # Alın perçemi (gözlerin yukarısına oturacak şekilde)
        draw.polygon([(140, 75), (200, 60), (255, 75), (245, 92), (185, 95), (145, 90)], fill=(240, 120, 30), outline=outline, width=3)
    elif has_red_beret:
        # Pofuduk yana yatık kırmızı Fransız beresi
        draw.ellipse([115, 25, 285, 96], fill=(225, 35, 45), outline=outline, width=4)
        draw.ellipse([135, 60, 265, 96], fill=(195, 25, 35), outline=outline, width=3)
        draw.rectangle([197, 12, 203, 27], fill=outline)
        draw.chord([138, 70, 154, 125], start=90, end=270, fill=(45, 32, 25), outline=outline, width=3)
        draw.chord([246, 70, 262, 125], start=270, end=90, fill=(45, 32, 25), outline=outline, width=3)
    else:
        # 🌟 Arda'nın İmzası: Hacimli Kıvırcık / Lüleli Saçları
        curl_col = (28, 22, 24)
        curl_border = (16, 12, 14)
        curls = [
            (135, 65, 24), (155, 48, 26), (182, 38, 28), (212, 36, 28), (242, 45, 27), (265, 62, 24),
            (125, 90, 22), (122, 115, 20), (124, 138, 18),
            (275, 90, 22), (278, 115, 20), (276, 138, 18),
            (145, 78, 20), (172, 62, 22), (200, 56, 24), (228, 62, 22), (252, 75, 20),
            (165, 82, 15), (195, 78, 16), (225, 82, 15)
        ]
        for cx, cy, cr in curls:
            draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=curl_col, outline=curl_border, width=2)
            draw.arc([cx - cr // 2, cy - cr // 2, cx + cr // 2, cy + cr // 2], start=30, end=240, fill=(48, 38, 42), width=2)

    # 7. Yüz Hatları (Feride İmzası)
    # İnce yay kaşlar
    draw.arc([165, 86, 192, 102], start=200, end=340, fill=outline, width=3)
    draw.arc([208, 86, 235, 102], start=200, end=340, fill=outline, width=3)

    # Gözler (Canlı siyah noktalar + parıltı)
    eye_y = 105
    draw.ellipse([172, eye_y, 186, eye_y + 14], fill=(20, 20, 25))
    draw.ellipse([214, eye_y, 228, eye_y + 14], fill=(20, 20, 25))
    draw.ellipse([175, eye_y + 2, 179, eye_y + 6], fill=(255, 255, 255))
    draw.ellipse([217, eye_y + 2, 221, eye_y + 6], fill=(255, 255, 255))

    # Yanak allığı
    draw.ellipse([152, 128, 172, 144], fill=(255, 195, 195))
    draw.ellipse([228, 128, 248, 144], fill=(255, 195, 195))

    # 8. İMZA UZUN KANCALI BURUN (Emir Sağlam)
    draw.line([(199, 100), (204, 150)], fill=outline, width=3)
    draw.arc([197, 142, 213, 158], start=270, end=90, fill=outline, width=3)

    # Bıyık (Sadece promptta istenmişse)
    if not is_female and has_mustache:
        draw.chord([182, 155, 201, 172], start=0, end=180, fill=(45, 32, 25), outline=outline, width=2)
        draw.chord([201, 155, 220, 172], start=0, end=180, fill=(45, 32, 25), outline=outline, width=2)

    # 9. İMZA FERİDE AĞZI: Yana kıvrık pembe esprili ağız
    draw.polygon([
        (188, 176), (198, 172), (214, 175),
        (216, 188), (204, 192), (188, 186)
    ], fill=(235, 75, 95), outline=outline, width=3)
    draw.line([(190, 178), (212, 178)], fill=outline, width=2)

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
    t_lower = (topic or '').lower()
    if 'barmen' in t_lower or 'bar' in t_lower or not topic or 'arda' in t_lower:
        return [
            {"speaker": "host",      "text": "Bayanlar baylar! Bu akşam tezgahın arkasından sahneye uzanan efsane barmenimiz Arda aramızda! Alkışlarınızla Arda sahnede!"},
            {"speaker": "audience_clap", "text": ""},
            {"speaker": "comedian",  "text": "Selam millet! Barda çalışınca insan psikolojisini çözüyorsunuz. Geçen bar taburesinde oturan bir abi geldi, 'Kardeşim bana öyle bir şey ver ki eski sevgilimi unuttursun' dedi. Ben de adisyona baktım, hesabı verdim. Anında unuttu!"},
            {"speaker": "audience_laugh", "text": ""},
            {"speaker": "comedian",  "text": "Bir de her masada bir kokteyl uzmanı var. Menüyü açıp 'Bunun içinde buz var mı?' diyor. Yahu mojito istedin, kutup ayısı mı bekliyordun?"},
            {"speaker": "audience_laugh", "text": ""},
            {"speaker": "comedian",  "text": "Gece ikide gelen müşteri var bir de. Tezgaha yaslanıp 'Barmen kardeşim, kadınları anlamak imkansız' diyor. Abi ben barmenim, filozof değilim! Ama bardağı doldururum, dinlerim!"},
            {"speaker": "audience_laugh", "text": ""},
            {"speaker": "comedian",  "text": "Gülmeye devam edin arkadaşlar, kadehleriniz hiç boş kalmasın! Şerefe!"},
            {"speaker": "audience_clap", "text": ""},
        ]

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
        {"speaker": "comedian",  "text": "Arkadaşlar, hayat zor ama gülmesek daha da zor olur. Teşekkür ederim, iyi geceler!"},
        {"speaker": "audience_clap", "text": ""},
    ]
    return lines


def _ensure_piper_model(model_name: str = 'tr_TR-fahrettin-medium') -> tuple[str, str]:
    """Türkçe nöral ses modelini hazır eder. Yoksa HuggingFace'ten indirir."""
    os.makedirs(PIPER_DIR, exist_ok=True)

    # 1. Öncelik: PIPER_DIR veya BASE_DIR/data_standup/piper_models içindeki dosyalar
    for base in [PIPER_DIR, os.path.join(BASE_DIR, 'data_standup', 'piper_models')]:
        o = os.path.join(base, f'{model_name}.onnx')
        j = os.path.join(base, f'{model_name}.onnx.json')
        if os.path.exists(o) and os.path.exists(j) and os.path.getsize(o) > 1000000:
            return o, j

    onnx_path = os.path.join(PIPER_DIR, f'{model_name}.onnx')
    json_path = os.path.join(PIPER_DIR, f'{model_name}.onnx.json')

    urls = {
        'tr_TR-fahrettin-medium': {
            'json': 'https://huggingface.co/rhasspy/piper-voices/raw/refs%2Fpr%2F25/tr/tr_TR/fahrettin/medium/tr_TR-fahrettin-medium.onnx.json',
            'onnx': 'https://huggingface.co/rhasspy/piper-voices/resolve/refs%2Fpr%2F25/tr/tr_TR/fahrettin/medium/tr_TR-fahrettin-medium.onnx'
        },
        'tr_TR-fettah-medium': {
            'json': 'https://huggingface.co/speaches-ai/piper-tr_TR-fettah-medium/resolve/main/config.json',
            'onnx': 'https://huggingface.co/speaches-ai/piper-tr_TR-fettah-medium/resolve/main/model.onnx'
        }
    }
    cfg = urls.get(model_name, urls['tr_TR-fahrettin-medium'])

    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
        try:
            r = requests.get(cfg['json'], timeout=30)
            with open(json_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f'Piper json indirme hata: {e}')

    if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) < 1000000:
        try:
            print(f'Piper {model_name} modeli indiriliyor (yaklaşık 60MB)...')
            r = requests.get(cfg['onnx'], stream=True, timeout=120)
            with open(onnx_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print('Model indirme tamamlandı!')
        except Exception as e:
            print(f'Piper onnx indirme hata: {e}')

    return onnx_path, json_path


def piper_text_to_speech(text: str, speaker: str = 'comedian') -> str | None:
    """Piper TTS ile yapay olmayan, doğal insan tonunda Türkçe ses üretimi."""
    if not text.strip():
        return None
    aid = str(uuid.uuid4())[:8]
    out_wav = os.path.join(AUDIO_DIR, f'piper_{aid}.wav')
    out_mp3 = os.path.join(AUDIO_DIR, f'{aid}.mp3')

    # Sunucu için tok 'fettah', komedyen için doğal vurgulu 'fahrettin'
    model_name = 'tr_TR-fettah-medium' if speaker == 'host' else 'tr_TR-fahrettin-medium'
    onnx_path, json_path = _ensure_piper_model(model_name)

    if not os.path.exists(onnx_path) or not os.path.exists(json_path):
        return None

    try:
        cmd = ['piper', '-m', onnx_path, '-c', json_path, '-f', out_wav]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        _, err = p.communicate(input=text, timeout=25)

        if p.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([
                ffmpeg_exe, '-y', '-i', out_wav,
                '-c:a', 'libmp3lame', '-b:a', '128k', out_mp3
            ], capture_output=True, timeout=15)

            try:
                os.remove(out_wav)
            except Exception:
                pass

            if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 0:
                return out_mp3
            return out_wav
        else:
            print(f'Piper hata: {err}')
    except Exception as e:
        print(f'Piper calistirma hata: {e}')

    return None


def text_to_speech(text: str, speaker: str, lang: str = 'tr') -> str | None:
    """TTS -> 1. OpenAI (varsa), 2. ElevenLabs (varsa), 3. Piper Doğal İnsan Sesi (Ücretsiz/Yerel), 4. gTTS Fallback."""
    if not text.strip():
        return None
    aid = str(uuid.uuid4())[:8]
    out = os.path.join(AUDIO_DIR, f'{aid}.mp3')

    # 1. Öncelik: OpenAI TTS (Varsa)
    if OPENAI_KEY:
        try:
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
            'comedian':  'pNInz6obpgDQGcFmaJgB',
            'comedian2': 'EXAVITQu4vr4xnSDxMaL',
            'host':      'VR6AewLTigWG4xSOukaG',
        }
        voice_id = voice_map.get(speaker, voice_map['comedian'])
        try:
            headers = {'xi-api-key': ELEVEN_KEY, 'Content-Type': 'application/json'}
            body = {'text': text, 'model_id': 'eleven_multilingual_v2',
                    'voice_settings': {'stability': 0.5, 'similarity_boost': 0.75}}
            r = requests.post(f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}', headers=headers, json=body, timeout=30)
            if r.status_code == 200:
                with open(out, 'wb') as f:
                    f.write(r.content)
                return out
        except Exception as e:
            print(f'ElevenLabs hata: {e}')

    # 3. Öncelik: Piper Nöral Ses Motoru (Yapay Olmayan, Gerçek İnsan Ses Tonu ve Vurgusu - 100% Ücretsiz)
    try:
        p_audio = piper_text_to_speech(text, speaker)
        if p_audio and os.path.exists(p_audio) and os.path.getsize(p_audio) > 1000:
            return p_audio
    except Exception as e:
        print(f'Piper fallback hata: {e}')

    # 4. Fallback: gTTS
    try:
        raw_out = os.path.join(AUDIO_DIR, f'raw_{aid}.mp3')
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(raw_out)
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

    # 1. Bar ortamı: Kullanıcının gerçek otantik bar fotoğrafını doğrudan arka plan yap
    if theme_key in ('bar', 'bar_photo'):
        real_bar_candidates = [
            custom_bg_path,
            os.path.join(STATIC_VENUE_DIR, 'real_bar.png'),
            os.path.join(VENUE_DIR, 'real_bar.png'),
            os.path.join(BASE_DIR, 'static_standup', 'venues', 'real_bar.png'),
            os.path.join(BASE_DIR, 'standup_data', 'venues', 'real_bar.png')
        ]
        for bp in real_bar_candidates:
            if bp and os.path.exists(bp):
                try:
                    bg_img = Image.open(bp).convert('RGB').resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
                    frame.paste(bg_img, (0, 0))
                    return
                except Exception as e:
                    print(f'Bar fotoğrafı yükleme hatası: {e}')

    if theme_key in ('bar_cartoon', 'club'):
        # 🍺 Otantik Ahşap Bar & Pub Atmosferi (Kullanıcı Referansına Uygun)
        # 1. Ahşap loş duvar degradesi
        for y in range(VIDEO_H - 140):
            ratio = y / (VIDEO_H - 140)
            r = int(28 - ratio * 14)
            g = int(16 - ratio * 8)
            b = int(12 - ratio * 6)
            draw.line([(0, y), (VIDEO_W, y)], fill=(r, g, b))

        # 2. ÜST TAVAN KADEH ASKILIĞI & TERS ASILI AYAKLI BARDAKLAR (Otantik Bar Askılığı)
        # Ahşap tavan kirişi / asma gantry
        draw.rectangle([(0, 0), (VIDEO_W, 45)], fill=(32, 18, 12), outline=(55, 30, 20), width=2)
        draw.line([(0, 45), (VIDEO_W, 45)], fill=(80, 45, 25), width=3)
        draw.line([(0, 52), (VIDEO_W, 52)], fill=(90, 85, 80), width=2)

        # Tavandan sarkan sıcak gömme tavan spotları
        spot_xs = [160, 360, 560, 760, 960, 1120]
        for sx in spot_xs:
            draw.ellipse([sx - 18, 38, sx + 18, 52], fill=(240, 210, 130), outline=(70, 40, 25), width=2)
            for r_step in range(1, 4):
                draw.polygon([
                    (sx - 10 - r_step * 3, 52),
                    (sx + 10 + r_step * 3, 52),
                    (sx + 35 + r_step * 12, 220),
                    (sx - 35 - r_step * 12, 220)
                ], outline=(255, 220 - r_step * 25, 90) if r_step == 1 else None)

        # Ters asılı ayaklı şarap ve kokteyl kadehleri
        for gx in range(60, VIDEO_W - 60, 32):
            draw.line([(gx - 8, 52), (gx + 8, 52)], fill=(210, 215, 225), width=2)
            stem_len = 38 + (gx % 3) * 6
            draw.line([(gx, 52), (gx, 52 + stem_len)], fill=(190, 205, 220), width=2)
            cy = 52 + stem_len
            bowl_w, bowl_h = 12, 24
            draw.polygon([
                (gx, cy),
                (gx - bowl_w, cy + bowl_h),
                (gx + bowl_w, cy + bowl_h)
            ], fill=(180, 210, 235), outline=(225, 235, 245), width=1)
            draw.ellipse([gx - bowl_w, cy + bowl_h - 2, gx + bowl_w, cy + bowl_h + 3], fill=(220, 235, 250), outline=(255, 255, 255), width=1)
            draw.line([(gx - 3, cy + 8), (gx - 3, cy + bowl_h - 4)], fill=(255, 245, 200), width=1)

        # 3. İKİ KATLI AHŞAP İÇKİ RAFLARI & ŞİŞELER
        draw.rectangle([(60, 165), (VIDEO_W - 60, 180)], fill=(65, 38, 22), outline=(100, 58, 35), width=2)
        draw.rectangle([(60, 285), (VIDEO_W - 60, 300)], fill=(65, 38, 22), outline=(100, 58, 35), width=2)

        bottle_cols = [
            (210, 100, 35), (40, 165, 80), (180, 40, 50),
            (230, 180, 50), (95, 45, 130), (220, 130, 40),
            (240, 200, 80), (160, 30, 30), (50, 130, 180)
        ]
        for i, sx in enumerate(range(85, VIDEO_W - 85, 38)):
            c1 = bottle_cols[i % len(bottle_cols)]
            bh1 = 58 + (i % 3) * 10
            draw.rectangle([(sx, 165 - bh1), (sx + 18, 165)], fill=c1, outline=(20, 12, 10), width=1)
            draw.rectangle([(sx + 5, 165 - bh1 - 12), (sx + 13, 165 - bh1)], fill=c1)
            draw.ellipse([(sx + 4, 165 - bh1 - 15), (sx + 14, 165 - bh1 - 10)], fill=(210, 190, 160))
            draw.rectangle([(sx + 2, 165 - bh1 + 18), (sx + 16, 165 - 10)], fill=(245, 240, 225))

            c2 = bottle_cols[(i + 3) % len(bottle_cols)]
            bh2 = 52 + ((i * 2) % 3) * 12
            draw.rectangle([(sx, 285 - bh2), (sx + 18, 285)], fill=c2, outline=(20, 12, 10), width=1)
            draw.rectangle([(sx + 5, 285 - bh2 - 12), (sx + 13, 285 - bh2)], fill=c2)
            draw.ellipse([(sx + 4, 285 - bh2 - 15), (sx + 14, 285 - bh2 - 10)], fill=(210, 190, 160))
            draw.rectangle([(sx + 2, 285 - bh2 + 16), (sx + 16, 285 - 10)], fill=(235, 230, 215))

        # 4. PARLAYAN NEON BAR TABELASI
        neon_txt = '🍻 STAND-UP COMEDY BAR & PUB 🍻'
        fnt_neon = _font(22)
        try:
            nb = draw.textbbox((0, 0), neon_txt, font=fnt_neon)
            nw = nb[2] - nb[0]
            nh = nb[3] - nb[1]
        except Exception:
            nw, nh = len(neon_txt) * 12, 24
        nx = (VIDEO_W - nw) // 2
        draw.rounded_rectangle([nx - 24, 102, nx + nw + 24, 102 + nh + 18], radius=10, fill=(25, 12, 20), outline=(255, 65, 130), width=3)
        draw.rounded_rectangle([nx - 20, 106, nx + nw + 20, 102 + nh + 14], radius=8, outline=(255, 140, 185), width=1)
        draw.text((nx, 110), neon_txt, fill=(255, 160, 200), font=fnt_neon)

        # 5. Sahne Spotu (Barmenin üzerine vuran altın ışık konisi)
        cx, cy = VIDEO_W // 2, 0
        for radius in range(400, 0, -15):
            draw.ellipse([cx - radius, cy - radius // 2, cx + radius, cy + radius // 2],
                         outline=(255, 210, 100))

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
        _draw_audience(draw)

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


def _sbox(x1, y1, x2, y2):
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _draw_bar_counter_and_patrons(draw: ImageDraw.ImageDraw, is_cheering: bool = False, t_sec: float = 0.0):
    """Barmenin önündeki cilalı ahşap bar tezgahını ve bar sandalyelerinde oturan müşterileri çizer."""
    bar_top_y = VIDEO_H - 185
    outline = (22, 18, 22)
    skin = (255, 226, 202)

    # 1. Cilalı Ahşap Bar Tezgahı (Barmenin önündedir)
    draw.polygon([
        (0, bar_top_y), (VIDEO_W, bar_top_y),
        (VIDEO_W, bar_top_y + 40), (0, bar_top_y + 40)
    ], fill=(95, 48, 25), outline=(135, 75, 40), width=3)
    draw.rectangle([(0, bar_top_y + 40), (VIDEO_W, VIDEO_H)], fill=(45, 22, 12))

    # 2. Altın/Krom Bira Musluğu & Kokteyl Shaker
    tap_x = VIDEO_W // 2 + 160
    draw.rectangle([(tap_x, bar_top_y - 65), (tap_x + 16, bar_top_y)], fill=(195, 195, 205), outline=(110, 110, 120), width=2)
    draw.polygon([(tap_x - 8, bar_top_y - 65), (tap_x + 24, bar_top_y - 65), (tap_x + 18, bar_top_y - 48), (tap_x - 3, bar_top_y - 48)], fill=(220, 180, 50))
    shk_x = VIDEO_W // 2 - 160
    draw.polygon([(shk_x, bar_top_y - 45), (shk_x + 22, bar_top_y - 45), (shk_x + 18, bar_top_y), (shk_x + 4, bar_top_y)], fill=(185, 190, 200), outline=(110, 110, 120), width=2)

    # 3. Bar Sandalyelerinde Oturan Müşteriler (Feride Karikatür Stili)
    def _draw_patron(px, side='left', p_type=0):
        stool_y = VIDEO_H - 150
        # Bar Taburesi (Kırmızı deri minder + krom ayaklar + ayak dayama halkası)
        draw.ellipse(_sbox(px - 28, stool_y, px + 28, stool_y + 14), fill=(160, 35, 45), outline=outline, width=2)
        draw.line([(px, stool_y + 12), (px, VIDEO_H - 25)], fill=(180, 180, 190), width=4)
        draw.line([(px - 18, stool_y + 35), (px - 25, VIDEO_H - 25)], fill=(120, 120, 130), width=3)
        draw.line([(px + 18, stool_y + 35), (px + 25, VIDEO_H - 25)], fill=(120, 120, 130), width=3)
        draw.ellipse(_sbox(px - 22, stool_y + 55, px + 22, stool_y + 67), outline=(190, 190, 200), width=3)

        face_dir = 1 if side == 'left' else -1  # Barmen Arda'ya doğru bakar
        cheer_lift = int(abs(math.sin(t_sec * 8 + px)) * 10) if is_cheering else 0
        body_y = stool_y - 12 - cheer_lift

        jacket_cols = [(52, 73, 94), (192, 57, 43), (39, 174, 96), (142, 68, 173)]
        j_col = jacket_cols[p_type % len(jacket_cols)]

        # Bacaklar (Ayak dayamaya basan)
        draw.line([(px, body_y + 10), (px + face_dir * 14, stool_y + 58)], fill=(30, 35, 45), width=6)
        draw.ellipse(_sbox(px + face_dir * 10, stool_y + 54, px + face_dir * 24, stool_y + 64), fill=(20, 20, 25))

        # Gövde
        draw.polygon([
            (px - 22, body_y + 12), (px + 22, body_y + 12),
            (px + face_dir * 18, body_y - 65), (px - face_dir * 14, body_y - 65)
        ], fill=j_col, outline=outline, width=3)

        # Kafa ve Boyun
        head_cx = px + face_dir * 8
        head_cy = body_y - 95
        draw.rectangle(_sbox(head_cx - 5, body_y - 70, head_cx + 5, body_y - 60), fill=skin, outline=outline, width=2)
        draw.ellipse(_sbox(head_cx - 18, head_cy - 22, head_cx + 18, head_cy + 18), fill=skin, outline=outline, width=3)

        # Saç / Şapka
        if p_type == 0:
            draw.chord(_sbox(head_cx - 18, head_cy - 25, head_cx + 18, head_cy - 5), start=180, end=360, fill=(230, 190, 60), outline=outline, width=2)
            draw.rectangle(_sbox(head_cx + face_dir * 2, head_cy - 6, head_cx + face_dir * 14, head_cy + 4), fill=(255, 255, 255), outline=outline, width=2)
        elif p_type == 1:
            draw.ellipse(_sbox(head_cx - 20, head_cy - 28, head_cx + 20, head_cy - 12), fill=(210, 45, 55), outline=outline, width=2)
        elif p_type == 2:
            draw.ellipse(_sbox(head_cx - face_dir * 14, head_cy - 30, head_cx - face_dir * 2, head_cy - 14), fill=(60, 40, 30), outline=outline, width=2)
            draw.chord(_sbox(head_cx - 18, head_cy - 25, head_cx + 18, head_cy - 2), start=180, end=360, fill=(60, 40, 30), outline=outline, width=2)
            draw.polygon([(head_cx - 10, body_y - 62), (head_cx + 10, body_y - 62), (head_cx, body_y - 48)], fill=(230, 60, 80))
        else:
            draw.chord(_sbox(head_cx - 18, head_cy - 25, head_cx + 18, head_cy - 5), start=180, end=360, fill=(50, 45, 45), outline=outline, width=2)
            draw.ellipse(_sbox(head_cx + face_dir * 6, head_cy + 4, head_cx + face_dir * 16, head_cy + 10), fill=(30, 25, 25))

        # Göz & Gülüş
        draw.ellipse(_sbox(head_cx + face_dir * 8, head_cy - 4, head_cx + face_dir * 14, head_cy + 2), fill=(20, 20, 25))
        if is_cheering:
            draw.chord(_sbox(head_cx + face_dir * 6, head_cy + 8, head_cx + face_dir * 18, head_cy + 18), start=0, end=180, fill=(120, 20, 30), outline=outline, width=2)
        else:
            draw.arc(_sbox(head_cx + face_dir * 6, head_cy + 6, head_cx + face_dir * 16, head_cy + 14), start=0, end=180, fill=outline, width=2)

        # Kadeh Kaldıran Kol
        glass_lift = 38 if is_cheering else 0
        hand_x = px + face_dir * 32
        hand_y = body_y - 35 - glass_lift
        draw.line([(px + face_dir * 8, body_y - 50), (hand_x, hand_y + 12)], fill=j_col, width=6)
        draw.line([(px + face_dir * 8, body_y - 50), (hand_x, hand_y + 12)], fill=outline, width=2)
        draw.ellipse(_sbox(hand_x - 4, hand_y + 8, hand_x + 8, hand_y + 18), fill=skin, outline=outline, width=2)

        # Bira Bardağı veya Kokteyl Kadehi
        if p_type % 2 == 0:
            draw.rectangle(_sbox(hand_x + face_dir * 4, hand_y - 8, hand_x + face_dir * 4 + 16, hand_y + 16), fill=(240, 185, 45), outline=(255, 255, 255), width=2)
            draw.ellipse(_sbox(hand_x + face_dir * 4 - 2, hand_y - 14, hand_x + face_dir * 4 + 18, hand_y - 4), fill=(255, 255, 255))
        else:
            bx1, bx2 = hand_x + face_dir * 4, hand_x + face_dir * 4 + 18
            draw.polygon([
                (bx1, hand_y - 12), (bx2, hand_y - 12),
                ((bx1 + bx2) // 2, hand_y + 6)
            ], fill=(235, 75, 120), outline=(255, 255, 255), width=2)
            draw.line([((bx1 + bx2) // 2, hand_y + 6), ((bx1 + bx2) // 2, hand_y + 16)], fill=(255, 255, 255), width=2)

    _draw_patron(110, side='left', p_type=0)
    _draw_patron(230, side='left', p_type=1)
    _draw_patron(VIDEO_W - 230, side='right', p_type=2)
    _draw_patron(VIDEO_W - 110, side='right', p_type=3)


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


_SPRITE_CACHE: dict = {}


def _build_sprite(char_path: str, scale: float, mouth_state: int, blink: bool, tilt_q: int) -> Image.Image:
    """Karakter sprite'ını (boyutlandırılmış, ağız modu, göz kırpma, eğik) üretir ve önbelleğe alır."""
    key = (char_path, scale, mouth_state, blink, tilt_q)
    cached = _SPRITE_CACHE.get(key)
    if cached is not None:
        return cached

    if tilt_q != 0:
        base = _build_sprite(char_path, scale, mouth_state, blink, 0)
        sprite = base.rotate(tilt_q, resample=Image.BICUBIC, expand=True)
        _SPRITE_CACHE[key] = sprite
        return sprite

    char_raw = Image.open(char_path).convert('RGBA')
    is_vector_sprite = (char_raw.width == 400 and char_raw.height == 600)

    new_w = max(1, int(char_raw.width * scale))
    new_h = max(1, int(char_raw.height * scale))

    # Fotogerçekçi karakterler için pürüzsüz konuşma ve nefes alma animasyonu
    if not is_vector_sprite:
        if mouth_state == 1:
            # Konuşma anında hafif doğal çene/gövde hareketi
            new_h = int(new_h * 1.015)
        elif mouth_state == 2:
            new_w = int(new_w * 1.02)
            new_h = int(new_h * 1.02)
        char = char_raw.resize((new_w, new_h), Image.LANCZOS)
        _SPRITE_CACHE[key] = char
        return char

    char = char_raw.resize((new_w, new_h), Image.LANCZOS)
    draw = ImageDraw.Draw(char)

    outline = (22, 18, 22)
    skin = (255, 226, 202)
    sx, sy = new_w / 400.0, new_h / 600.0

    # 1. Göz kırpma animasyonu (blink)
    if blink:
        draw.ellipse([156 * sx, 95 * sy, 188 * sx, 122 * sy], fill=skin)
        draw.ellipse([206 * sx, 95 * sy, 238 * sx, 122 * sy], fill=skin)
        draw.arc([158 * sx, 98 * sy, 186 * sx, 120 * sy], start=200, end=340, fill=outline, width=max(2, int(3 * sx)))
        draw.arc([208 * sx, 98 * sy, 236 * sx, 120 * sy], start=200, end=340, fill=outline, width=max(2, int(3 * sx)))

    # 2. Ağız durumu (mouth_state: 0 = tebessüm/dinleme, 1 = konuşma, 2 = kahkaha/coşku)
    if mouth_state == 1:
        draw.ellipse([180 * sx, 168 * sy, 222 * sx, 195 * sy], fill=skin)
        mx1, my1, mx2, my2 = 184 * sx, 170 * sy, 218 * sx, 194 * sy
        draw.ellipse([mx1, my1, mx2, my2], fill=(65, 15, 25), outline=outline, width=max(2, int(2.5 * sx)))
        draw.chord([mx1 + 4 * sx, my1 + 1 * sy, mx2 - 4 * sx, my1 + 10 * sy], start=0, end=180, fill=(255, 255, 255))
        draw.ellipse([mx1 + 7 * sx, my2 - 8 * sy, mx2 - 7 * sx, my2 - 1 * sy], fill=(235, 65, 85))

    elif mouth_state == 2:
        draw.ellipse([175 * sx, 165 * sy, 227 * sx, 202 * sy], fill=skin)
        mx1, my1, mx2, my2 = 178 * sx, 168 * sy, 224 * sx, 200 * sy
        draw.chord([mx1, my1, mx2, my2], start=0, end=180, fill=(60, 10, 20), outline=outline, width=max(2, int(2.5 * sx)))
        draw.rectangle([mx1 + 6 * sx, my1, mx2 - 6 * sx, my1 + 6 * sy], fill=(255, 255, 255))
        draw.chord([mx1 + 9 * sx, my2 - 12 * sy, mx2 - 9 * sx, my2], start=0, end=180, fill=(235, 65, 85))

    _SPRITE_CACHE[key] = char
    return char


def _paste_character(frame: Image.Image, char_path: str,
                     x: int, y: int, scale: float = 1.0, mouth_state: int = 0, blink: bool = False, bounce: int = 0, tilt: float = 0.0):
    """Karakter PNG'yi frame'e yapıştırır, konuşurken ağzını açar ve sahnede canlı animasyon uygular."""
    try:
        tilt_q = int(round(tilt)) if abs(tilt) > 0.4 else 0
        char = _build_sprite(char_path, scale, mouth_state, blink, tilt_q)
        paste_x = x - char.width // 2
        paste_y = y - char.height - bounce
        frame.paste(char, (paste_x, paste_y), char)
    except Exception as e:
        print(f'Karakter yapıştırma hata: {e}')


def _draw_subtitle(draw: ImageDraw.ImageDraw, text: str, speaker_name: str):
    """Altyazı bandı çiz (Pillow anchor hatası vermeyen güvenli çizim)."""
    text = _clean_no_emoji(text)
    speaker_name = _clean_no_emoji(speaker_name) or "Konuşmacı"
    if not text.strip():
        return
    fnt_big = _font(26)
    fnt_sm  = _font(19)
    band_y  = VIDEO_H - 125

    # Altyazı arka plan
    draw.rectangle([(0, band_y), (VIDEO_W, VIDEO_H - 10)],
                   fill=(0, 0, 0, 205))

    # Konuşmacı adı
    try:
        sb = draw.textbbox((0, 0), speaker_name, font=fnt_sm)
        sw = sb[2] - sb[0]
    except Exception:
        sw = len(speaker_name) * 10
    draw.text(((VIDEO_W - sw) // 2, band_y + 8), speaker_name,
              font=fnt_sm, fill=(255, 205, 50))

    # Metin (max 2 satır, satır satır ortalanır — Pillow anchor hatası engellenir)
    wrapped_lines = textwrap.wrap(text, width=65)[:2]
    curr_y = band_y + 36
    for l in wrapped_lines:
        try:
            lb = draw.textbbox((0, 0), l, font=fnt_big)
            lw = lb[2] - lb[0]
        except Exception:
            lw = len(l) * 13
        draw.text(((VIDEO_W - lw) // 2, curr_y), l,
                  font=fnt_big, fill=(255, 255, 255))
        curr_y += 32


def resolve_line_audio(line: dict, lang: str = 'tr', voice_settings: dict = None):
    """Her satır için ses dosyasını (kullanıcı sesi, efekt veya TTS) ve süresini belirler."""
    speaker = line.get('speaker', 'comedian')
    actual_voice = (voice_settings or {}).get(speaker, speaker)
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

    # 3. Metin varsa TTS üretimi (Seçilen erkek/kadın tonuyla)
    if text.strip():
        audio_path = text_to_speech(text, actual_voice, lang)
        if audio_path:
            dur = get_audio_duration(audio_path)
            return audio_path, max(1.5, dur)

    return None, 3.0


def _generate_video_worker(job_id: str, project: dict):
    """Arka planda video üret (Tek geçişli, yüksek performanslı ve hatasız ses birleştirmeli)."""
    JOB_STORE[job_id] = {'status': 'running', 'progress': 0, 'result': None, 'error': None}

    try:
        import imageio, math, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        theme_key = project.get('theme', 'bar')
        lines     = project.get('lines', [])
        characters = project.get('characters', {})  # role -> cid
        lang           = project.get('lang', 'tr')
        voice_settings = project.get('voice_settings', {})
        custom_bg      = project.get('custom_bg')

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

        # 🌟 Otomatik Barmen Arda Garantisi (Kullanıcı karakter seçmese bile sahne ASLA boş kalmaz!)
        default_arda_path = os.path.join(CHAR_DIR, 'arda_default.png')
        if not os.path.exists(default_arda_path):
            _draw_local_character(default_arda_path, 'comedian', '#2c3e50', 'arda kıvırcık saçlı barmen')
            _copy_to_static('arda_default')

        if not char_paths:
            char_paths['comedian'] = default_arda_path

        # Eğer senaryoda geçen komedyen rolleri atanmamışsa Arda'ya bağla
        for line in lines:
            spk = line.get('speaker')
            if spk in ('comedian', 'comedian2') and spk not in char_paths:
                char_paths[spk] = char_paths.get('comedian', default_arda_path)

        if theme_key in ('bar', 'bar_photo', 'club'):
            CHAR_POS = {
                'comedian':  (640, 515),
                'comedian2': (640, 515),
            }
        else:
            CHAR_POS = {
                'comedian':  (VIDEO_W // 2, VIDEO_H - 155),
                'comedian2': (VIDEO_W // 3, VIDEO_H - 155),
                'host':      (VIDEO_W // 4, VIDEO_H - 155),
            }

        video_id = str(uuid.uuid4())[:8]
        raw_video_path = os.path.join(OUTPUT_DIR, f'raw_{video_id}.mp4')
        final_path     = os.path.join(OUTPUT_DIR, f'final_{video_id}.mp4')

        total = len(lines)
        speaker_names = {
            'comedian': 'Arda (Barmen)', 'comedian2': 'Arda (Barmen)',
            'host': 'Sunucu', 'audience_laugh': 'Kahkaha',
            'audience_clap': 'Alkış'
        }

        # ── AŞAMA 1: Tüm seslendirmeleri üret (%0 → %30) ─────────────────
        JOB_STORE[job_id]['stage'] = 'Doğal insan seslendirmeleri hazırlanıyor'
        line_audio = []   # [(audio_path|None, n_frames)]
        for idx, line in enumerate(lines):
            audio_path, duration = resolve_line_audio(line, lang, voice_settings)
            n_frames = max(int(round(duration * FPS)), FPS)
            line_audio.append((audio_path, n_frames))
            JOB_STORE[job_id]['progress'] = int((idx + 1) / max(1, total) * 30)

        total_frames = max(1, sum(n for _, n in line_audio))

        # ── AŞAMA 2: Kareleri çiz (%30 → %85) ─────────────────────────────
        JOB_STORE[job_id]['stage'] = 'Bar ve karikatür animasyon kareleri çiziliyor'
        stage_bg = Image.new('RGB', (VIDEO_W, VIDEO_H), (0, 0, 0))
        _draw_stage_bg(stage_bg, theme_key, custom_bg_path=custom_bg_path)

        writer = imageio.get_writer(raw_video_path, fps=FPS,
                                    codec='libx264', quality=6,
                                    macro_block_size=None,
                                    pixelformat='yuv420p',
                                    ffmpeg_params=['-preset', 'ultrafast'])

        frames_done = 0
        for idx, line in enumerate(lines):
            speaker = line.get('speaker', 'comedian')
            text    = line.get('text', '')
            n_frames = line_audio[idx][1]
            is_cheer = speaker in ('audience_laugh', 'audience_clap')

            for f_idx in range(n_frames):
                frame = stage_bg.copy()
                t_sec = f_idx / FPS

                # 1. Barmen / Komedyen Karakterleri (Tezgah arkasında)
                for role, pos in CHAR_POS.items():
                    cp = char_paths.get(role)
                    if not cp:
                        continue
                    is_speaking = (role == speaker)
                    scale = 0.48 if role == 'comedian' else 0.45

                    # Doğal göz kırpma (Her ~3.3 saniyede 0.16 sn sevimli göz kırpma)
                    blink = ((t_sec % 3.3) < 0.16)

                    # Ağız durumu (0 = tebessüm, 1 = konuşma, 2 = kahkaha/coşku)
                    if is_speaking:
                        mouth_state = 1 if (int(t_sec * 8) % 2 == 1) else 0
                    elif is_cheer and role == 'comedian':
                        mouth_state = 2
                    else:
                        mouth_state = 0

                    bounce = int(abs(math.sin(t_sec * 6)) * 3) if is_speaking else int(math.sin(t_sec * 2.0) * 1.5)
                    tilt = math.sin(t_sec * 4.0) * 1.2 if is_speaking else 0.0

                    _paste_character(frame, cp, pos[0], pos[1], scale,
                                     mouth_state=mouth_state, blink=blink, bounce=bounce, tilt=tilt)

                # 2. Karikatür Bar Tezgahı ve Müşteriler Katmanı (Sadece bar_cartoon modunda çizilir)
                if theme_key == 'bar_cartoon':
                    _draw_bar_counter_and_patrons(ImageDraw.Draw(frame), is_cheering=is_cheer, t_sec=t_sec)

                # 3. Altyazı Bandı
                if is_cheer:
                    sub_txt = '🍻 (Müşteriler Kadeh Kaldırıyor!)' if speaker == 'audience_laugh' else '👏 (Müşteriler Alkışlıyor!)'
                else:
                    sub_txt = text
                _draw_subtitle(ImageDraw.Draw(frame), sub_txt, speaker_names.get(speaker, speaker))

                writer.append_data(np.asarray(frame))
                frames_done += 1
                if frames_done % 10 == 0:
                    JOB_STORE[job_id]['progress'] = 30 + int(frames_done / total_frames * 55)

        writer.close()
        _SPRITE_CACHE.clear()

        # ── AŞAMA 3: Ses-video birleştirme (%85 → %100) ──────────────────
        JOB_STORE[job_id]['stage'] = 'Ses ve video birleştiriliyor'
        JOB_STORE[job_id]['progress'] = 88
        has_any_audio = any(a and os.path.exists(a) for a, _ in line_audio)
        if has_any_audio:
            norm_segments = []
            for s_idx, (a_path, n_frames) in enumerate(line_audio):
                seg_dur = f'{n_frames / FPS:.3f}'
                seg_wav = os.path.join(OUTPUT_DIR, f'norm_{video_id}_{s_idx}.wav')
                if a_path and os.path.exists(a_path):
                    # Satır sesini tam olarak kare süresine sabitle (kayma olmaz)
                    cmd = [ffmpeg_exe, '-y', '-i', a_path,
                           '-af', 'apad', '-t', seg_dur,
                           '-ar', '44100', '-ac', '2', seg_wav]
                else:
                    # Ses yoksa aynı uzunlukta sessizlik
                    cmd = [ffmpeg_exe, '-y', '-f', 'lavfi',
                           '-i', 'anullsrc=r=44100:cl=stereo',
                           '-t', seg_dur, seg_wav]
                subprocess.run(cmd, capture_output=True, timeout=60)
                if os.path.exists(seg_wav):
                    norm_segments.append(seg_wav)

            if norm_segments:
                concat_list = os.path.join(OUTPUT_DIR, f'concat_{video_id}.txt')
                with open(concat_list, 'w', encoding='utf-8') as f:
                    for ns in norm_segments:
                        safe_path = ns.replace('\\', '/')
                        f.write(f"file '{safe_path}'\n")

                merged_audio = os.path.join(OUTPUT_DIR, f'audio_{video_id}.wav')
                subprocess.run([
                    ffmpeg_exe, '-y', '-f', 'concat', '-safe', '0',
                    '-i', concat_list, '-c:a', 'pcm_s16le', merged_audio
                ], capture_output=True, timeout=120)

                if os.path.exists(merged_audio):
                    subprocess.run([
                        ffmpeg_exe, '-y',
                        '-i', raw_video_path,
                        '-i', merged_audio,
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                        '-movflags', '+faststart',
                        '-shortest', final_path
                    ], capture_output=True, timeout=180)

                # Geçici wav parçalarını temizle
                for ns in norm_segments:
                    try:
                        os.remove(ns)
                    except Exception:
                        pass
                try:
                    if os.path.exists(concat_list): os.remove(concat_list)
                    if os.path.exists(merged_audio): os.remove(merged_audio)
                except Exception:
                    pass

        if not os.path.exists(final_path):
            if os.path.exists(raw_video_path):
                os.replace(raw_video_path, final_path)
        else:
            if os.path.exists(raw_video_path):
                try:
                    os.remove(raw_video_path)
                except Exception:
                    pass

        JOB_STORE[job_id]['status']   = 'done'
        JOB_STORE[job_id]['progress'] = 100
        JOB_STORE[job_id]['result']   = os.path.basename(final_path)

    except Exception as e:
        JOB_STORE[job_id]['status'] = 'error'
        JOB_STORE[job_id]['error']  = str(e)
        print(f'Video üretim hata: {e}')


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
    """Önizleme sesi üretir veya efekt sesini döner."""
    data    = request.json or {}
    text    = data.get('text', '')
    speaker = data.get('speaker', 'comedian')
    lang    = data.get('lang', 'tr')

    # Efekt sesleri (Kahkaha veya alkış)
    if speaker in ('audience_laugh', 'audience_clap'):
        eff = 'laugh.mp3' if speaker == 'audience_laugh' else 'clap.mp3'
        return jsonify({'audio_url': f'/static_standup/sounds/{eff}'})

    if not text.strip():
        text = 'Merhaba sevgili seyirciler, bu akşamki gösterime hoş geldiniz!'

    path = text_to_speech(text, speaker, lang)
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Ses üretilemedi, lütfen tekrar deneyin'}), 500
    fname = os.path.basename(path)
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
    """Kullanıcının MP4 dosyasını doğrudan cihazına indirmesini sağlar."""
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        return 'Video bulunamadı', 404
    return send_file(path, as_attachment=True,
                     download_name='standup_bar_komedi.mp4',
                     mimetype='video/mp4')


@app.route('/video/<fname>')
def stream_video(fname: str):
    """Tarayıcı içi video oynatıcının takılmadan izlenebilmesi için inline video yayını."""
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        return 'Video bulunamadı', 404
    return send_file(path, as_attachment=False,
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

