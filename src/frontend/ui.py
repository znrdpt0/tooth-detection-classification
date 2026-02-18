import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import os
st.set_page_config(page_title="Dental AI Diagnosis", layout="wide")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
COLOR_MAP = {
    "Caries": "red",              
    "Deep_Caries": "darkred",     
    "Periapical_Lesion": "orange",
    "Impacted": "blue"          
}

def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    
    img_w, img_h = image.size
    font_size = max(20, int(img_h * 0.035)) 
    
    try:
        # Mac font
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=font_size)
    except OSError:
        try:
            # Alternative Mac Font (Helvetica)
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=font_size)
        except OSError:
            try:
                # Linux/Docker
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=font_size)
            except OSError:
                # default
                print("⚠️ Uyarı: Sistem fontu bulunamadı, varsayılan küçük font kullanılıyor.")
                font = ImageFont.load_default()

    for pred in predictions:
        box = pred['bbox'] # [x1, y1, x2, y2]
        label = pred['diagnosis']
        conf = pred['confidence']
        
        color = COLOR_MAP.get(label, "white")
        
        # --- 2. box thicness ---
        line_width = max(3, int(img_w * 0.005))
        draw.rectangle(box, outline=color, width=line_width)
        
        text = f"{label} %{int(conf*100)}"
        
        try:
            # new pillow versions
            left, top, right, bottom = draw.textbbox((box[0], box[1]), text, font=font)
        except AttributeError:
            # old pillow versions
            text_w, text_h = draw.textsize(text, font=font)
            left, top, right, bottom = box[0], box[1], box[0] + text_w, box[1] + text_h

        # Yazının arka planına (etiket) kutu çiz (Daha okunaklı olsun diye biraz padding ekle)
        padding = 5
        draw.rectangle((left, top - padding, right + padding, bottom + padding), fill=color)
        
        # Yazıyı Yaz (Beyaz Renk)
        draw.text((box[0] + padding//2, top - padding//2), text, fill="white", font=font)
        
    return image

# --- interface ---
st.title("🦷 Dental AI Diagnosis System")
st.write("Yapay Zeka Destekli Diş Radyolojisi Analizi ")

# Sol taraf: Yükleme, Sağ taraf: Sonuç
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Bir Panoramik Röntgen Yükleyin", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Resmi göster (Ham hali)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Yüklenen Orijinal Görüntü", use_column_width=True)
        
        if st.button("🔍 Analiz Et (API'ye Gönder)"):
            with st.spinner("Yapay Zeka Röntgeni İnceliyor..."):
                try:
                    # 1. Resmi Byte'a çevir
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # 2. API'ye İstek At (POST)
                    files = {"file": ("xray.png", img_byte_arr, "image/png")}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Analiz Tamamlandı! {result['process_time']} saniye sürdü.")
                        
                        # 3. Sonuçları Çizdir
                        final_img = draw_boxes(image.copy(), result['results'])
                        
                        # Resmi Session State'e kaydet (Yenilenince gitmesin)
                        st.session_state['final_img'] = final_img
                        st.session_state['json_data'] = result
                        
                    else:
                        st.error(f"Hata: {response.text}")
                        
                except Exception as e:
                    st.error(f"Bağlantı Hatası: API çalışıyor mu? ({e})")

with col2:
    if 'final_img' in st.session_state:
        st.image(st.session_state['final_img'], caption="Yapay Zeka Tespiti", use_column_width=True)
        
        # Detaylı Tablo
        st.subheader("📋 Tespit Detayları")
        results = st.session_state['json_data']['results']
        
        if results:
            # JSON verisini tabloya çevir
            table_data = []
            for r in results:
                table_data.append({
                    "Bölge (Quadrant)": r['quadrant'],
                    "Diş Tipi": r['tooth_type'],
                    "Teşhis": r['diagnosis'],
                    "Güven Skoru": f"%{r['confidence']*100:.1f}"
                })
            st.table(table_data)
        else:
            st.info("Bu röntgende herhangi bir hastalık tespit edilemedi.")