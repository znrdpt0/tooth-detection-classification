<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232088FF.svg?style=for-the-badge&logo=github-actions&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
---

# Tooth Detection Classification: End-to-End MLOps Pipeline

Bu proje, panoramik dental röntgen görüntülerinden diş hastalıklarının teşhis sürecini otomatize etmek ve diş hekimlerinin karar verme süreçlerine destek olmak amacıyla geliştirilmiş **3 aşamalı (hierarchical)** bir derin öğrenme sistemidir.

---

## Proje Vizyonu
Geleneksel diş hastalıkları teşhisi, yoğun iş yükü altında gözden kaçabilecek detaylara gebedir. Bu proje, quadrant tespiti ile başlayan ve tekil diş bazlı hastalık teşhisi ile sonlanan uçtan uca bir yapay zeka asistanı sunarak teşhis hızını ve doğruluğunu artırmayı hedefler.

---

## Veri Seti: DENTEX
Proje, hiyerarşik etiketleme yapısına sahip olan ve Hugging Face üzerinde paylaşılan **[DENTEX Dataset](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)** (Ibrahim Hamamci et al.) kullanılarak geliştirilmiştir. Veri seti, 2D panoramik röntgenler ve bunlara ait hiyerarşik patoloji etiketlerini içerir.

---

## Hiyerarşik Mimari (3-Stage Pipeline)

Sistem, karmaşık panoramik görüntüleri anlamlandırmak için parçala-yönet (divide and conquer) stratejisini izler:



| Aşama | Görev | Amacı | Sınıf Sayısı |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **Quadrant Detection** | Çene yapısını 4 ana bölgeye ayırarak odaklanmayı sağlar. | 4 Sınıf (FDI Quadrants) |
| **Stage 2** | **Tooth Detection** | Bölge içindeki dişlerin tekil konumlarını ve FDI numaralarını belirler. | 4 Sınıf |
| **Stage 3** | **Disease Diagnosis** | Tespit edilen dişler üzerinde patoloji araması yapar. | Patoloji Sınıfları + **Healthy** (5 sınıf)|

<img width="1570" height="783" alt="image" src="https://github.com/user-attachments/assets/ae0def93-128e-4505-94be-ab290b4c7524" />
<img width="950" height="487" alt="image" src="https://github.com/user-attachments/assets/2699220e-688c-4780-a909-34fda9a94f00" />
<img width="1182" height="598" alt="image" src="https://github.com/user-attachments/assets/fe437562-7af3-4ea8-90f6-1071ff6ddcba" />



---

## Teknik İnovasyonlar & Mühendislik Yaklaşımları

### 1. Veri Mühendisliği & Preprocessing
* **JSON to YOLO Conversion:** Karmaşık hiyerarşik JSON etiketleri, model eğitimi için optimize edilmiş YOLO formatına (`class x_center y_center width height`) dönüştürülmüştür.
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Röntgenlerdeki düşük kontrast sorununu çözmek ve kanal detaylarını belirginleştirmek için uygulanmıştır.



* **Dinamik Padding:** Stage 2'den Stage 3'e geçerken, dişin çevresel dokusunu (kök ucu ve çevre dokular) kaybetmemek için bounding box'lara dinamik genişletme uygulanmıştır.
* **Dinamik Thresholding:** Her aşama için farklı güven eşikleri belirlenerek, quadrant tespiti için yüksek hassasiyet, hastalık tespiti için ise dengeli bir duyarlılık (recall) sağlanmıştır.

### 2. Sentetik Veri Üretimi: "Healthy" Sınıfı
Hastalık tespiti modellerinde "yalancı pozitif" (false positive) oranını düşürmek için, veri setindeki etiketlenmemiş (sağlam) dişlerden otomatik olarak **Healthy** sınıfı oluşturulmuştur. Bu, modelin sağlıklı doku ile patolojiyi ayırt etme yeteneğini %X oranında artırmıştır.

---

## Teknoloji Yığını (Tech Stack)

* **Core AI:** PyTorch, Ultralytics (YOLO serisi)
* **Backend:** FastAPI (Async Inference API)
* **Frontend:** Streamlit (Hekim Arayüzü)
* **DevOps & MLOps:** Docker, GitHub Actions, AWS EC2
* **Image Processing:** OpenCV, NumPy

---
## Frontend: Streamlit (Hekim Arayüzü)
<img width="3456" height="2160" alt="image" src="https://github.com/user-attachments/assets/c8d945ef-b89b-41df-8b5d-e15a4764b07b" />


## End-to-End MLOps: Deployment Mimarisi

Bu projenin en güçlü yönü, sadece bir model değil, tam otomatize edilmiş canlı bir sistem olmasıdır:



### Dockerization & Optimizasyon
API ve UI servisleri, izole konteynerler olarak paketlenmiştir. **Docker Layer Caching** stratejisi kullanılarak, bağımlılıkların (PyTorch, vb.) her seferinde indirilmesi engellenmiş ve build süreleri optimize edilmiştir.



### ⚙️ CI/CD Pipeline (GitHub Actions)
`git push` yapıldığı an tetiklenen otomasyon hattı:
1.  **Build:** Docker imajlarını oluşturur ve **Docker Hub**'a yükler.
2.  **Deploy:** **AWS EC2** sunucusuna SSH üzerinden bağlanarak mevcut konteynerleri günceller ve yayına alır.

### ☁️ Cloud Infrastructure (AWS)
* **Volume Mounting:** Model ağırlıkları (`.pt`) imajın içine gömülmek yerine sunucu üzerinde (EBS) tutulur. Bu sayede her deployment'ta devasa modellerin transfer edilmesine gerek kalmaz.
* **Security & Scalability:** Güvenlik grupları ile port bazlı erişim kontrolü sağlanmıştır.

### 🔄 İş Akış Diyagramı

```mermaid
graph TD

Input["Panoramik Röntgen Görüntüsü"] --> Pre["Preprocessing: CLAHE & Resize"]

subgraph AI_Logic
S1["Stage 1: Quadrant Detection"]
S2["Stage 2: Tooth & FDI Numbering"]
Pad["Dynamic Padding & Cropping"]
S3["Stage 3: Disease Diagnosis"]
end

Pre --> S1
S1 --> S2
S2 --> Pad
Pad --> S3

subgraph Application_Layer
UI["Streamlit Frontend"]
API["FastAPI Backend"]
end

UI --- API
API --> Pre

subgraph MLOps_Deployment
Code["GitHub Push"]
GH["GitHub Actions"]
DB["Docker Build & Push"]
DH["Docker Hub"]
AWS["AWS EC2 Deploy"]
Vol["AWS Volume: Models"]
end

Code --> GH
GH --> DB
DB --> DH
DH --> AWS
AWS -.-> Vol

S3 --> Result["Teşhis Edilmiş Röntgen & Rapor"]
Result --> UI
```

---
## Quick Start & Access (Hızlı Erişim)

Projenin canlı demosuna ulaşmak veya konteynerleri kendi yerelinizde çalıştırmak için aşağıdaki bağlantıları kullanabilirsiniz:

### Live Demo
AWS EC2 üzerinde aktif:
```
http://18.193.138.156:8501
```

### 🐳 Docker Hub Images

**API Image**
```bash
docker pull senin_kullanici_adin/dental-api:latest
```

**UI Image**
```bash
docker pull senin_kullanici_adin/dental-ui:latest
```

---

## 🛠️ Local Run (Yerel Çalıştırma)

Projeyi kendi bilgisayarınızda Docker ile ayağa kaldırmak için:

```bash
# Repo'yu klonlayın
git clone https://github.com/znrdpt0/tooth-detection-classification.git

# Proje klasörüne girin
cd tooth-detection-classification

# Docker Compose ile tüm sistemi başlatın
docker-compose up -d
```

Uygulama başlatıldıktan sonra arayüze şu adresten erişebilirsiniz:

```
http://localhost:8501
```
## 📂 Proje Yapısı
```text
├── .github/workflows/ci.yml    # GitHub Actions Pipeline
├── src/
│   ├── main.py                 # FastAPI Backend
│   ├── services/               # Inference Logic
│   └── frontend/               # Streamlit UI
├── weights/                    # Model Weights (Sunucu üzerinde saklanır)
├── .dockerignore               # Build Optimizasyonu
└── requirements.txt            # Python Bağımlılıkları
