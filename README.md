# Backend API - Deteksi Citra Sintetis AI (FastAPI & Railway)

Layanan backend REST API berbasis **FastAPI** untuk mengklasifikasikan citra sintetis AI vs nyata menggunakan model **EfficientNetB0 + SE-Attention**, dilengkapi visualisasi **Grad-CAM++**.

---

## 1. Menjalankan Lokal

1. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Salin bobot model hasil training (`cifake_model.keras`) ke dalam folder ini.
   *(Catatan: Jika file model belum ada, backend tetap dapat berjalan dalam mode simulasi)*.
3. Jalankan server FastAPI:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
4. Dokumentasi interaktif Swagger UI dapat diakses di: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 2. Deploy ke Railway

1. **Inisialisasi Git**:
   Pastikan folder backend ini berada di repository GitHub atau inisialisasi git baru:
   ```bash
   git init
   git add .
   git commit -m "Deploy CIFAKE Backend to Railway"
   ```
2. **Deploy di Railway**:
   - Buka [railway.app](https://railway.app) dan login.
   - Klik **New Project** $\rightarrow$ **Deploy from GitHub repo**.
   - Pilih repository backend ini.
   - Railway akan otomatis membaca `Dockerfile` dan men-deploy container.
3. **Generate Public Domain**:
   - Di dashboard Railway, masuk ke menu **Settings** pada service backend.
   - Klik **Generate Domain** (contoh: `cifake-api-production.up.railway.app`).
   - Gunakan domain tersebut di frontend Streamlit.

---

## 3. Dokumentasi Endpoint

### `GET /health`
Cek kesehatan server dan status pemuatan model.

### `POST /predict`
Unggah file citra untuk mendapatkan prediksi kelas dan nilai kepercayaan.
- **Request**: `multipart/form-data` dengan field `file`.
- **Response**:
  ```json
  {
    "label": "FAKE",
    "class_id": 0,
    "confidence": 0.9782,
    "raw_score": 0.0218,
    "is_simulated": false,
    "inference_time_ms": 38.2,
    "filename": "sample.jpg"
  }
  ```

### `POST /explain`
Unggah file citra untuk mendapatkan prediksi sekaligus gambar visualisasi heatmap Grad-CAM++ (Base64).
