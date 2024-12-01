# DTS_FastAPI dan LangServe

## Pengantar

FastAPI adalah framework modern dan cepat untuk membangun API dengan Python 3.6+ berdasarkan tipe anotasi. Ini dirancang untuk memberikan performa tinggi dan kemudahan penggunaan. LangServe adalah library yang memungkinkan pengembang untuk mendistribusikan `LangChain` runnables dan chains sebagai REST API, terintegrasi dengan FastAPI.


## 🚀 Fitur Utama
- **FastAPI**:
  - Mendukung pengembangan API yang cepat dan efisien.
  - Validasi data otomatis menggunakan Pydantic.
  - Dokumentasi API otomatis dengan Swagger UI.

- **LangServe**:
  - Mendistribusikan runnables dan chains dari LangChain sebagai REST API.
  - Endpoint efisien untuk pemanggilan model bahasa.
  - Dukungan untuk streaming dan tracing.

---

## 🛠️ Instalasi

### Prasyarat
Pastikan Anda telah menginstal:
- Python 3.8 atau lebih tinggi
- Pip atau Anaconda

1. **Kloning Repositori**

   ```bash
   git clone https://github.com/DTSense/DTS_FASTAPI_LANGSERVE.git
   cd DTS_FASTAPI_LANGSERVE
   ```
2. Buat dan Aktifkan Virtual Environment
    ```bash
   python -m venv venv
   source venv/bin/activate   # Pada Windows gunakan: venv\Scripts\activate    
   ```
3. Install Dependensi
    ```bash
   pip install -r requirements.txt    
   ```

4. Atau install FastAPI/LangServe jika tidak ada di requirements.txt
    ```bash
   pip install uvicorn fastapi langserve[all]  
   ```

5. Jalankan FastAPI/LangServe di Terminal
    ```bash
   uvicorn app.server:app --host 0.0.0.0 --port 8000 
   ```
6. ADDITIONAL, Jalankan dengan STREAMLIT di Terminal
    ```bash
   streamlit run app_st.py 
   ```

