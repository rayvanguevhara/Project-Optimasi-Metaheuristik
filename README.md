# RouteOptima — SA vs Tabu Search TSP Dashboard

Aplikasi web *dashboard* analitik interaktif untuk membandingkan kinerja algoritma metaheuristik **Simulated Annealing (SA)** dan **Tabu Search (TS)** dalam menyelesaikan permasalahan *Traveling Salesman Problem* (TSP). Proyek ini dikembangkan untuk memenuhi penugasan Mata Kuliah Metode Optimasi Metaheuristik.

## 🌐 Akses Aplikasi (Live Demo)
Aplikasi telah di-*deploy* ke *cloud server* dan dapat diakses secara langsung tanpa instalasi melalui tautan berikut:
👉 **[https://web-optimasi-metaheuristik.vercel.app](https://web-optimasi-metaheuristik.vercel.app)**

---

## ✨ Fitur Utama
* **Sistem Entry Data Excel:** Mendukung pengunggahan matriks jarak antarkota dalam format `.xlsx` atau `.xls`.
* **Kombinasi Greedy Construction Heuristics (Inisialisasi Rute Awal):**
  * *Nearest Neighbor (NN)*
  * *Farthest Insertion (FI)*
  * *Cheapest Insertion (CI)*
* **Kombinasi Local Search (Neighborhood Move):**
  * *Node Swap (SWAP)*
  * *2-Opt (2OPT)*
* **Pengaturan Parameter Metaheuristik Kustom:**
  * **Simulated Annealing (SA):** *Initial Temperature*, *Cooling Rate / Alpha*, dan *Max Iterations*.
  * **Tabu Search (TS):** *Tabu Tenure* dan *Max Iterations*.
* **Visualisasi Analitik Interaktif:**
  * Kartu ringkasan jarak optimal terbaik (*Best Distance*), urutan rute terbaik, dan waktu eksekusi komputasi (*Execution Time*).
  * *Route Map Visualization* (peta rute melingkar interaktif dengan animasi jalur SA dan TS).
  * *Convergence Performance Chart* (grafik perbandingan penurunan jarak terbaik per iterasi).
* **Menu Bantuan (In-App Help):** Panduan penggunaan langsung di dalam aplikasi melalui tombol ikon **`?`** di pojok kanan atas bilah navigasi.

---

## 📖 Panduan Penggunaan Aplikasi (Help Guide)

1. **Konfigurasi Data & Heuristik Awal (Tab Data):**
   * Klik area unggah **Dataset (.xlsx / .xls)** pada panel *Optimization Parameters* di sebelah kiri, lalu pilih file Excel yang berisi matriks jarak simetris antarkota.
   * Pilih **Initialization Method** (*Nearest Neighbor*, *Farthest Insertion*, atau *Cheapest Insertion*) sebagai metode *Greedy Construction Heuristic* pembentuk solusi rute awal.
   * Pilih **Local Search Method** (*2-Opt* atau *Node Swap*) sebagai mekanisme pembentukan solusi tetangga (*neighborhood move*) pada saat iterasi berlangsung.
   * Tentukan **Start Node Index** (indeks kota keberangkatan awal, dimulai dari `0`).
2. **Pengaturan Parameter Simulated Annealing (Tab SA):**
   * Atur nilai **Initial Temp** (suhu awal eksplorasi), **Cooling Rate / Alpha** (laju pendinginan suhu, nilai antara `0` hingga `0.999`), serta **Max Iterations** (batas maksimum iterasi).
3. **Pengaturan Parameter Tabu Search (Tab Tabu):**
   * Atur nilai **Tabu Tenure** (jumlah iterasi suatu pertukaran disimpan di dalam *Tabu List* agar tidak terjadi siklus berulang) dan **Max Iterations**.
4. **Eksekusi dan Pembacaan Hasil Optimasi:**
   * Klik tombol **Run Optimization** di pojok kanan atas.
   * Panel sebelah kanan akan menampilkan perbandingan **Total Jarak Minimum**, **Waktu Eksekusi (detik)**, dan **Urutan Rute Optimal** antara *Simulated Annealing* dan *Tabu Search*.
   * Gunakan tombol *toggle* **SA Route / TS Route** pada kartu **Route Map Visualization** untuk melihat visualisasi graf rute perjalanan terbaik masing-masing algoritma.
   * Amati grafik **Convergence Performance** di bagian bawah untuk menganalisis kecepatan konvergensi jarak terbaik dari iterasi awal hingga akhir.

---

## 💻 Panduan Instalasi & Cara Menjalankan Secara Lokal (Local Development)

Jika Anda ingin mengunduh dan menjalankan proyek ini di komputer lokal (*localhost*), ikuti langkah-langkah di bawah ini:

### Prasyarat Sistem
* **Python** (versi 3.8 atau yang lebih baru) terinstal di sistem.
* **Node.js** (versi 18 atau yang lebih baru) beserta manajer paket **npm** terinstal di sistem.

### Langkah 1: Mendapatkan Salinan Proyek
Unduh proyek dengan melakukan *clone* dari repositori GitHub menggunakan terminal / Command Prompt:
```bash
git clone [https://github.com/rayvanguevhara/Project-Optimasi-Metaheuristik.git](https://github.com/rayvanguevhara/Project-Optimasi-Metaheuristik.git)
cd Project-Optimasi-Metaheuristik
```
*(Atau unduh file arsip dalam bentuk `.zip` dari GitHub, lalu ekstrak ke folder pilihan Anda dan buka direktorinya melalui terminal).*

### Langkah 2: Menjalankan Sisi Backend (Python FastAPI)
1. Masuk ke direktori utama proyek (jika belum berada di sana).
2. Instal pustaka (*libraries*) Python yang dibutuhkan dengan menjalankan perintah:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan server *backend* menggunakan Uvicorn agar API dapat merespons permintaan dari *frontend*:
   ```bash
   uvicorn api.index:app --reload --port 8000
   ```
   *Backend* kini aktif dan berjalan di `http://localhost:8000`.

### Langkah 3: Menjalankan Sisi Frontend (React + Vite / TanStack Router)
1. Buka jendela **terminal baru** (biarkan terminal *backend* tetap berjalan di latar belakang).
2. Pastikan Anda berada di direktori utama proyek yang sama.
3. Instal dependensi *frontend* menggunakan npm:
   ```bash
   npm install
   ```
4. Jalankan server *development frontend*:
   ```bash
   npm run dev
   ```
5. Terminal akan menampilkan tautan lokal (biasanya `http://localhost:5173` atau `http://localhost:3000`). Buka tautan tersebut melalui peramban (*browser*) untuk mengakses aplikasi secara penuh.
