# RouteOptima — SA vs Tabu Search TSP Dashboard

Aplikasi web interaktif untuk membandingkan kinerja algoritma metaheuristik **Simulated Annealing (SA)** dan **Tabu Search (TS)** dalam menyelesaikan permasalahan *Traveling Salesman Problem* (TSP). Disusun untuk memenuhi penugasan Mata Kuliah Metode Optimasi Metaheuristik.

## 🌐 Akses Aplikasi (Live Demo)
Aplikasi dapat diakses secara langsung tanpa instalasi melalui tautan berikut:
👉 **[https://web-optimasi-metaheuristik.vercel.app](https://web-optimasi-metaheuristik.vercel.app)**

## ✨ Fitur Utama
* **Sistem Entry Data Excel:** Mendukung pengunggahan matriks jarak antarkota dalam format `.xlsx` atau `.xls`.
* **Kombinasi Greedy Construction Heuristics (Inisialisasi):**
  * *Nearest Neighbor (NN)*
  * *Farthest Insertion (FI)*
  * *Cheapest Insertion (CI)*
* **Kombinasi Local Search (Neighborhood Move):**
  * *Node Swap (SWAP)*
  * *2-Opt (2OPT)*
* **Pengaturan Parameter Metaheuristik:**
  * **Simulated Annealing:** *Initial Temperature*, *Cooling Rate / Alpha*, dan *Max Iterations*.
  * **Tabu Search:** *Tabu Tenure* dan *Max Iterations*.
* **Visualisasi Analitik:** Kartu ringkasan jarak terbaik & waktu komputasi, *Route Map Visualization* (peta rute melingkar interaktif), serta *Convergence Performance Chart* (grafik penurunan jarak terbaik per iterasi).
* **Menu Bantuan (Help):** Panduan penggunaan langsung di dalam aplikasi melalui tombol ikon **`?`** di bilah navigasi atas.

## 💻 Cara Menjalankan Secara Lokal (Local Development)

### 1. Menjalankan Backend (Python FastAPI)
```bash
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
