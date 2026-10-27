import { createFileRoute } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Route as RouteIcon,
  Loader2,
  Rocket,
  BarChart3,
  Upload,
  Clock,
  Thermometer,
  ListChecks,
  Database,
  HelpCircle,
  Shuffle,
  FileSpreadsheet,
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { buildChartData, type OptimizeResult } from "@/lib/optimize";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "RouteOptima — SA vs Tabu Search TSP Dashboard" },
      { name: "description", content: "Compare Simulated Annealing and Tabu Search performance on the Traveling Salesman Problem." },
    ],
  }),
  component: Index,
});

// Generator Matriks Jarak Simetris Berbasis Koordinat 2D
function createRandomDistanceMatrix(n: number): number[][] {
  const coords = Array.from({ length: n }, () => ({
    x: Math.random() * 90 + 5,
    y: Math.random() * 90 + 5,
  }));

  return coords.map((c1, i) =>
    coords.map((c2, j) =>
      i === j ? 0 : Math.max(5, Math.round(Math.hypot(c1.x - c2.x, c1.y - c2.y)))
    )
  );
}

// Komponen Pembungkus Form Field
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs font-medium text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

// 1. KOMPONEN: Visualisasi Rute Melingkar (Mendukung hingga 50+ Kota)
function RouteMap({ numCities, tour, color }: { numCities: number; tour: number[]; color: string }) {
  const size = 380;
  const center = size / 2;
  const radius = 152;

  // Penyesuaian ukuran titik otomatis berdasarkan jumlah kota agar tidak bertumpuk
  const nodeRadius = numCities <= 12 ? 13 : numCities <= 20 ? 10.5 : numCities <= 35 ? 8.5 : 6.5;
  const fontSize = numCities <= 12 ? 11 : numCities <= 20 ? 9.5 : numCities <= 35 ? 8 : 6.5;

  const nodes = Array.from({ length: numCities }).map((_, i) => {
    const angle = (i * 2 * Math.PI) / numCities - Math.PI / 2;
    return {
      id: i,
      x: center + radius * Math.cos(angle),
      y: center + radius * Math.sin(angle),
    };
  });

  const pathData = tour
    .map((nodeIndex, i) => {
      const node = nodes[nodeIndex];
      if (!node) return "";
      return `${i === 0 ? "M" : "L"} ${node.x} ${node.y}`;
    })
    .join(" ");

  return (
    <div className="flex justify-center items-center py-4 overflow-hidden">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="max-w-full h-auto drop-shadow-md overflow-visible">
        {nodes.map((n1, i) =>
          nodes.slice(i + 1).map((n2, j) => (
            <line
              key={`bg-${i}-${j}`}
              x1={n1.x}
              y1={n1.y}
              x2={n2.x}
              y2={n2.y}
              stroke="#e2e8f0"
              strokeWidth="0.5"
              strokeOpacity={numCities > 15 ? 0.35 : 0.75}
            />
          ))
        )}

        <motion.path
          key={tour.join("-")}
          d={pathData}
          fill="none"
          stroke={color}
          strokeWidth={numCities > 25 ? "2.2" : "3"}
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.5, ease: "easeInOut" }}
        />

        {nodes.map((node) => (
          <g key={`node-${node.id}`}>
            <circle cx={node.x} cy={node.y} r={nodeRadius} fill="white" stroke={color} strokeWidth={numCities > 25 ? "2" : "2.5"} />
            <text x={node.x} y={node.y} textAnchor="middle" dy=".35em" fontSize={fontSize} fontWeight="bold" fill="#34495E">
              {node.id}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function Index() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [dataMode, setDataMode] = useState<"random" | "upload">("random");
  const [numCitiesInput, setNumCitiesInput] = useState<number>(30);
  const [randomMatrix, setRandomMatrix] = useState<number[][]>(() => createRandomDistanceMatrix(30));
  const [file, setFile] = useState<File | null>(null);
  const [mapView, setMapView] = useState<"sa" | "ts">("sa");

  const [p, setP] = useState({
    init: "NN",
    local: "2OPT",
    start: 0,
    temp: 10000,
    alpha: 0.99,
    saIter: 15000,
    tenure: 5,
    tsIter: 100,
  });

  // Perbarui matriks acak secara otomatis saat pengguna mengubah jumlah kota
  useEffect(() => {
    const validN = Math.max(3, Math.min(60, numCitiesInput || 30));
    setRandomMatrix(createRandomDistanceMatrix(validN));
    if (p.start >= validN) {
      setP((prev) => ({ ...prev, start: 0 }));
    }
  }, [numCitiesInput]);

  const handleRegenerateMatrix = () => {
    const validN = Math.max(3, Math.min(60, numCitiesInput || 30));
    setRandomMatrix(createRandomDistanceMatrix(validN));
  };

  const set = (k: keyof typeof p) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setP({ ...p, [k]: Number(e.target.value) });

  // 2. FUNGSI FETCH API (Support Vercel & Local)
  const run = async () => {
    if (dataMode === "upload" && !file) {
      alert("Mohon unggah file dataset (.xlsx / .xls) terlebih dahulu atau pilih mode Generate Random!");
      return;
    }

    setLoading(true);
    const fd = new FormData();
    fd.append("data_mode", dataMode);

    if (dataMode === "upload" && file) {
      fd.append("file", file);
    } else {
      const validN = Math.max(3, Math.min(60, numCitiesInput || 30));
      const activeMatrix = randomMatrix.length === validN ? randomMatrix : createRandomDistanceMatrix(validN);
      fd.append("num_cities", String(validN));
      fd.append("random_matrix", JSON.stringify(activeMatrix));
    }

    fd.append("init_method", p.init);
    fd.append("ls_method", p.local);
    fd.append("start_node", String(p.start));
    fd.append("sa_initial_temp", String(p.temp));
    fd.append("sa_alpha", String(p.alpha));
    fd.append("sa_max_iter", String(p.saIter));
    fd.append("ts_tabu_tenure", String(p.tenure));
    fd.append("ts_max_iter", String(p.tsIter));
    fd.append("ils_max_iter", "50");

    try {
      const apiUrl = import.meta.env.DEV
        ? "http://localhost:8000/api/optimize"
        : "/api/optimize";

      const res = await fetch(apiUrl, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) throw new Error("Terjadi kesalahan pada server.");

      const json = await res.json();
      if (json.status === "success") {
        setResult(json.data);
      }
    } catch (err) {
      console.error(err);
      alert("Gagal melakukan optimasi. Pastikan backend Python berjalan.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F7F8FA]">
      <header className="sticky top-0 z-10 border-b bg-white/90 backdrop-blur shadow-sm">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#34495E] text-white">
              <RouteIcon className="h-5 w-5" />
            </div>
            <span className="text-lg font-bold tracking-tight text-[#34495E]">RouteOptima</span>
          </div>

          <div className="flex items-center gap-3">
            {/* Modal Panduan (Help) */}
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="icon" className="text-[#34495E] border-gray-200 hover:bg-gray-100">
                  <HelpCircle className="h-5 w-5" />
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[600px] max-h-[80vh] overflow-y-auto bg-white border-none shadow-xl rounded-xl">
                <DialogHeader>
                  <DialogTitle className="text-[#34495E] text-xl font-bold">Panduan Penggunaan Aplikasi</DialogTitle>
                  <DialogDescription>
                    Informasi parameter algoritma dan cara menjalankan RouteOptima.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 text-sm text-[#34495E] mt-2 leading-relaxed">
                  <div>
                    <h4 className="font-bold text-[#8DA4BA] mb-1">1. Sistem Entry Data (Random Generator & Excel)</h4>
                    <p>
                      Anda dapat memilih mode <strong>Generate Random</strong> untuk membangkitkan matriks jarak simetris secara otomatis (mendukung hingga 30+ kota) dan langsung menekan <strong>Run Optimization</strong>, atau memilih mode <strong>Upload Excel</strong> untuk mengunggah file dataset (<code>.xlsx</code> / <code>.xls</code>).
                    </p>
                  </div>
                  <div>
                    <h4 className="font-bold text-[#8DA4BA] mb-1">2. Kombinasi Inisialisasi & Local Search</h4>
                    <ul className="list-disc pl-5 space-y-1">
                      <li><strong>Nearest / Farthest / Cheapest Insertion:</strong> Metode <em>Greedy Construction Heuristic</em> untuk membangun rute awal sebelum dioptimasi.</li>
                      <li><strong>Node Swap / 2-Opt:</strong> Metode pertukaran <em>Neighborhood</em> (Local Search) untuk memperbaiki kualitas rute pada setiap iterasi.</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-bold text-[#8DA4BA] mb-1">3. Parameter Simulated Annealing (SA)</h4>
                    <ul className="list-disc pl-5 space-y-1">
                      <li><strong>Initial Temp:</strong> Suhu awal. Semakin tinggi, algoritma lebih sering menerima rute buruk di awal untuk keluar dari <em>local optimum</em>.</li>
                      <li><strong>Cooling Rate (Alpha):</strong> Laju penurunan suhu. Direkomendasikan mendekati 1 (contoh: 0.99).</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-bold text-[#8DA4BA] mb-1">4. Parameter Tabu Search (TS)</h4>
                    <ul className="list-disc pl-5 space-y-1">
                      <li><strong>Tabu Tenure:</strong> Masa berlaku memori (Tabu List). Menentukan berapa iterasi sebuah langkah perbaikan dilarang digunakan kembali.</li>
                    </ul>
                  </div>
                </div>
              </DialogContent>
            </Dialog>

            <Button onClick={run} disabled={loading} className="gap-2 bg-[#8DA4BA] hover:bg-[#6c859c] text-white shadow-none">
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Rocket className="h-4 w-4" />}
              {loading ? "Optimizing…" : "Run Optimization"}
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-[35fr_65fr]">
        {/* KOLOM KIRI - KONFIGURASI */}
        <Card className="h-fit shadow-sm border-0 ring-1 ring-black/5 bg-white">
          <CardHeader>
            <CardTitle className="text-base text-[#34495E]">Optimization Parameters</CardTitle>
            <CardDescription>Configure dataset and algorithm settings.</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="data">
              <TabsList className="grid w-full grid-cols-3 bg-[#DCE4EC]/50 text-[#34495E]">
                <TabsTrigger value="data"><Database className="mr-1 h-3.5 w-3.5" />Data</TabsTrigger>
                <TabsTrigger value="sa"><Thermometer className="mr-1 h-3.5 w-3.5" />SA</TabsTrigger>
                <TabsTrigger value="ts"><ListChecks className="mr-1 h-3.5 w-3.5" />Tabu</TabsTrigger>
              </TabsList>

              <TabsContent value="data" className="mt-5 space-y-4">
                {/* Pilihan Sumber Data */}
                <Field label="Data Source Mode">
                  <div className="grid grid-cols-2 gap-2 rounded-lg bg-[#F7F8FA] p-1 border border-[#DCE4EC]">
                    <button
                      type="button"
                      onClick={() => setDataMode("random")}
                      className={`flex items-center justify-center gap-1.5 rounded-md py-2 text-xs font-semibold transition-all ${
                        dataMode === "random"
                          ? "bg-[#34495E] text-white shadow-sm"
                          : "text-[#34495E]/70 hover:text-[#34495E]"
                      }`}
                    >
                      <Shuffle className="h-3.5 w-3.5" />
                      Generate Random
                    </button>
                    <button
                      type="button"
                      onClick={() => setDataMode("upload")}
                      className={`flex items-center justify-center gap-1.5 rounded-md py-2 text-xs font-semibold transition-all ${
                        dataMode === "upload"
                          ? "bg-[#34495E] text-white shadow-sm"
                          : "text-[#34495E]/70 hover:text-[#34495E]"
                      }`}
                    >
                      <FileSpreadsheet className="h-3.5 w-3.5" />
                      Upload Excel
                    </button>
                  </div>
                </Field>

                {dataMode === "random" ? (
                  <div className="space-y-3 rounded-lg border border-[#DCE4EC] bg-[#F7F8FA] p-3.5">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs font-semibold text-[#34495E]">Jumlah Kota (N Cities)</Label>
                      <span className="rounded bg-[#8DA4BA]/20 px-2 py-0.5 text-[11px] font-bold text-[#34495E]">
                        Matriks {randomMatrix.length}x{randomMatrix.length}
                      </span>
                    </div>

                    {/* Preset Cepat Jumlah Kota */}
                    <div className="grid grid-cols-5 gap-1.5">
                      {[8, 15, 20, 30, 40].map((preset) => (
                        <button
                          key={preset}
                          type="button"
                          onClick={() => setNumCitiesInput(preset)}
                          className={`rounded border py-1 text-xs font-medium transition-colors ${
                            numCitiesInput === preset
                              ? "border-[#8DA4BA] bg-[#8DA4BA] text-white"
                              : "border-gray-200 bg-white text-[#34495E] hover:bg-gray-50"
                          }`}
                        >
                          {preset}
                        </button>
                      ))}
                    </div>

                    <div className="flex gap-2">
                      <Input
                        type="number"
                        min={3}
                        max={60}
                        value={numCitiesInput}
                        onChange={(e) => setNumCitiesInput(Number(e.target.value))}
                        className="bg-white h-9 text-xs"
                        placeholder="Jumlah kota (3-60)"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={handleRegenerateMatrix}
                        className="h-9 gap-1.5 border-[#8DA4BA] text-xs text-[#34495E] hover:bg-[#DCE4EC]/40"
                      >
                        <Shuffle className="h-3.5 w-3.5 text-[#8DA4BA]" />
                        Acak Ulang
                      </Button>
                    </div>

                    {/* Preview Matriks Jarak Acak */}
                    <div className="space-y-1">
                      <span className="text-[11px] text-muted-foreground">
                        Preview Matriks Jarak (5 kota pertama):
                      </span>
                      <div className="overflow-x-auto rounded border border-[#DCE4EC] bg-white p-2 font-mono text-[10px] text-[#34495E]">
                        {randomMatrix.slice(0, 5).map((row, idx) => (
                          <div key={idx} className="whitespace-nowrap">
                            <span className="font-bold text-[#8DA4BA]">K{idx}:</span> [
                            {row.slice(0, 6).join(", ")}
                            {row.length > 6 ? ", ..." : ""}]
                          </div>
                        ))}
                        {randomMatrix.length > 5 && (
                          <div className="text-muted-foreground mt-0.5">
                            ... ({randomMatrix.length}x{randomMatrix.length} siap dioptimasi)
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <Field label="Dataset (.xlsx / .xls)">
                    <label className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed border-[#8DA4BA] bg-[#F7F8FA] p-5 text-center text-sm text-[#34495E] transition-colors hover:bg-[#DCE4EC]/30">
                      <Upload className="h-5 w-5 text-[#8DA4BA]" />
                      <span className="truncate max-w-full font-medium">
                        {file ? file.name : "Click to upload spreadsheet"}
                      </span>
                      <input
                        type="file"
                        accept=".xlsx,.xls"
                        className="hidden"
                        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                      />
                    </label>
                  </Field>
                )}

                <Field label="Initialization Method">
                  <Select value={p.init} onValueChange={(v) => setP({ ...p, init: v as never })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="NN">Nearest Neighbor (NN)</SelectItem>
                      <SelectItem value="FI">Farthest Insertion (FI)</SelectItem>
                      <SelectItem value="CI">Cheapest Insertion (CI)</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
                <Field label="Local Search Method">
                  <Select value={p.local} onValueChange={(v) => setP({ ...p, local: v as never })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="SWAP">Node Swap (SWAP)</SelectItem>
                      <SelectItem value="2OPT">2-Opt (2OPT)</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
                <Field label="Start Node Index">
                  <Input
                    type="number"
                    min={0}
                    max={dataMode === "random" ? Math.max(0, numCitiesInput - 1) : undefined}
                    value={p.start}
                    onChange={set("start")}
                  />
                </Field>
              </TabsContent>

              <TabsContent value="sa" className="mt-5 space-y-4">
                <Field label="Initial Temp"><Input type="number" value={p.temp} onChange={set("temp")} /></Field>
                <Field label="Cooling Rate / Alpha"><Input type="number" step={0.001} max={0.999} value={p.alpha} onChange={set("alpha")} /></Field>
                <Field label="Max Iterations"><Input type="number" value={p.saIter} onChange={set("saIter")} /></Field>
              </TabsContent>

              <TabsContent value="ts" className="mt-5 space-y-4">
                <Field label="Tabu Tenure"><Input type="number" value={p.tenure} onChange={set("tenure")} /></Field>
                <Field label="Max Iterations"><Input type="number" value={p.tsIter} onChange={set("tsIter")} /></Field>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* KOLOM KANAN - HASIL ANALISIS */}
        <section className="min-w-0">
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex h-full min-h-[480px] flex-col items-center justify-center rounded-xl border border-dashed border-gray-300 bg-white p-10 text-center"
              >
                <div className="mb-5 flex h-20 w-20 items-center justify-center rounded-full bg-[#DCE4EC]/50">
                  {loading ? (
                    <Loader2 className="h-10 w-10 animate-spin text-[#8DA4BA]" />
                  ) : (
                    <BarChart3 className="h-10 w-10 text-[#8DA4BA]" />
                  )}
                </div>
                <p className="max-w-sm text-sm font-medium text-[#34495E]">
                  {loading
                    ? "Running intensive optimization…"
                    : dataMode === "random"
                    ? `Random dataset (${randomMatrix.length} cities) is ready! Click "Run Optimization" at the top right to start.`
                    : "Upload dataset and configure parameters to see optimization results"}
                </p>
              </motion.div>
            ) : (
              <motion.div key="res" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                {/* KARTU SUMMARY */}
                <div className="grid gap-6 md:grid-cols-2">
                  <ResultCard title="Simulated Annealing" colorCode="#8DA4BA" r={result.sa} />
                  <ResultCard title="Tabu Search" colorCode="#10b981" r={result.ts} />
                </div>

                {/* KOMPONEN ROUTE MAP VISUALIZATION */}
                <Card className="shadow-sm border-0 ring-1 ring-black/5">
                  <CardHeader className="flex flex-row items-center justify-between pb-2 border-b mb-2">
                    <div>
                      <CardTitle className="text-base text-[#34495E]">Route Map Visualization</CardTitle>
                      <CardDescription>Visualizing best path for {result.num_cities} cities</CardDescription>
                    </div>
                    <div className="flex bg-[#F7F8FA] rounded-md p-1 border">
                      <button
                        onClick={() => setMapView("sa")}
                        className={`px-4 py-1.5 text-xs rounded-sm transition-all ${
                          mapView === "sa"
                            ? "bg-white shadow-sm font-bold text-[#8DA4BA]"
                            : "text-[#34495E]/60 hover:text-[#34495E]"
                        }`}
                      >
                        SA Route
                      </button>
                      <button
                        onClick={() => setMapView("ts")}
                        className={`px-4 py-1.5 text-xs rounded-sm transition-all ${
                          mapView === "ts"
                            ? "bg-white shadow-sm font-bold text-[#10b981]"
                            : "text-[#34495E]/60 hover:text-[#34495E]"
                        }`}
                      >
                        TS Route
                      </button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <RouteMap
                      numCities={result.num_cities}
                      tour={mapView === "sa" ? result.sa.best_tour : result.ts.best_tour}
                      color={mapView === "sa" ? "#8DA4BA" : "#10b981"}
                    />
                  </CardContent>
                </Card>

                {/* GRAFIK KONVERGENSI */}
                <Card className="shadow-sm border-0 ring-1 ring-black/5">
                  <CardHeader>
                    <CardTitle className="text-base text-[#34495E]">Convergence Performance</CardTitle>
                    <CardDescription>Best distance per iteration comparison</CardDescription>
                  </CardHeader>
                  <CardContent className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={buildChartData(result.sa.history, result.ts.history)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="iteration" stroke="#94a3b8" fontSize={12} />
                        <YAxis stroke="#94a3b8" fontSize={12} domain={["auto", "auto"]} />
                        <Tooltip contentStyle={{ borderRadius: "8px", border: "none", boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)" }} />
                        <Legend />
                        <Line type="monotone" name="Simulated Annealing" dataKey="SA" stroke="#8DA4BA" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                        <Line type="monotone" name="Tabu Search" dataKey="TS" stroke="#10b981" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </main>
    </div>
  );
}

function ResultCard({ title, colorCode, r }: { title: string; colorCode: string; r: any }) {
  return (
    <Card className="relative overflow-hidden shadow-sm border-0 ring-1 ring-black/5 bg-white">
      <div className="absolute inset-x-0 top-0 h-1" style={{ backgroundColor: colorCode }} />
      <CardHeader className="pb-2">
        <CardDescription className="font-semibold" style={{ color: colorCode }}>{title}</CardDescription>
        <CardTitle className="text-3xl font-extrabold text-[#34495E]">{r.best_distance.toLocaleString()}</CardTitle>
        <p className="flex items-center gap-1 text-xs font-medium text-[#8DA4BA]">
          <Clock className="h-3 w-3" />Execution Time: {r.time_seconds.toFixed(4)}s
        </p>
      </CardHeader>
      <CardContent>
        <Label className="text-xs text-muted-foreground">Optimal Path Sequence</Label>
        <div className="mt-1.5 max-h-24 overflow-y-auto rounded-md bg-[#F7F8FA] p-3 font-mono text-xs text-[#34495E] leading-relaxed border border-[#DCE4EC]">
          {r.best_tour.join(" → ")}
        </div>
      </CardContent>
    </Card>
  );
}