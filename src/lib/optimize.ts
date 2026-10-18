export interface AlgoResult {
  best_distance: number;
  best_tour: number[];
  history: number[];
  time_seconds: number;
}
export interface OptimizeResult {
  num_cities: number;
  sa: AlgoResult;
  ts: AlgoResult;
}

/** Merge histories of different lengths; shorter one carries its last value forward. */
export function buildChartData(sa: number[], ts: number[]) {
  const n = Math.max(sa.length, ts.length);
  return Array.from({ length: n }, (_, i) => ({
    iteration: i,
    SA: sa.length ? sa[Math.min(i, sa.length - 1)] : null,
    TS: ts.length ? ts[Math.min(i, ts.length - 1)] : null,
  }));
}
