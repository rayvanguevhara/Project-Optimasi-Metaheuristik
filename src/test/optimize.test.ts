import { describe, it, expect } from "vitest";
import { buildChartData } from "@/lib/optimize";

describe("buildChartData", () => {
  it("carries the shorter history's last value forward", () => {
    const d = buildChartData([4000, 3500, 3000, 2500, 2150.4], [4000, 3200, 2800, 2010]);
    expect(d).toHaveLength(5);
    expect(d[4]).toEqual({ iteration: 4, SA: 2150.4, TS: 2010 });
  });
});
