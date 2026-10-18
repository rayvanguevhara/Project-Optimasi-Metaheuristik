# RouteOptima — SA vs Tabu Search TSP Dashboard

A web dashboard for comparing Simulated Annealing and Tabu Search on the Traveling Salesman Problem, built with TanStack Start (React + Vite), Tailwind CSS v4, and shadcn/ui.

## Run locally

You need Node.js 20+ (npm works; bun is also supported).

```sh
cd routeoptima
npm install
npm run dev
```

Then open http://localhost:3000 (the terminal prints the exact URL).

Other commands:

```sh
npm run build      # production build
npm run preview    # preview the production build
npm run test       # run unit tests
```

## Project structure

```
├── package.json            # dependencies and scripts
├── vite.config.ts          # Vite/TanStack Start configuration
├── tsconfig.json           # TypeScript configuration
├── vitest.config.ts        # test configuration
├── components.json         # shadcn/ui configuration
├── public/                 # static assets (favicon, robots.txt)
└── src/
    ├── routes/
    │   ├── __root.tsx      # root layout, head metadata, error pages
    │   └── index.tsx       # the dashboard page (form, result cards, chart)
    ├── components/ui/      # shadcn/ui components
    ├── lib/
    │   ├── optimize.ts     # chart data helpers / result types
    │   └── utils.ts        # cn() class merge helper
    ├── hooks/
    ├── test/               # unit tests (vitest)
    ├── router.tsx          # router + query client
    ├── start.ts            # client/server start wiring
    ├── server.ts           # SSR entry
    └── styles.css          # Tailwind v4 theme and design tokens
```

## Note on the backend

The dashboard currently shows sample results; it is wired to call a FastAPI optimization backend at `http://localhost:8000/api/optimize` once available (see `src/routes/index.tsx`). Styling uses Tailwind CSS v4 via `src/styles.css` (there is no legacy `tailwind.config.js`; theme values live in the `@theme` block).
