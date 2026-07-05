import { useState } from "react";
import { EcommerceSearch } from "./pages/EcommerceSearch";
import { MusicSearch } from "./pages/MusicSearch";
import { QueryConsole } from "./pages/QueryConsole";
import { CompareView } from "./pages/CompareView";

type Tab = "ecommerce" | "music" | "sql" | "compare";

const TABS: { id: Tab; label: string }[] = [
  { id: "ecommerce", label: "Visual E-commerce" },
  { id: "music", label: "Búsqueda Musical" },
  { id: "sql", label: "Consola SQL" },
  { id: "compare", label: "Comparativas" },
];

export function App() {
  const [tab, setTab] = useState<Tab>("ecommerce");
  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 980, margin: "0 auto", padding: 24 }}>
      <h1>Búsqueda Multimodal</h1>
      <nav style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        {TABS.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} disabled={tab === t.id}>
            {t.label}
          </button>
        ))}
      </nav>
      {tab === "ecommerce" && <EcommerceSearch />}
      {tab === "music" && <MusicSearch />}
      {tab === "sql" && <QueryConsole />}
      {tab === "compare" && <CompareView />}
    </div>
  );
}
