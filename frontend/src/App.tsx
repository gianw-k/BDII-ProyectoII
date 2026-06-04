import { useState } from "react";
import { EcommerceSearch } from "./pages/EcommerceSearch";
import { MusicSearch } from "./pages/MusicSearch";

type Tab = "ecommerce" | "music";

export function App() {
  const [tab, setTab] = useState<Tab>("ecommerce");
  return (
    <div style={{ fontFamily: "system-ui", maxWidth: 900, margin: "0 auto", padding: 24 }}>
      <h1>Búsqueda Multimodal</h1>
      <nav style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        <button onClick={() => setTab("ecommerce")} disabled={tab === "ecommerce"}>
          Visual E-commerce
        </button>
        <button onClick={() => setTab("music")} disabled={tab === "music"}>
          Búsqueda Musical
        </button>
      </nav>
      {tab === "ecommerce" ? <EcommerceSearch /> : <MusicSearch />}
    </div>
  );
}
