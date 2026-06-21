// App 1: subir foto y obtener top-N productos similares
import { useState } from "react";

type Product = {
  external_id: string;
  filename?: string;
  score: number;
};

export function EcommerceSearch() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<Product[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function search() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch("/api/ecommerce/search?top_n=10", { method: "POST", body: form });
      if (!r.ok) throw new Error((await r.json()).detail || `error ${r.status}`);
      setResults((await r.json()).results);
    } catch (e) {
      setError(String(e));
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <h2>Buscar producto por imagen</h2>
      <p>Sube una foto y obtén los productos más similares.</p>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button onClick={search} disabled={loading || !file}>
          {loading ? "Buscando..." : "Buscar"}
        </button>
      </div>

      {error && <p style={{ color: "crimson" }}>{error}</p>}

      <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 16 }}>
        {results.map((p, i) => (
          <div key={i} style={{ border: "1px solid #ddd", padding: 8, borderRadius: 8, textAlign: "center" }}>
            {p.filename ? (
              <img 
                src={`/api/images/${p.filename}`} 
                alt={p.filename} 
                style={{ width: "100%", height: "150px", objectFit: "cover", borderRadius: 4 }} 
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                }}
              />
            ) : (
              <div style={{ width: "100%", height: "150px", backgroundColor: "#f0f0f0", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <span>No Image</span>
              </div>
            )}
            <div style={{ marginTop: 8, fontSize: "14px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={p.filename || p.external_id}>
              {p.filename || p.external_id}
            </div>
            <div style={{ color: "#888", fontSize: "12px" }}>
              Similitud: {p.score.toFixed(4)}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
