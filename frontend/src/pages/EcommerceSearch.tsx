// App 1: subir foto y obtener top-N productos visualmente similares.
// El motor extrae SIFT (+ color HSV), cuantiza en visual words (K-Means) y
// busca por coseno sobre el indice invertido.

import { useEffect, useState } from "react";

type Product = {
  external_id?: string;
  filename?: string;
  score: number;
};

export function EcommerceSearch() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [results, setResults] = useState<Product[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // controles del motor
  const [topN, setTopN] = useState(10);

  // generar/limpiar el thumbnail de la query
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function search() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const qs = new URLSearchParams({
        top_n: String(topN),
      });
      const r = await fetch(`/api/ecommerce/search?${qs}`, {
        method: "POST",
        body: form,
      });
      if (!r.ok) throw new Error((await r.json()).detail || `error ${r.status}`);
      setResults((await r.json()).results);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  const maxScore = results.length ? results[0].score || 1 : 1;

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <h2 style={{ margin: 0 }}>Busqueda Visual E-commerce</h2>
      <p style={descStyle}>
        Sube la foto de un producto y obten los mas parecidos del catalogo. El
        motor usa descriptores locales SIFT + color, cuantizados en un
        Bag-of-Visual-Words sobre indice invertido.
      </p>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
        {/* query + controles */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 240 }}>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          {preview && (
            <img
              src={preview}
              alt="query"
              style={{
                width: 200,
                height: 200,
                objectFit: "cover",
                borderRadius: 8,
                border: "1px solid #ddd",
              }}
            />
          )}

          <label style={ctrlRow}>
            Top-N:
            <input
              type="number"
              min={1}
              max={50}
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              style={numInput}
            />
          </label>
          <button onClick={search} disabled={loading || !file} style={btnStyle}>
            {loading ? "Buscando..." : "Buscar"}
          </button>
        </div>

        {/* resultados */}
        <div style={{ flex: 1, minWidth: 280 }}>
          {error && <p style={{ color: "crimson" }}>{error}</p>}
          {results.length > 0 && (
            <h4 style={{ margin: "0 0 10px" }}>{results.length} productos similares</h4>
          )}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
              gap: 14,
            }}
          >
            {results.map((p, i) => (
              <ResultCard key={i} p={p} rank={i + 1} maxScore={maxScore} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function ResultCard({
  p,
  rank,
  maxScore,
}: {
  p: Product;
  rank: number;
  maxScore: number;
}) {
  const pct = Math.max(0, Math.min(100, (p.score / maxScore) * 100));
  return (
    <div style={cardStyle}>
      <div style={{ position: "relative" }}>
        <span style={rankBadge}>#{rank}</span>
        {p.filename ? (
          <img
            src={`/api/images/${p.filename}`}
            alt={p.filename}
            style={{ width: "100%", height: 140, objectFit: "cover", borderRadius: 4 }}
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          <div style={noImg}>Sin imagen</div>
        )}
      </div>
      <div style={{ marginTop: 6, fontSize: 12 }} title={p.filename || p.external_id}>
        {p.filename || p.external_id}
      </div>
      {/* barra de similitud */}
      <div style={barTrack}>
        <div style={{ ...barFill, width: `${pct}%` }} />
      </div>
      <div style={{ color: "#888", fontSize: 11 }}>
        Similitud: {p.score.toFixed(4)}
      </div>
    </div>
  );
}

// ── estilos ──────────────────────────────────────────────────────────────────

const descStyle: React.CSSProperties = { color: "#555", fontSize: 13, margin: 0 };
const btnStyle: React.CSSProperties = {
  padding: "8px 18px",
  background: "#1976d2",
  color: "white",
  border: "none",
  borderRadius: 6,
  cursor: "pointer",
  fontWeight: 600,
};
const ctrlRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  fontSize: 13,
  color: "#444",
};
const numInput: React.CSSProperties = {
  width: 60,
  padding: "4px 6px",
  border: "1px solid #ccc",
  borderRadius: 4,
  fontSize: 13,
};
const cardStyle: React.CSSProperties = {
  border: "1px solid #e0e0e0",
  padding: 8,
  borderRadius: 8,
  background: "#fafafa",
};
const noImg: React.CSSProperties = {
  width: "100%",
  height: 140,
  background: "#f0f0f0",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "#999",
  fontSize: 12,
  borderRadius: 4,
};
const rankBadge: React.CSSProperties = {
  position: "absolute",
  top: 4,
  left: 4,
  background: "rgba(25,118,210,0.85)",
  color: "white",
  fontSize: 11,
  padding: "1px 6px",
  borderRadius: 10,
};
const barTrack: React.CSSProperties = {
  height: 5,
  background: "#eee",
  borderRadius: 3,
  marginTop: 6,
  overflow: "hidden",
};
const barFill: React.CSSProperties = {
  height: "100%",
  background: "#1976d2",
};
