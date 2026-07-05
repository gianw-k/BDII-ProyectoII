// Pestana Comparativas: la misma consulta por el motor propio y por
// PostgreSQL nativo, lado a lado con resultados y latencia de cada uno.
// Consume los endpoints /compare/* del backend (los mismos de la Fase 4):
//   • texto  -> indice invertido propio vs GIN full-text vs pgvector
//   • imagen -> indice invertido propio vs pgvector (HNSW)
//   • audio  -> indice invertido propio vs pgvector (HNSW)

import { useState, useEffect } from "react";

// ── tipos ──────────────────────────────────────────────────────────────────

type MethodResult = {
  method: string;
  latency_ms: number;
  count: number;
  results: Record<string, any>[];
};

type Modality = "text" | "image" | "audio";

type Track = { filename: string; label: string };

// ── constantes ─────────────────────────────────────────────────────────────

const METHOD_LABELS: Record<string, string> = {
  inverted_index: "Índice invertido propio (SPIMI)",
  gin_fulltext: "PostgreSQL GIN (full-text)",
  pgvector_cosine: "PostgreSQL pgvector (HNSW)",
};

const GENRES = [
  "blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock",
];

// ── componente principal ───────────────────────────────────────────────────

export function CompareView() {
  const [modality, setModality] = useState<Modality>("text");
  const [methods, setMethods]   = useState<MethodResult[]>([]);
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);

  async function runCompare(url: string) {
    setLoading(true); setError(""); setMethods([]);
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error((await r.json()).detail || `error ${r.status}`);
      setMethods((await r.json()).methods);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <h2 style={{ margin: 0 }}>Comparativas: motor propio vs PostgreSQL</h2>
      <p style={descStyle}>
        Ejecuta la misma consulta con el índice invertido propio (SPIMI) y con
        las técnicas nativas de PostgreSQL (GIN full-text, pgvector + HNSW), y
        compara resultados y latencia lado a lado.
      </p>

      <div style={{ display: "flex", gap: 8 }}>
        <ModeBtn label="Texto (letras)" active={modality === "text"}
                 onClick={() => { setModality("text"); setMethods([]); setError(""); }} />
        <ModeBtn label="Imagen (productos)" active={modality === "image"}
                 onClick={() => { setModality("image"); setMethods([]); setError(""); }} />
        <ModeBtn label="Audio (pistas)" active={modality === "audio"}
                 onClick={() => { setModality("audio"); setMethods([]); setError(""); }} />
      </div>

      {modality === "text"  && <TextForm  loading={loading} onRun={runCompare} />}
      {modality === "image" && <ImageForm loading={loading} onRun={runCompare} />}
      {modality === "audio" && <AudioForm loading={loading} onRun={runCompare} />}

      {error && <p style={{ color: "crimson", margin: 0 }}>{error}</p>}

      {methods.length > 0 && (
        <>
          <LatencyBars methods={methods} />
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${methods.length}, minmax(0, 1fr))`,
              gap: 12,
              alignItems: "start",
            }}
          >
            {methods.map((m) => (
              <MethodColumn key={m.method} m={m} modality={modality} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}

// ── formularios por modalidad ──────────────────────────────────────────────

function TextForm({ loading, onRun }: { loading: boolean; onRun: (url: string) => void }) {
  const [q, setQ] = useState("");
  const go = () => q.trim() && onRun(`/api/compare/text?q=${encodeURIComponent(q)}&top_n=10`);
  return (
    <div style={{ display: "flex", gap: 8 }}>
      <input
        type="text"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && go()}
        placeholder='fragmento de letra, ej: "love you baby"'
        style={inputStyle}
      />
      <button onClick={go} disabled={loading} style={btnStyle}>
        {loading ? "Comparando..." : "Comparar"}
      </button>
    </div>
  );
}

function ImageForm({ loading, onRun }: { loading: boolean; onRun: (url: string) => void }) {
  const [id, setId] = useState("15970");
  const go = () => id.trim() &&
    onRun(`/api/compare/vector?external_id=${encodeURIComponent(id.trim())}&top_n=10`);
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
      <label style={{ fontWeight: 600 }}>ID de producto:</label>
      <input
        type="text"
        value={id}
        onChange={(e) => setId(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && go()}
        placeholder="ej: 15970"
        style={{ ...inputStyle, flex: "0 0 140px" }}
      />
      {id.trim() && (
        <img
          src={`/api/images/${id.trim()}.jpg`}
          alt="consulta"
          style={{ height: 64, borderRadius: 6, border: "1px solid #ddd" }}
          onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
        />
      )}
      <button onClick={go} disabled={loading} style={btnStyle}>
        {loading ? "Comparando..." : "Comparar"}
      </button>
    </div>
  );
}

function AudioForm({ loading, onRun }: { loading: boolean; onRun: (url: string) => void }) {
  const [genre, setGenre]       = useState("blues");
  const [tracks, setTracks]     = useState<Track[]>([]);
  const [selected, setSelected] = useState("");

  useEffect(() => {
    setTracks([]); setSelected("");
    fetch(`/api/music/audio/tracks?genre=${genre}`)
      .then((r) => r.json())
      .then((d) => {
        const list: Track[] = d.tracks || [];
        setTracks(list);
        setSelected(list.length ? list[0].filename : "");
      })
      .catch(() => setTracks([]));
  }, [genre]);

  const go = () => selected &&
    onRun(`/api/compare/audio?filename=${encodeURIComponent(selected)}&top_n=10`);

  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
      <label style={{ fontWeight: 600 }}>Género:</label>
      <select value={genre} onChange={(e) => setGenre(e.target.value)} style={selectStyle}>
        {GENRES.map((g) => <option key={g} value={g}>{g}</option>)}
      </select>
      <label style={{ fontWeight: 600 }}>Pista:</label>
      <select value={selected} onChange={(e) => setSelected(e.target.value)}
              style={{ ...selectStyle, minWidth: 200 }}>
        {tracks.map((t) => <option key={t.filename} value={t.filename}>{t.filename}</option>)}
        {tracks.length === 0 && <option disabled>Sin pistas</option>}
      </select>
      <button onClick={go} disabled={!selected || loading} style={btnStyle}>
        {loading ? "Comparando..." : "Comparar"}
      </button>
    </div>
  );
}

// ── barras de latencia ─────────────────────────────────────────────────────

function LatencyBars({ methods }: { methods: MethodResult[] }) {
  const max = Math.max(...methods.map((m) => m.latency_ms), 0.001);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {methods.map((m) => (
        <div key={m.method} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ flex: "0 0 260px", fontSize: 13 }}>
            {METHOD_LABELS[m.method] || m.method}
          </span>
          <div style={{ flex: 1, background: "#eee", borderRadius: 4, height: 16 }}>
            <div
              style={{
                width: `${Math.max((m.latency_ms / max) * 100, 2)}%`,
                height: "100%",
                borderRadius: 4,
                background: m.method === "inverted_index" ? "#1976d2" : "#e08a3c",
              }}
            />
          </div>
          <span style={{ flex: "0 0 80px", fontSize: 13, fontWeight: 700, textAlign: "right" }}>
            {m.latency_ms.toFixed(1)} ms
          </span>
        </div>
      ))}
    </div>
  );
}

// ── columna de resultados de un metodo ─────────────────────────────────────

function MethodColumn({ m, modality }: { m: MethodResult; modality: Modality }) {
  return (
    <div style={colStyle}>
      <div style={{ fontWeight: 700, fontSize: 14 }}>
        {METHOD_LABELS[m.method] || m.method}
      </div>
      <div style={{ color: "#888", fontSize: 12, marginBottom: 8 }}>
        {m.count} resultados · {m.latency_ms.toFixed(1)} ms
      </div>
      {m.results.length === 0 && (
        <div style={{ color: "#888", fontSize: 13 }}>Sin resultados.</div>
      )}
      <ol style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 6 }}>
        {m.results.map((r, i) => (
          <li key={i} style={{ fontSize: 13 }}>
            {modality === "text" && (
              <>
                <strong>{r.title || r.external_id}</strong>
                {r.artist && <> — {r.artist}</>}{" "}
                <span style={scoreStyle}>({r.score})</span>
              </>
            )}
            {modality === "image" && (
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <img
                  src={`/api/images/${r.external_id}.jpg`}
                  alt={String(r.external_id)}
                  style={{ height: 44, borderRadius: 4, border: "1px solid #eee" }}
                  onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
                />
                <span>
                  {r.external_id} <span style={scoreStyle}>({r.score})</span>
                </span>
              </span>
            )}
            {modality === "audio" && (
              <>
                {r.filename}{" "}
                <span style={scoreStyle}>
                  ({typeof r.score === "number" ? r.score.toFixed(4) : r.score})
                </span>
              </>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

// ── helpers de UI ──────────────────────────────────────────────────────────

function ModeBtn({ label, active, onClick }: {
  label: string; active: boolean; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "6px 14px",
        border: `2px solid ${active ? "#1976d2" : "#ccc"}`,
        borderRadius: 20,
        cursor: "pointer",
        background: active ? "#e3f0ff" : "white",
        color: active ? "#1976d2" : "#555",
        fontWeight: active ? 600 : 400,
        fontSize: 13,
      }}
    >
      {label}
    </button>
  );
}

// ── estilos ────────────────────────────────────────────────────────────────

const inputStyle: React.CSSProperties = {
  flex: 1,
  padding: "8px 12px",
  fontSize: 14,
  border: "1px solid #ccc",
  borderRadius: 6,
};
const btnStyle: React.CSSProperties = {
  padding: "8px 18px",
  background: "#1976d2",
  color: "white",
  border: "none",
  borderRadius: 6,
  cursor: "pointer",
  fontWeight: 600,
};
const selectStyle: React.CSSProperties = {
  padding: "6px 10px",
  fontSize: 14,
  border: "1px solid #ccc",
  borderRadius: 6,
};
const scoreStyle: React.CSSProperties = { color: "#888", fontSize: 12 };
const descStyle: React.CSSProperties = { color: "#555", fontSize: 13, margin: 0 };
const colStyle: React.CSSProperties = {
  border: "1px solid #e0e0e0",
  borderRadius: 8,
  padding: 12,
  background: "#fafafa",
};
