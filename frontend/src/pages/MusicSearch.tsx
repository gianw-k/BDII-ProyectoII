// App 2: búsqueda musical — dos modos:
//   • Por letra (full-text, índice invertido TF-IDF)
//   • Por similitud acústica (MFCC → Bag-of-Acoustic-Words)
//     - subir un archivo .wav/.mp3
//     - o elegir una pista de ejemplo del dataset

import { useState, useEffect, useRef } from "react";

// ── tipos ──────────────────────────────────────────────────────────────────

type Song = {
  title?: string;
  artist?: string;
  score: number;
  match?: string;
};

type Track = {
  filename: string;
  label: string;
  n_windows?: number;
};

type AcousticResult = {
  track_id: number;
  filename: string;
  label: string;
  score: number;
};

type MusicTab = "lyrics" | "acoustic";

// ── constantes ─────────────────────────────────────────────────────────────

const GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"];

// ── componente principal ───────────────────────────────────────────────────

export function MusicSearch() {
  const [tab, setTab] = useState<MusicTab>("lyrics");

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <h2 style={{ margin: 0 }}>🎵 Búsqueda Musical Inteligente</h2>

      {/* selector de modo */}
      <div style={{ display: "flex", gap: 0, borderBottom: "2px solid #ccc" }}>
        <TabBtn label="Por letra (TF-IDF)"     active={tab === "lyrics"}   onClick={() => setTab("lyrics")} />
        <TabBtn label="Por similitud acústica" active={tab === "acoustic"} onClick={() => setTab("acoustic")} />
      </div>

      {tab === "lyrics"   && <LyricsSearch />}
      {tab === "acoustic" && <AcousticSearch />}
    </section>
  );
}

// ── tab: búsqueda por letra ────────────────────────────────────────────────

function LyricsSearch() {
  const [q, setQ]           = useState("");
  const [results, setResults] = useState<Song[]>([]);
  const [error, setError]   = useState("");
  const [loading, setLoading] = useState(false);

  async function search() {
    if (!q.trim()) return;
    setLoading(true); setError("");
    try {
      const r = await fetch(`/api/music/search/lyrics?q=${encodeURIComponent(q)}&top_n=10`);
      if (!r.ok) throw new Error((await r.json()).detail || `error ${r.status}`);
      setResults((await r.json()).results);
    } catch (e) {
      setError(String(e)); setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <p style={descStyle}>Escribe un fragmento de letra y el motor buscará las canciones más similares usando TF-IDF + índice invertido.</p>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          type="text" value={q}
          onChange={e => setQ(e.target.value)}
          onKeyDown={e => e.key === "Enter" && search()}
          placeholder='ej: "love you baby" o "midnight driving rain"'
          style={inputStyle}
        />
        <button onClick={search} disabled={loading} style={btnStyle}>
          {loading ? "Buscando…" : "Buscar"}
        </button>
      </div>

      {error && <p style={{ color: "crimson", marginTop: 8 }}>{error}</p>}

      <ol style={{ marginTop: 16, paddingLeft: 20 }}>
        {results.map((s, i) => (
          <li key={i} style={{ marginBottom: 12 }}>
            <strong>{s.title}</strong> — {s.artist}{" "}
            <span style={scoreStyle}>({s.score})</span>
            {s.match && <div style={matchStyle}>…{s.match.slice(0, 140)}…</div>}
          </li>
        ))}
      </ol>
    </div>
  );
}

// ── tab: búsqueda acústica ─────────────────────────────────────────────────

function AcousticSearch() {
  const [mode, setMode]           = useState<"upload" | "demo">("demo");
  const [file, setFile]           = useState<File | null>(null);
  const [genre, setGenre]         = useState("blues");
  const [tracks, setTracks]       = useState<Track[]>([]);
  const [tracksMsg, setTracksMsg] = useState("");
  const [selected, setSelected]   = useState("");
  const [results, setResults]     = useState<AcousticResult[]>([]);
  const [error, setError]         = useState("");
  const [loading, setLoading]     = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // cargar 10 pistas aleatorias cuando se elige género
  useEffect(() => {
    if (mode !== "demo") return;
    setTracks([]); setSelected(""); setTracksMsg("Cargando…");
    fetch(`/api/music/audio/tracks?genre=${genre}`)
      .then(r => r.json())
      .then(d => {
        const list: Track[] = d.tracks || [];
        setTracks(list);
        setSelected(list.length ? list[0].filename : "");
        if (!list.length) setTracksMsg(d.message || "No hay pistas disponibles. ¿Ya corriste el ingest de audio?");
        else setTracksMsg("");
      })
      .catch(() => { setTracks([]); setTracksMsg("Error al cargar pistas."); });
  }, [genre, mode]);

  async function searchByFile() {
    if (!file) return;
    setLoading(true); setError(""); setResults([]);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch("/api/music/search/audio?top_n=10", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || `error ${r.status}`);
      setResults((await r.json()).results);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function searchByFilename() {
    if (!selected) return;
    setLoading(true); setError(""); setResults([]);
    try {
      const r = await fetch(`/api/music/search/audio/by_filename?filename=${encodeURIComponent(selected)}&top_n=10`);
      if (!r.ok) throw new Error((await r.json()).detail || `error ${r.status}`);
      setResults((await r.json()).results);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <p style={descStyle}>
        Encuentra canciones acústicamente similares. El motor extrae características MFCC,
        las cuantiza en <em>acoustic words</em> (K-Means) y busca por similitud coseno.
      </p>

      {/* selector de modo */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <ModeBtn label="🎧 Subir archivo (.wav/.mp3)" active={mode === "upload"} onClick={() => setMode("upload")} />
        <ModeBtn label="🗂 Elegir pista del dataset"  active={mode === "demo"}   onClick={() => setMode("demo")} />
      </div>

      {/* modo: subir archivo */}
      {mode === "upload" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <input
            ref={fileRef}
            type="file"
            accept=".wav,.mp3,.ogg,.flac"
            onChange={e => setFile(e.target.files?.[0] ?? null)}
            style={{ fontSize: 14 }}
          />
          {file && <span style={{ color: "#555", fontSize: 13 }}>📁 {file.name}</span>}
          <button onClick={searchByFile} disabled={!file || loading} style={btnStyle}>
            {loading ? "Procesando audio…" : "Buscar similares"}
          </button>
          <p style={{ color: "#888", fontSize: 12, margin: 0 }}>
            ⏱ La extracción MFCC con librosa puede tardar unos segundos en el servidor.
          </p>
        </div>
      )}

      {/* modo: demo por pista conocida */}
      {mode === "demo" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <label style={{ fontWeight: 600 }}>Género:</label>
            <select value={genre} onChange={e => setGenre(e.target.value)} style={selectStyle}>
              {GENRES.map(g => <option key={g} value={g}>{g}</option>)}
            </select>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <label style={{ fontWeight: 600 }}>Pista:</label>
            <select
              value={selected}
              onChange={e => setSelected(e.target.value)}
              style={{ ...selectStyle, flex: 1 }}
            >
              {tracks.map(t => <option key={t.filename} value={t.filename}>{t.filename}</option>)}
              {tracks.length === 0 && <option disabled>{tracksMsg || "Sin pistas"}</option>}
            </select>
          </div>
          <button onClick={searchByFilename} disabled={!selected || loading} style={btnStyle}>
            {loading ? "Buscando…" : "Buscar similares"}
          </button>
        </div>
      )}

      {error && <p style={{ color: "crimson", marginTop: 10 }}>{error}</p>}

      {/* resultados acústicos */}
      {results.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ margin: "0 0 10px" }}>Resultados ({results.length} pistas similares)</h4>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 10 }}>
            {results.map((r, i) => (
              <div key={i} style={cardStyle}>
                <div style={{ fontSize: 28, textAlign: "center" }}>{genreEmoji(r.label)}</div>
                <div style={{ fontWeight: 700, fontSize: 13, marginTop: 4 }}>{r.filename}</div>
                <div style={{ color: "#666", fontSize: 12, textTransform: "capitalize" }}>{r.label}</div>
                <div style={scoreStyle}>similitud: {r.score}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── helpers de UI ──────────────────────────────────────────────────────────

function TabBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} style={{
      padding: "8px 18px", border: "none", cursor: "pointer", fontWeight: active ? 700 : 400,
      borderBottom: active ? "3px solid #1976d2" : "3px solid transparent",
      background: "transparent", color: active ? "#1976d2" : "#555",
    }}>{label}</button>
  );
}

function ModeBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} style={{
      padding: "6px 14px", border: `2px solid ${active ? "#1976d2" : "#ccc"}`,
      borderRadius: 20, cursor: "pointer", background: active ? "#e3f0ff" : "white",
      color: active ? "#1976d2" : "#555", fontWeight: active ? 600 : 400, fontSize: 13,
    }}>{label}</button>
  );
}

function genreEmoji(label: string): string {
  const map: Record<string, string> = {
    blues: "🎸", classical: "🎻", country: "🤠", disco: "🕺",
    hiphop: "🎤", jazz: "🎺", metal: "🤘", pop: "🎵", reggae: "🌴", rock: "⚡",
  };
  return map[label?.toLowerCase()] ?? "🎵";
}

// ── estilos ────────────────────────────────────────────────────────────────

const inputStyle: React.CSSProperties = { flex: 1, padding: "8px 12px", fontSize: 14, border: "1px solid #ccc", borderRadius: 6 };
const btnStyle: React.CSSProperties   = { padding: "8px 18px", background: "#1976d2", color: "white", border: "none", borderRadius: 6, cursor: "pointer", fontWeight: 600 };
const selectStyle: React.CSSProperties = { padding: "6px 10px", fontSize: 14, border: "1px solid #ccc", borderRadius: 6 };
const scoreStyle: React.CSSProperties  = { color: "#888", fontSize: 12 };
const matchStyle: React.CSSProperties  = { color: "#555", fontSize: 12, marginTop: 4, fontStyle: "italic" };
const descStyle: React.CSSProperties   = { color: "#555", fontSize: 13, margin: "0 0 14px" };
const cardStyle: React.CSSProperties   = {
  border: "1px solid #e0e0e0", borderRadius: 8, padding: 12, background: "#fafafa",
  display: "flex", flexDirection: "column", gap: 2,
};
