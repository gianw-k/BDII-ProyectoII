// App 2: busqueda musical — dos modos:
//   • Por letra (full-text, indice invertido TF-IDF)
//   • Por similitud acustica (MFCC → Bag-of-Acoustic-Words)
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

const GENRES = [
  "blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock",
];

// URL base para los archivos de audio servidos por el backend
// Los archivos estan en /data/raw/music/genres_original/{genre}/{filename}
// El backend los sirve en /audio-files/{genre}/{filename}
function audioUrl(label: string, filename: string): string {
  return `/api/audio-files/${label}/${filename}`;
}

// ── componente principal ───────────────────────────────────────────────────

export function MusicSearch() {
  const [tab, setTab] = useState<MusicTab>("lyrics");

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <h2 style={{ margin: 0 }}>Busqueda Musical Inteligente</h2>

      {/* selector de modo */}
      <div style={{ display: "flex", gap: 0, borderBottom: "2px solid #ccc" }}>
        <TabBtn
          label="Por letra (TF-IDF)"
          active={tab === "lyrics"}
          onClick={() => setTab("lyrics")}
        />
        <TabBtn
          label="Por similitud acustica (MFCC)"
          active={tab === "acoustic"}
          onClick={() => setTab("acoustic")}
        />
      </div>

      {tab === "lyrics"   && <LyricsSearch />}
      {tab === "acoustic" && <AcousticSearch />}
    </section>
  );
}

// ── tab: busqueda por letra ────────────────────────────────────────────────

function LyricsSearch() {
  const [q, setQ]             = useState("");
  const [results, setResults] = useState<Song[]>([]);
  const [error, setError]     = useState("");
  const [loading, setLoading] = useState(false);

  async function search() {
    if (!q.trim()) return;
    setLoading(true); setError("");
    try {
      const r = await fetch(
        `/api/music/search/lyrics?q=${encodeURIComponent(q)}&top_n=10`
      );
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
      <p style={descStyle}>
        Escribe un fragmento de letra. El motor busca las canciones mas similares
        usando TF-IDF e indice invertido.
      </p>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          type="text"
          value={q}
          onChange={e => setQ(e.target.value)}
          onKeyDown={e => e.key === "Enter" && search()}
          placeholder='ej: "love you baby" o "midnight driving rain"'
          style={inputStyle}
        />
        <button onClick={search} disabled={loading} style={btnStyle}>
          {loading ? "Buscando..." : "Buscar"}
        </button>
      </div>

      {error && <p style={{ color: "crimson", marginTop: 8 }}>{error}</p>}

      <ol style={{ marginTop: 16, paddingLeft: 20 }}>
        {results.map((s, i) => (
          <li key={i} style={{ marginBottom: 12 }}>
            <strong>{s.title}</strong> — {s.artist}{" "}
            <span style={scoreStyle}>({s.score})</span>
            {s.match && (
              <div style={matchStyle}>...{s.match.slice(0, 140)}...</div>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

// ── tab: busqueda acustica ─────────────────────────────────────────────────

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

  // cargar 10 pistas aleatorias cuando se elige genero
  useEffect(() => {
    if (mode !== "demo") return;
    setTracks([]); setSelected(""); setTracksMsg("Cargando...");
    fetch(`/api/music/audio/tracks?genre=${genre}`)
      .then(r => r.json())
      .then(d => {
        const list: Track[] = d.tracks || [];
        setTracks(list);
        setSelected(list.length ? list[0].filename : "");
        if (!list.length)
          setTracksMsg(
            d.message || "No hay pistas. Reconstruye el indice de audio."
          );
        else setTracksMsg("");
      })
      .catch(() => {
        setTracks([]);
        setTracksMsg("Error al cargar pistas.");
      });
  }, [genre, mode]);

  async function searchByFile() {
    if (!file) return;
    setLoading(true); setError(""); setResults([]);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch("/api/music/search/audio?top_n=10", {
        method: "POST",
        body: fd,
      });
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
      const r = await fetch(
        `/api/music/search/audio/by_filename?filename=${encodeURIComponent(selected)}&top_n=10`
      );
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
        Encuentra pistas musicalmente similares. El motor extrae caracteristicas MFCC,
        las cuantiza en acoustic words (K-Means) y busca por similitud coseno
        sobre un indice invertido (Bag-of-Acoustic-Words).
      </p>

      {/* selector de modo */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <ModeBtn
          label="Subir archivo (.wav / .mp3)"
          active={mode === "upload"}
          onClick={() => setMode("upload")}
        />
        <ModeBtn
          label="Elegir pista del dataset"
          active={mode === "demo"}
          onClick={() => setMode("demo")}
        />
      </div>

      {/* modo: subir archivo */}
      {mode === "upload" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <input
            type="file"
            accept=".wav,.mp3,.ogg,.flac"
            onChange={e => setFile(e.target.files?.[0] ?? null)}
            style={{ fontSize: 14 }}
          />
          {file && (
            <span style={{ color: "#555", fontSize: 13 }}>Archivo: {file.name}</span>
          )}
          <button
            onClick={searchByFile}
            disabled={!file || loading}
            style={btnStyle}
          >
            {loading ? "Procesando audio..." : "Buscar similares"}
          </button>
          <p style={{ color: "#888", fontSize: 12, margin: 0 }}>
            La extraccion MFCC puede tardar unos segundos en el servidor.
          </p>
        </div>
      )}

      {/* modo: demo por pista conocida */}
      {mode === "demo" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <label style={{ fontWeight: 600 }}>Genero:</label>
            <select
              value={genre}
              onChange={e => setGenre(e.target.value)}
              style={selectStyle}
            >
              {GENRES.map(g => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <label style={{ fontWeight: 600 }}>Pista:</label>
            <select
              value={selected}
              onChange={e => setSelected(e.target.value)}
              style={{ ...selectStyle, flex: 1 }}
            >
              {tracks.map(t => (
                <option key={t.filename} value={t.filename}>
                  {t.filename}
                </option>
              ))}
              {tracks.length === 0 && (
                <option disabled>{tracksMsg || "Sin pistas"}</option>
              )}
            </select>
          </div>

          {/* reproductor de la pista seleccionada como query */}
          {selected && (
            <AudioPlayer
              src={audioUrl(genre, selected)}
              label={`Escuchar query: ${selected}`}
            />
          )}

          <button
            onClick={searchByFilename}
            disabled={!selected || loading}
            style={btnStyle}
          >
            {loading ? "Buscando..." : "Buscar similares"}
          </button>
        </div>
      )}

      {error && <p style={{ color: "crimson", marginTop: 10 }}>{error}</p>}

      {/* resultados acusticos */}
      {results.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ margin: "0 0 10px" }}>
            {results.length} pistas similares
          </h4>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
              gap: 10,
            }}
          >
            {results.map((r, i) => (
              <ResultCard key={i} result={r} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── componente: tarjeta de resultado con reproductor ──────────────────────

function ResultCard({ result }: { result: AcousticResult }) {
  const url = audioUrl(result.label, result.filename);
  return (
    <div style={cardStyle}>
      <div style={{ fontWeight: 700, fontSize: 13 }}>{result.filename}</div>
      <div style={{ color: "#555", fontSize: 12, textTransform: "capitalize" }}>
        Genero: {result.label}
      </div>
      <div style={scoreStyle}>Similitud: {result.score}</div>
      <AudioPlayer src={url} label="Reproducir" />
    </div>
  );
}

// ── componente: reproductor de audio minimalista ──────────────────────────

function AudioPlayer({ src, label }: { src: string; label?: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError]     = useState(false);

  function toggle() {
    const el = audioRef.current;
    if (!el) return;
    if (playing) {
      el.pause();
      setPlaying(false);
    } else {
      el.play().catch(() => setError(true));
      setPlaying(true);
    }
  }

  function handleEnded() {
    setPlaying(false);
  }

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
      <audio
        ref={audioRef}
        src={src}
        onEnded={handleEnded}
        onError={() => setError(true)}
        preload="none"
      />
      <button
        onClick={toggle}
        style={{
          padding: "4px 12px",
          fontSize: 12,
          border: "1px solid #aaa",
          borderRadius: 4,
          cursor: error ? "not-allowed" : "pointer",
          background: playing ? "#1976d2" : "#f5f5f5",
          color: playing ? "white" : "#333",
        }}
        title={error ? "Audio no disponible" : (playing ? "Pausar" : "Reproducir")}
        disabled={error}
      >
        {error ? "No disponible" : playing ? "Pausar" : label || "Reproducir"}
      </button>
    </div>
  );
}

// ── helpers de UI ──────────────────────────────────────────────────────────

function TabBtn({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "8px 18px",
        border: "none",
        cursor: "pointer",
        fontWeight: active ? 700 : 400,
        borderBottom: active ? "3px solid #1976d2" : "3px solid transparent",
        background: "transparent",
        color: active ? "#1976d2" : "#555",
      }}
    >
      {label}
    </button>
  );
}

function ModeBtn({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
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
const matchStyle: React.CSSProperties = {
  color: "#555",
  fontSize: 12,
  marginTop: 4,
  fontStyle: "italic",
};
const descStyle: React.CSSProperties = {
  color: "#555",
  fontSize: 13,
  margin: "0 0 14px",
};
const cardStyle: React.CSSProperties = {
  border: "1px solid #e0e0e0",
  borderRadius: 8,
  padding: 12,
  background: "#fafafa",
  display: "flex",
  flexDirection: "column",
  gap: 4,
};
