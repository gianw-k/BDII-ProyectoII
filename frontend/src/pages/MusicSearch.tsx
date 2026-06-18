// App 2: buscar canciones por letra (full-text via motor propio)
import { useState } from "react";

type Song = {
  title: string;
  artist: string;
  score: number;
  match: string;
};

export function MusicSearch() {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<Song[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function search() {
    if (!q.trim()) return;
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`/api/music/search/lyrics?q=${encodeURIComponent(q)}&top_n=10`);
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
      <h2>Buscar música por letra</h2>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          type="text"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search()}
          placeholder="ej: love you baby"
          style={{ flex: 1, padding: 6 }}
        />
        <button onClick={search} disabled={loading}>
          {loading ? "Buscando..." : "Buscar"}
        </button>
      </div>

      {error && <p style={{ color: "crimson" }}>{error}</p>}

      <ol style={{ marginTop: 16 }}>
        {results.map((s, i) => (
          <li key={i} style={{ marginBottom: 10 }}>
            <strong>{s.title}</strong> — {s.artist}{" "}
            <span style={{ color: "#888" }}>({s.score})</span>
            <div style={{ color: "#555", fontSize: 13 }}>{s.match}</div>
          </li>
        ))}
      </ol>
    </section>
  );
}
