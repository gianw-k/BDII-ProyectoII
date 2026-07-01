// Consola SQL: cara unificada del motor. El usuario escribe una consulta tipo
// SQL y el backend la parsea (ParserSQL) y la despacha a la modalidad correcta
// (texto / audio / imagen), todas sobre el mismo nucleo de indice invertido.

import { useState } from "react";

type Parsed = {
  modality: string;
  collection: string;
  field: string;
  op: string;
  value: string;
  limit: number;
  fields: string[];
};

type Row = Record<string, unknown> & { score?: number };

type Response = {
  parsed: Parsed;
  count: number;
  results: Row[];
};

const EXAMPLES = [
  "SELECT * FROM songs WHERE lyrics @@ 'love you baby' LIMIT 10",
  "SELECT title, artist FROM songs WHERE lyrics LIKE 'midnight rain'",
  "SELECT * FROM tracks WHERE audio <-> 'blues.00000.wav' LIMIT 5",
];

export function QueryConsole() {
  const [sql, setSql] = useState(EXAMPLES[0]);
  const [resp, setResp] = useState<Response | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function run() {
    if (!sql.trim()) return;
    setLoading(true);
    setError("");
    setResp(null);
    try {
      const r = await fetch("/api/query/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sql }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || `error ${r.status}`);
      setResp(body);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setLoading(false);
    }
  }

  // columnas de la tabla = union de claves de las filas (orden estable)
  const columns =
    resp && resp.results.length
      ? Array.from(
          resp.results.reduce<Set<string>>((set, row) => {
            Object.keys(row).forEach((k) => set.add(k));
            return set;
          }, new Set())
        )
      : [];

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <h2 style={{ margin: 0 }}>Consola SQL multimodal</h2>
      <p style={descStyle}>
        Consulta el motor con un mini-lenguaje tipo SQL. Una sola sintaxis sirve
        para texto, audio e imagen: cambia la coleccion y el operador.
      </p>

      <textarea
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
        }}
        spellCheck={false}
        rows={3}
        style={textareaStyle}
        placeholder="SELECT * FROM songs WHERE lyrics @@ 'tu consulta' LIMIT 10"
      />

      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <button onClick={run} disabled={loading} style={btnStyle}>
          {loading ? "Ejecutando..." : "Ejecutar (Ctrl+Enter)"}
        </button>
        <span style={{ color: "#888", fontSize: 12 }}>Ejemplos:</span>
        {EXAMPLES.map((ex, i) => (
          <button
            key={i}
            onClick={() => setSql(ex)}
            style={chipStyle}
            title={ex}
          >
            {ex.split(" FROM ")[1]?.split(" ")[0] ?? `ej ${i + 1}`}
          </button>
        ))}
      </div>

      {error && (
        <pre style={errBox}>{error}</pre>
      )}

      {resp && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <ParsedBadge parsed={resp.parsed} count={resp.count} />
          {resp.results.length === 0 ? (
            <p style={{ color: "#888" }}>Sin resultados.</p>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    {columns.map((c) => (
                      <th key={c} style={thStyle}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {resp.results.map((row, i) => (
                    <tr key={i} style={{ background: i % 2 ? "#fafafa" : "white" }}>
                      {columns.map((c) => (
                        <td key={c} style={tdStyle}>{fmt(row[c])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function ParsedBadge({ parsed, count }: { parsed: Parsed; count: number }) {
  return (
    <div style={badgeBox}>
      <span style={pill}>modalidad: {parsed.modality}</span>
      <span style={pill}>coleccion: {parsed.collection}</span>
      <span style={pill}>op: {parsed.op}</span>
      <span style={pill}>limit: {parsed.limit}</span>
      <span style={{ ...pill, background: "#e3f0ff", color: "#1976d2" }}>
        {count} resultados
      </span>
    </div>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return v.toString();
  return String(v);
}

// ── estilos ──────────────────────────────────────────────────────────────────

const descStyle: React.CSSProperties = { color: "#555", fontSize: 13, margin: 0 };
const textareaStyle: React.CSSProperties = {
  width: "100%",
  fontFamily: "ui-monospace, Menlo, monospace",
  fontSize: 14,
  padding: 12,
  border: "1px solid #ccc",
  borderRadius: 6,
  resize: "vertical",
  boxSizing: "border-box",
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
const chipStyle: React.CSSProperties = {
  padding: "4px 10px",
  fontSize: 12,
  border: "1px solid #ccc",
  borderRadius: 14,
  background: "white",
  cursor: "pointer",
  color: "#555",
};
const errBox: React.CSSProperties = {
  background: "#fff0f0",
  border: "1px solid #f3b7b7",
  color: "#b71c1c",
  padding: 12,
  borderRadius: 6,
  fontSize: 13,
  whiteSpace: "pre-wrap",
  margin: 0,
};
const badgeBox: React.CSSProperties = { display: "flex", gap: 6, flexWrap: "wrap" };
const pill: React.CSSProperties = {
  background: "#f0f0f0",
  borderRadius: 12,
  padding: "3px 10px",
  fontSize: 12,
  color: "#555",
};
const tableStyle: React.CSSProperties = {
  borderCollapse: "collapse",
  width: "100%",
  fontSize: 13,
};
const thStyle: React.CSSProperties = {
  textAlign: "left",
  borderBottom: "2px solid #ddd",
  padding: "6px 10px",
  color: "#333",
};
const tdStyle: React.CSSProperties = {
  borderBottom: "1px solid #eee",
  padding: "6px 10px",
  color: "#444",
  maxWidth: 320,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};
