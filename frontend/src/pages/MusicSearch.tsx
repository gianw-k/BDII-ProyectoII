// App 2: buscar canciones por letra (full-text) o por similitud acustica
export function MusicSearch() {
  // TODO
  return (
    <section>
      <h2>Buscar música</h2>
      <p>Por letra (texto) o por similitud acústica (audio).</p>
      <input type="text" placeholder="Buscar por letra..." disabled />
      <div style={{ marginTop: 12 }}>
        <input type="file" accept="audio/*" disabled />
      </div>
      {/* TODO */}
    </section>
  );
}
