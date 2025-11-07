export default function PersonaBasics() {
  return (
    <main className="p-8">
      <h2 className="text-xl font-bold mb-4">Set Your Persona</h2>
      <div>
        <label>Tone</label>
        {/* ToneSlider component */}
        <input type="range" min={0} max={100} />
      </div>
      <div className="mt-4">
        <label>Name</label>
        <input type="text" className="input" />
      </div>
      <div className="mt-4">
        <button className="btn btn-primary">Next</button>
      </div>
    </main>
  );
}