export default function ImportSources() {
  return (
    <main className="p-8">
      <h2 className="text-xl font-bold mb-4">Import Memories</h2>
      <div>
        <label>Drop JSON/CSV export here:</label>
        {/* FileDrop component placeholder */}
        <input type="file" accept=".json,.csv" multiple />
      </div>
      <div className="mt-4">
        <button className="btn btn-primary">Next</button>
      </div>
    </main>
  );
}