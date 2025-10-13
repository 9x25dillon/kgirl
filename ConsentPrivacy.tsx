export default function ConsentPrivacy() {
  return (
    <main className="p-8">
      <h2 className="text-xl font-bold mb-4">Consent & Privacy</h2>
      <div>
        <label>
          <input type="checkbox" /> Enable cloud sync (encrypted)
        </label>
      </div>
      <div>
        <label>
          <input type="checkbox" /> Retain sensitive memories
        </label>
      </div>
      <div className="mt-4">
        <button className="btn btn-primary">Next</button>
      </div>
    </main>
  );
}