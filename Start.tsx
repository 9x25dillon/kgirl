export default function OnboardingStart() {
  return (
    <main className="p-8">
      <h1 className="text-3xl font-bold mb-4">Welcome to CarryOn</h1>
      <p className="mb-6">Keep your AI you across updates & devices.</p>
      <div className="flex gap-4">
        <button className="btn btn-primary">Create Soulpack</button>
        <button className="btn btn-secondary">Import Soulpack</button>
      </div>
    </main>
  );
}