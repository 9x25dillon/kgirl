import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./routes/Dashboard";
import PrimerPreview from "./routes/PrimerPreview";
import OnboardingStart from "./routes/Onboarding/Start";
import PersonaBasics from "./routes/Onboarding/PersonaBasics";
import ConsentPrivacy from "./routes/Onboarding/ConsentPrivacy";
import ImportSources from "./routes/Onboarding/ImportSources";
import ReviewPin from "./routes/Onboarding/ReviewPin";
import Timeline from "./routes/Memories/Timeline";
import Studio from "./routes/Persona/Studio";
import AdaptersList from "./routes/Adapters/List";
import AdapterDetail from "./routes/Adapters/Detail";
import Backup from "./routes/Settings/Backup";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<OnboardingStart />} />
        <Route path="/onboarding/persona" element={<PersonaBasics />} />
        <Route path="/onboarding/consent" element={<ConsentPrivacy />} />
        <Route path="/onboarding/import" element={<ImportSources />} />
        <Route path="/onboarding/review" element={<ReviewPin />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/primer" element={<PrimerPreview />} />
        <Route path="/memories" element={<Timeline />} />
        <Route path="/persona" element={<Studio />} />
        <Route path="/adapters" element={<AdaptersList />} />
        <Route path="/adapters/:id" element={<AdapterDetail />} />
        <Route path="/settings/backup" element={<Backup />} />
      </Routes>
    </Router>
  );
}
export default App;