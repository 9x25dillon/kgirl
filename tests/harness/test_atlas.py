import unittest
from pathlib import Path

from tests.harness._fixtures import make_repos

from kgirl.harness.atlas import Atlas, architecture_card, blast_radius, coupling
from kgirl.harness.atlas.parse import extract


class ParseTests(unittest.TestCase):
    def test_python_symbols_signatures_and_refs(self):
        f = extract("m.py", "class A(B):\n    '''Doc.'''\n    def f(self, x: int = 1) -> str:\n"
                            "        return g(x)\n", "python")
        kinds = {(s.kind, s.qualname) for s in f.symbols}
        self.assertIn(("class", "A"), kinds)
        self.assertIn(("method", "A.f"), kinds)
        sig = next(s.signature for s in f.symbols if s.name == "f")
        self.assertEqual(sig, "def f(self, x: int=1) -> str")
        self.assertIn("g", f.refs)

    def test_broken_python_falls_back_to_regex(self):
        f = extract("b.py", "def ok():\n    return 1\n\ndef bad(:\n", "python")
        self.assertTrue(f.parse_error)
        self.assertIn("ok", {s.name for s in f.symbols})

    def test_clone_hash_ignores_formatting_and_docstring(self):
        a = extract("a.py", "def f(v):\n    '''x'''\n    t = sum(v)\n    return [i / t for i in v if t]\n", "python")
        b = extract("b.py", "def f(v):\n    t = sum( v )\n    return [i/t for i in v if t]\n", "python")
        self.assertTrue(a.symbols[0].body_hash)
        self.assertEqual(a.symbols[0].body_hash, b.symbols[0].body_hash)

    def test_typescript_and_julia(self):
        ts = extract("x.ts", "export const add = (a: number) => a;\nexport class K {}\nimport {y} from './y';\n",
                     "typescript")
        self.assertEqual({s.name for s in ts.symbols}, {"add", "K"})
        self.assertEqual(ts.imports[0].target, "./y")
        jl = extract("x.jl", "module M\nusing LinearAlgebra, Random\nf(x) = x + 1\nfunction g(y)\n y\nend\n"
                             "include(\"h.jl\")\nend\n", "julia")
        self.assertTrue({"M", "f", "g"} <= {s.name for s in jl.symbols})
        self.assertEqual([i.target for i in jl.imports], ["LinearAlgebra", "Random", "h.jl"])


class AtlasTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.hub, self.lib = make_repos()
        self.atlas = Atlas(Path(self.tmp.name) / "atlas.db")
        self.atlas.ingest(self.lib, "lib")
        self.atlas.ingest(self.hub, "hub")

    def tearDown(self):
        self.atlas.close()
        self.tmp.cleanup()

    def test_ingest_skips_console_shims_and_is_incremental(self):
        paths = {r["path"] for r in self.atlas.db.execute("SELECT path FROM files")}
        self.assertNotIn("tqdm", paths)
        st = self.atlas.ingest(self.hub, "hub")
        self.assertEqual((st.changed, st.removed), (0, 0))
        (self.hub / "util.py").write_text("def changed():\n    return 2\n")
        (self.hub / "broken.py").unlink()
        st = self.atlas.ingest(self.hub, "hub")
        self.assertEqual((st.changed, st.removed), (1, 1))
        self.assertTrue(self.atlas.find_symbol("changed"))
        self.assertFalse(self.atlas.find_symbol("ok"))

    def test_search_ranks_symbol(self):
        hits = self.atlas.search("engine run")
        self.assertTrue(hits)
        self.assertIn(hits[0].qualname, {"Engine", "Engine.run"})
        self.assertEqual(hits[0].repo, "lib")

    def test_cross_repo_import_resolution(self):
        row = self.atlas.db.execute(
            "SELECT f2.path, r2.name FROM imports i JOIN files f ON f.id=i.file_id "
            "JOIN files f2 ON f2.id=i.resolved_file_id JOIN repos r2 ON r2.id=f2.repo_id "
            "WHERE f.path='app.py'").fetchone()
        self.assertEqual((row["name"], row["path"]), ("lib", "engine.py"))
        rel = self.atlas.db.execute(
            "SELECT f2.path FROM imports i JOIN files f ON f.id=i.file_id JOIN files f2 ON f2.id=i.resolved_file_id"
            " WHERE f.path='pkg/rel.py'").fetchone()
        self.assertEqual(rel["path"], "pkg/other.py")
        ts = self.atlas.db.execute(
            "SELECT f2.path FROM imports i JOIN files f ON f.id=i.file_id JOIN files f2 ON f2.id=i.resolved_file_id"
            " WHERE f.path='web/main.ts'").fetchone()
        self.assertEqual(ts["path"], "web/math.ts")

    def test_blast_radius_file_symbol_and_clone(self):
        rep = blast_radius(self.atlas, "lib:engine.py")
        got = {(i.repo, i.path): (i.depth, i.via) for i in rep.impacts}
        self.assertEqual(got[("hub", "app.py")], (1, "import"))
        self.assertEqual(got[("hub", "pkg/other.py")], (2, "import"))
        self.assertEqual(got[("hub", "pkg/rel.py")][0], 3)
        self.assertEqual(got[("hub", "util.py")], (1, "clone"))
        sym = blast_radius(self.atlas, "engine.py::Engine")
        self.assertEqual({(i.path, i.via) for i in sym.impacts if i.depth == 1}, {("app.py", "ref")})
        self.assertFalse(blast_radius(self.atlas, "Nope").seeds)

    def test_coupling_and_card(self):
        c = {x.repo: x for x in coupling(self.atlas, "hub")}["lib"]
        self.assertEqual(c.imports_from, 1)
        self.assertGreaterEqual(c.clones, 1)
        self.assertEqual(c.hub_files_exposed, 3)
        card = architecture_card(self.atlas, "hub")
        self.assertIn("app.py", card["entrypoints"])
        self.assertEqual(card["unparseable_python_files"], 1)
        self.assertIn("python", card["languages"])


if __name__ == "__main__":
    unittest.main()
