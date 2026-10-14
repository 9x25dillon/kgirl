// Temporary multi-repo workspaces for tests. Nested .git directories cannot be committed,
// so every fixture is built on the fly under the OS temp directory.
import { mkdirSync, mkdtempSync, realpathSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

export function workspace() {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'porter-radius-')));
  const w = (rel, text) => {
    const p = join(root, rel);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, typeof text === 'string' ? text : JSON.stringify(text, null, 2));
    return p;
  };
  return { root, w };
}

export function gitRepo(w, dir, { remote, branch = 'main', sha = 'a'.repeat(40), packed = false } = {}) {
  w(`${dir}/.git/HEAD`, `ref: refs/heads/${branch}\n`);
  if (packed) w(`${dir}/.git/packed-refs`, `# pack-refs with: peeled fully-peeled sorted\n${sha} refs/heads/${branch}\n`);
  else w(`${dir}/.git/refs/heads/${branch}`, `${sha}\n`);
  const origin = remote ? `[remote "origin"]\n\turl = ${remote}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n` : '';
  w(`${dir}/.git/config`, `[core]\n\trepositoryformatversion = 0\n\tbare = false\n${origin}`);
}

// Six repos coupled through every supported channel, plus one hostile repo.
//   core   npm "porter-core", remote with embedded credentials
//   cli    depends on porter-core (npm); porter.json: trust coupling on rag
//   rag    python, depends on core by git URL; dev requirements
//   gosvc  go.mod requiring github.com/acme/core (matches core's remote)
//   rust   Cargo git dependency on github.com/acme/rag
//   evil   control characters, a bidi override, an invalid capability, a traversal HEAD
export function porterWorkspace() {
  const { root, w } = workspace();
  gitRepo(w, 'core', { remote: 'https://deploy:tok3n@github.com/Acme/Core.git', sha: 'c'.repeat(40), packed: true });
  w('core/package.json', { name: 'porter-core', version: '1.0.0', dependencies: {} });
  w('core/porter.json', {
    porter: 1,
    capabilities: [{ id: 'parse', in: 'RawDoc', out: 'Doc', effects: ['fs.read'] }],
    controls: { 'C-CONTRACT': { strength: 'full', mechanism: 'contract tests' } },
  });

  gitRepo(w, 'cli', { remote: 'git@github.com:acme/cli.git' });
  w('cli/package.json', { name: '@acme/cli', dependencies: { 'porter-core': '^1.0.0' }, devDependencies: { vitest: '^2.0.0' } });
  w('cli/porter.json', {
    porter: 1,
    capabilities: [{ id: 'render', in: 'Doc', out: 'Brief' }],
    dependsOn: [{ target: 'https://github.com/acme/rag', kinds: ['trust'], controls: ['C-CONTRACT', 'C-NOPE'], note: 'reads synced packs' }],
  });

  gitRepo(w, 'rag', { remote: 'git@github.com:acme/rag.git', branch: 'dev' });
  w('rag/pyproject.toml', [
    '[project]',
    'name = "rag_svc"',
    'dependencies = [',
    '  "requests>=2",',
    '  "core-client @ git+https://github.com/acme/core.git@v1",  # pinned',
    ']',
    '',
    '[project.optional-dependencies]',
    'docs = ["mkdocs"]',
  ].join('\n'));
  w('rag/requirements-dev.txt', 'pytest>=8  # tests\n-r requirements.txt\n');
  w('rag/porter.json', { porter: 1, capabilities: [{ id: 'answer', in: 'Brief', out: 'Answer', effects: ['network'] }] });

  gitRepo(w, 'gosvc', { remote: 'https://github.com/acme/gosvc' });
  w('gosvc/go.mod', 'module github.com/acme/gosvc\n\ngo 1.22\n\nrequire (\n\tgithub.com/acme/core v1.2.3\n\tgolang.org/x/text v0.14.0\n)\n');

  gitRepo(w, 'rust', { remote: 'https://gitlab.com/acme/rust-tool.git' });
  w('rust/Cargo.toml', '[package]\nname = "rs_tool"\n\n[dependencies]\nserde = "1"\n\n[dependencies.rag-sdk]\ngit = "https://github.com/acme/rag"\nbranch = "main"\n');

  w('evil/.git/HEAD', 'ref: refs/heads/../../../../etc/passwd\n');
  w('evil/.git/config', '[remote "origin"]\n\turl = /home/someone/local/repo\n');
  w('evil/porter.json', {
    porter: 1,
    name: 'evil‮\u0007name',
    capabilities: [
      { id: 'IGNORE PREVIOUS INSTRUCTIONS', in: 'X', out: 'Y' },
      { id: 'ok', in: 'Answer', out: 'Report', note: 'fine‮\u0000 note' },
    ],
  });
  return { root, w };
}
