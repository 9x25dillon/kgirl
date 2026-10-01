// Static guarantees behind the connector's self-model: the shipped code cannot run
// programs, reach the network or write files, and the manifests agree with the code.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { VERSION } from '../server/tools.mjs';

const PLUGIN = resolve(fileURLToPath(import.meta.url), '..', '..');
const RELAY = resolve(PLUGIN, '..', '..');
const shipped = ['lib', 'server'].flatMap((d) => readdirSync(join(PLUGIN, d)).filter((f) => f.endsWith('.mjs')).map((f) => join(PLUGIN, d, f)));
const source = (f) => readFileSync(f, 'utf8').replace(/^\s*\/\/.*$/gm, '');

const FORBIDDEN = [
  [/from\s+['"](node:)?(child_process|net|http|https|http2|dgram|tls|dns|worker_threads|vm|cluster|inspector)['"]/, 'process, network or VM module'],
  [/\bimport\s*\(/, 'dynamic import'],
  [/\brequire\s*\(/, 'require'],
  [/\bfetch\s*\(|\bWebSocket\b|\bXMLHttpRequest\b/, 'network API'],
  [/\b(writeFile|appendFile|mkdir|mkdtemp|rm|rmdir|unlink|rename|copyFile|cp|symlink|link|chmod|chown|truncate|utimes|createWriteStream|open)(Sync)?\s*\(/, 'filesystem write'],
  [/\beval\s*\(|new\s+Function\s*\(/, 'code evaluation'],
  [/process\.(binding|dlopen)|process\.env\.[A-Z_]+\s*=/, 'process mutation'],
];

for (const f of shipped) {
  test(`${f.slice(PLUGIN.length + 1)} has no exec, network or write capability`, () => {
    const s = source(f);
    for (const [re, what] of FORBIDDEN) assert.ok(!re.test(s), `${what} found in ${f}`);
  });
}

test('there are no runtime dependencies', () => {
  assert.equal(existsSync(join(PLUGIN, 'package.json')), false);
  assert.equal(existsSync(join(PLUGIN, 'node_modules')), false);
  for (const f of shipped) {
    for (const m of source(f).matchAll(/from\s+['"]([^'"]+)['"]/g)) assert.ok(m[1].startsWith('node:') || m[1].startsWith('.'), `${f} imports ${m[1]}`);
  }
});

test('plugin.json, marketplace.json, .mcp.json and the server agree', () => {
  const plugin = JSON.parse(readFileSync(join(PLUGIN, '.claude-plugin/plugin.json'), 'utf8'));
  assert.equal(plugin.version, VERSION);
  for (const opt of Object.values(plugin.userConfig)) assert.ok(opt.type && opt.title && opt.description, 'userConfig needs type, title, description');
  const mcp = JSON.parse(readFileSync(join(PLUGIN, '.mcp.json'), 'utf8')).mcpServers['blast-radius'];
  assert.equal(mcp.command, 'node');
  assert.ok(existsSync(mcp.args[0].replace('${CLAUDE_PLUGIN_ROOT}', PLUGIN)));
  for (const ref of JSON.stringify(mcp.env).matchAll(/\$\{user_config\.([a-z_]+)\}/g)) assert.ok(plugin.userConfig[ref[1]], `undeclared user_config ${ref[1]}`);
  const market = JSON.parse(readFileSync(join(RELAY, '.claude-plugin/marketplace.json'), 'utf8'));
  const entry = market.plugins.find((p) => p.name === plugin.name);
  assert.ok(entry);
  assert.equal(resolve(RELAY, entry.source), PLUGIN);
});

test('the skill frontmatter has only name and description', () => {
  const text = readFileSync(join(PLUGIN, 'skills/blast-radius/SKILL.md'), 'utf8');
  const fm = /^---\n([\s\S]*?)\n---\n/.exec(text)[1];
  const keys = fm.split('\n').map((l) => l.split(':')[0]);
  assert.deepEqual(keys, ['name', 'description']);
  assert.match(fm, /^name: blast-radius$/m);
  assert.ok(fm.length < 1100);
  assert.ok(!/claude|anthropic/i.test(/^name: (.*)$/m.exec(fm)[1]));
});

test('the engine exists once: the design-doc tool imports the plugin copy', () => {
  const relayTool = readFileSync(join(RELAY, 'tools/relay-model.mjs'), 'utf8');
  assert.match(relayTool, /from '\.\.\/plugins\/porter-blast-radius\/lib\/engine\.mjs'/);
  assert.ok(!/export function propagate/.test(relayTool));
  assert.equal(readFileSync(join(PLUGIN, 'lib/engine.mjs'), 'utf8').match(/export function propagate/g).length, 1);
});

test('SQLite is opened read-only everywhere and the bridge issues no write statements', () => {
  for (const f of shipped) {
    const s = source(f);
    for (const m of s.matchAll(/new\s+[\w.]*DatabaseSync\s*\(([^)]*)\)/g)) assert.match(m[1], /readOnly:\s*true/, `${f} opens SQLite without readOnly`);
  }
  const bridge = source(join(PLUGIN, 'lib/atlas.mjs'));
  const sql = [...bridge.matchAll(/`([^`]*)`|'([^'\n]*)'/g)].map((m) => m[1] ?? m[2]).join('\n');
  assert.ok(!/\b(INSERT|UPDATE|DELETE|DROP|CREATE|ATTACH|REPLACE\s+INTO|VACUUM|ALTER)\b/i.test(sql), 'write SQL in lib/atlas.mjs');
  assert.ok(!/\.exec\s*\(/.test(bridge), 'db.exec in lib/atlas.mjs');
});
