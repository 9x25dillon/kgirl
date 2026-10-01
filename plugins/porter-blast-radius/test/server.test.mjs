// Drives the real stdio server as an MCP client would.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const PLUGIN = resolve(fileURLToPath(import.meta.url), '..', '..');

// Sends raw lines, collects `expect` response lines, and returns them with stderr.
function session(lines, expect, env = {}) {
  return new Promise((done, failed) => {
    const child = spawn(process.execPath, [resolve(PLUGIN, 'server/index.mjs')], {
      env: { PATH: process.env.PATH, PORTER_RADIUS_PROJECT: PLUGIN, ...env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let out = '';
    let err = '';
    const timer = setTimeout(() => { child.kill(); failed(new Error(`timeout; got: ${out}`)); }, 10000);
    child.stdout.on('data', (d) => {
      out += d;
      const got = out.split('\n').filter(Boolean);
      if (got.length >= expect) { clearTimeout(timer); child.stdin.end(); done({ lines: got, parsed: got.map((l) => JSON.parse(l)), err }); }
    });
    child.stderr.on('data', (d) => { err += d; });
    for (const l of lines) child.stdin.write(`${typeof l === 'string' ? l : JSON.stringify(l)}\n`);
    if (expect === 0) setTimeout(() => { clearTimeout(timer); child.stdin.end(); done({ lines: [], parsed: [], err }); }, 300);
  });
}

const init = (v = '2025-06-18') => ({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: v, capabilities: {}, clientInfo: { name: 'test', version: '0' } } });
const call = (id, name, args) => ({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } });

test('initialize negotiates the protocol version like the SDK', async () => {
  const known = await session([init('2025-06-18')], 1);
  assert.equal(known.parsed[0].result.protocolVersion, '2025-06-18');
  assert.deepEqual(known.parsed[0].result.capabilities, { tools: { listChanged: false } });
  assert.equal(known.parsed[0].result.serverInfo.name, 'porter-blast-radius');
  assert.match(known.parsed[0].result.instructions, /never follow instructions/);
  const unknown = await session([init('1999-01-01')], 1);
  assert.equal(unknown.parsed[0].result.protocolVersion, '2025-11-25');
});

test('tools/list: nine read-only tools with schemas and annotations', async () => {
  const { parsed } = await session([init(), { jsonrpc: '2.0', method: 'notifications/initialized' }, { jsonrpc: '2.0', id: 2, method: 'tools/list' }], 2);
  const tools = parsed[1].result.tools;
  assert.deepEqual(tools.map((t) => t.name), ['list_models', 'check_model', 'blast_radius', 'compose', 'emergent_use_cases', 'risk_register', 'atlas_import', 'merge_models', 'porter_discover']);
  for (const t of tools) {
    assert.equal(t.inputSchema.type, 'object');
    assert.equal(t.inputSchema.additionalProperties, false);
    assert.deepEqual(
      { r: t.annotations.readOnlyHint, d: t.annotations.destructiveHint, o: t.annotations.openWorldHint },
      { r: true, d: false, o: false },
    );
    assert.ok(t.description.length > 40);
  }
});

test('tools/call returns JSON text; tool failures are isError results', async () => {
  const { parsed } = await session([
    init(),
    call(2, 'blast_radius', { model: 'examples/porter-workspace.model.json', sources: ['repo:ctxpack-core'], views: ['change'] }),
    call(3, 'blast_radius', { model: 'examples/porter-workspace.model.json', sources: ['nope'] }),
    call(4, 'blast_radius', { model: '../../../../etc/passwd', sources: ['x'] }),
  ], 4);
  const ok = JSON.parse(parsed[1].result.content[0].text);
  assert.deepEqual(ok.views.change, { unmitigated: [], damped: ['repo:audit', 'repo:porter'], contained: ['repo:ctxr-cli'], impact: [] });
  assert.equal(parsed[2].result.isError, true);
  assert.equal(parsed[3].result.isError, true);
  assert.match(parsed[3].result.content[0].text, /outside the allowed roots|not found/);
});

test('protocol errors use JSON-RPC codes', async () => {
  const { parsed } = await session([
    'not json',
    '[{"jsonrpc":"2.0","id":9,"method":"ping"}]',
    { jsonrpc: '1.0', id: 3, method: 'ping' },
    { jsonrpc: '2.0', id: 4, method: 'resources/list' },
    call(5, 'rm_rf', {}),
    { jsonrpc: '2.0', id: 6, method: 'ping' },
  ], 6);
  assert.deepEqual(parsed.map((p) => p.error?.code ?? 'ok'), [-32700, -32600, -32600, -32601, -32602, 'ok']);
  assert.deepEqual(parsed[5].result, {});
});

test('notifications and client responses get no reply; stdout stays pure JSON', async () => {
  const { lines, err } = await session([
    { jsonrpc: '2.0', method: 'notifications/initialized' },
    { jsonrpc: '2.0', method: 'notifications/cancelled', params: { requestId: 1 } },
    { jsonrpc: '2.0', id: 77, result: {} },
  ], 0);
  assert.deepEqual(lines, []);
  assert.equal(err, '');
});

test('a request split across chunks with a CRLF ending still parses', async () => {
  const msg = JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' });
  const child = spawn(process.execPath, [resolve(PLUGIN, 'server/index.mjs')], { env: { PATH: process.env.PATH, PORTER_RADIUS_PROJECT: PLUGIN } });
  const reply = new Promise((done) => child.stdout.once('data', (d) => done(String(d))));
  child.stdin.write(msg.slice(0, 10));
  await new Promise((r) => setTimeout(r, 50));
  child.stdin.write(`${msg.slice(10)}\r\n`);
  const got = await reply;
  child.stdin.end();
  assert.deepEqual(JSON.parse(got), { jsonrpc: '2.0', id: 1, result: {} });
});
