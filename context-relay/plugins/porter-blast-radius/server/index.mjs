#!/usr/bin/env node
// porter-blast-radius MCP server over stdio. No dependencies.
// Framing and negotiation follow the MCP TypeScript SDK (1.31.0): one JSON-RPC message
// per line, the client's protocol version echoed when supported, else the latest.

import { INSTRUCTIONS, TOOLS, VERSION, callTool, makeContext, publicSpec } from './tools.mjs';

const SUPPORTED_PROTOCOL_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'];
const MAX_LINE_BYTES = 10 * 1024 * 1024; // the SDK's STDIO_DEFAULT_MAX_BUFFER_SIZE

// stdout carries protocol messages only; anything else a module prints goes to stderr.
const toStderr = (...a) => process.stderr.write(`${a.map(String).join(' ')}\n`);
console.log = toStderr;
console.info = toStderr;
console.warn = toStderr;
console.debug = toStderr;

const send = (msg) => process.stdout.write(`${JSON.stringify(msg)}\n`);
const reply = (id, result) => send({ jsonrpc: '2.0', id, result });
const fail = (id, code, message) => send({ jsonrpc: '2.0', id, error: { code, message } });

const ctx = makeContext(process.env);

function handle(line) {
  if (!line.trim()) return;
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return fail(null, -32700, 'parse error');
  }
  if (Array.isArray(msg)) return fail(null, -32600, 'JSON-RPC batches are not supported');
  if (!msg || typeof msg !== 'object' || msg.jsonrpc !== '2.0') return fail(msg?.id ?? null, -32600, 'invalid request');
  if (typeof msg.method !== 'string') return; // a response to a request we never sent
  const isRequest = msg.id !== undefined && msg.id !== null;
  if (!isRequest) return; // notifications: initialized, cancelled, roots/list_changed

  try {
    switch (msg.method) {
      case 'initialize': {
        const requested = msg.params?.protocolVersion;
        return reply(msg.id, {
          protocolVersion: SUPPORTED_PROTOCOL_VERSIONS.includes(requested) ? requested : SUPPORTED_PROTOCOL_VERSIONS[0],
          capabilities: { tools: { listChanged: false } },
          serverInfo: { name: 'porter-blast-radius', title: 'Porter Blast Radius', version: VERSION },
          instructions: INSTRUCTIONS,
        });
      }
      case 'ping':
        return reply(msg.id, {});
      case 'tools/list':
        return reply(msg.id, { tools: TOOLS.map(publicSpec) });
      case 'tools/call': {
        const name = msg.params?.name;
        if (!TOOLS.some((t) => t.name === name)) return fail(msg.id, -32602, `unknown tool: ${String(name).slice(0, 80)}`);
        return reply(msg.id, callTool(ctx, name, msg.params?.arguments ?? {}));
      }
      default:
        return fail(msg.id, -32601, `method not found: ${msg.method.slice(0, 80)}`);
    }
  } catch (e) {
    return fail(msg.id, -32603, `internal error: ${e.message}`);
  }
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  if (Buffer.byteLength(buffer) > MAX_LINE_BYTES && !buffer.includes('\n')) {
    buffer = '';
    fail(null, -32600, `message exceeds ${MAX_LINE_BYTES} bytes`);
    return;
  }
  let i;
  while ((i = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, i).replace(/\r$/, '');
    buffer = buffer.slice(i + 1);
    handle(line);
  }
});
process.stdin.on('end', () => process.exit(0));
