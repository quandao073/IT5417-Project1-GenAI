import { describe, expect, it } from "vitest";

import { readNdjson } from "@/lib/api";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const ch of chunks) c.enqueue(enc.encode(ch));
      c.close();
    },
  });
}

async function all<T>(gen: AsyncGenerator<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const x of gen) out.push(x);
  return out;
}

describe("readNdjson", () => {
  it("yields each event, even when a line is split across chunks", async () => {
    const got = await all(readNdjson(streamOf([
      '{"event":"results","n":1}\n{"event":"expl', 'anation","summary":"tim to"}\n',
    ])));
    expect(got).toEqual([{ event: "results", n: 1 }, { event: "explanation", summary: "tim to" }]);
  });

  it("decodes a multi-byte character split between chunks", async () => {
    const bytes = new TextEncoder().encode('{"s":"phổi"}\n');
    const cut = bytes.length - 4;
    const stream = new ReadableStream<Uint8Array>({
      start(c) { c.enqueue(bytes.slice(0, cut)); c.enqueue(bytes.slice(cut)); c.close(); },
    });
    expect(await all(readNdjson(stream))).toEqual([{ s: "phổi" }]);
  });

  it("reads a last line without a newline and skips blank lines", async () => {
    expect(await all(readNdjson(streamOf(['\n{"a":1}\n\n', '{"b":2}'])))).toEqual([{ a: 1 }, { b: 2 }]);
  });
});
