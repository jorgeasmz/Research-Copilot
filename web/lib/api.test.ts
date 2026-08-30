import { describe, expect, it } from "vitest";

import { splitEvents } from "./api";

const CRLF = "event: token\r\ndata: \"first\"\r\n\r\nevent: token\r\ndata: \"second\"\r\n\r\n";
const LF = 'event: token\ndata: "first"\n\nevent: token\ndata: "second"\n\n';

describe("splitEvents", () => {
  it("reads events terminated with CRLF, which is what the server sends", () => {
    const { events } = splitEvents(CRLF);

    expect(events.map((e) => e.data)).toEqual(["first", "second"]);
  });

  it("reads events terminated with bare newlines", () => {
    expect(splitEvents(LF).events.map((e) => e.data)).toEqual(["first", "second"]);
  });

  it("keeps the event name", () => {
    const { events } = splitEvents('event: citations\r\ndata: {"invalid": [9]}\r\n\r\n');

    expect(events[0].event).toBe("citations");
    expect(events[0].data).toEqual({ invalid: [9] });
  });

  it("holds an incomplete event back for the next chunk", () => {
    const { events, rest } = splitEvents('event: token\r\ndata: "done"\r\n\r\nevent: tok');

    expect(events).toHaveLength(1);
    expect(rest).toBe("event: tok");
  });

  it("reassembles an event split across arbitrary chunk boundaries", () => {
    let buffer = "";
    const collected: unknown[] = [];

    for (let i = 0; i < CRLF.length; i += 7) {
      buffer += CRLF.slice(i, i + 7);
      const { events, rest } = splitEvents(buffer);
      buffer = rest;
      collected.push(...events.map((e) => e.data));
    }

    expect(collected).toEqual(["first", "second"]);
  });

  it("ignores a block carrying no data line", () => {
    expect(splitEvents(": keep-alive\r\n\r\n").events).toEqual([]);
  });
});
