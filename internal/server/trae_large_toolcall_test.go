package server

import (
	"encoding/json"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
)

// 线上回归（2026-11-01 用户截图）：
//
// 用户让模型「配置 Burp MCP，写一个 python 桥接脚本」。模型用 write 工具写
// 整份源码，参数字节数远超 traeFnMaxHold(8KB)。上一版实现只按长度判断
// 「标记未闭合 → 当正文放行」，于是：
//
//	1. 一大坨 {"name":"write","parameters":{"file_path":"D:\\...","content":"..."}}
//	   被当正文吐给客户端（截图里那一大段 JSON）；
//	2. 真正的 tool_calls 帧没产出来，客户端显示「已调用工具」但参数是
//	   <![CDATA[]]> 之类的占位内容。
//
// 判定必须结合「载荷像不像调用」，不能只看长度。

// bigWritePayload 构造一个远超 hold 上限的 write 调用。
func bigWritePayload(size int) string {
	unit := "print('hello world')\\n"
	content := strings.Repeat(unit, size/len(unit)+1)
	content = content[:size]
	return `[{"name":"write","parameters":{"file_path":"D:\\burp_mcp_bridge.py","content":"` +
		strings.ReplaceAll(content, `"`, `\"`) + `"}}]`
}

// TestStreamTraeUpstreamLargeToolCallNotLeakedToContent 大参数工具调用
// 必须走 tool_calls，绝不能出现在正文里。
func TestStreamTraeUpstreamLargeToolCallNotLeakedToContent(t *testing.T) {
	payload := bigWritePayload(74 << 10) // 截图里那份脚本量级
	raw := strings.Join([]string{
		`event: output`,
		`data: {"response":` + jsonQuote(traeFnBegin+payload+traeFnEnd) + `}`,
		`event: done`,
		`data: {}`,
		``,
	}, "\n")

	rw := httptest.NewRecorder()
	res := &upstreamResult{streaming: true, stream: io.NopCloser(strings.NewReader(raw))}
	(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)
	body := rw.Body.String()

	// 正文必须干净：不能出现调用载荷的任何片段。
	if got := sseContents(t, body); got != "" {
		t.Fatalf("工具调用载荷泄漏进正文（%d 字节）: %.200q", len(got), got)
	}
	if strings.Contains(body, `"content"`) {
		t.Fatalf("载荷片段出现在响应里")
	}
	// 必须产出 tool_calls 帧。
	if !strings.Contains(body, `"tool_calls"`) {
		t.Fatalf("未产出 tool_calls: %.400s", body)
	}
	if !strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Fatalf("finish_reason 应为 tool_calls: %.400s", body)
	}
}

// TestStreamTraeUpstreamLargeToolCallArgumentsIntact 参数必须完整送达客户端
// （不能被 hold 上限或任何截断逻辑砍掉）。
func TestStreamTraeUpstreamLargeToolCallArgumentsIntact(t *testing.T) {
	const size = 40 << 10
	payload := bigWritePayload(size)
	raw := strings.Join([]string{
		`event: output`,
		`data: {"response":` + jsonQuote(traeFnBegin+payload+traeFnEnd) + `}`,
		`event: done`,
		`data: {}`,
		``,
	}, "\n")

	rw := httptest.NewRecorder()
	res := &upstreamResult{streaming: true, stream: io.NopCloser(strings.NewReader(raw))}
	(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)

	// 从 SSE 里抠出 tool_calls 的 arguments，检查长度与可解析性。
	var args string
	for _, line := range strings.Split(rw.Body.String(), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") || line == "data: [DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk) != nil {
			continue
		}
		for _, c := range chunk.Choices {
			for _, tc := range c.Delta.ToolCalls {
				if tc.Function.Name == "write" {
					args = tc.Function.Arguments
				}
			}
		}
	}
	if args == "" {
		t.Fatal("没拿到 write 调用的 arguments")
	}
	if len(args) < size {
		t.Fatalf("arguments 被截断: %d < %d", len(args), size)
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(args), &obj); err != nil {
		t.Fatalf("arguments 不是合法 JSON: %v", err)
	}
	if obj["file_path"] != `D:\burp_mcp_bridge.py` {
		t.Fatalf("file_path 变形: %v", obj["file_path"])
	}
}

// TestStreamTraeUpstreamTruncatedCallNotLeaked 上游在工具调用中途断流时，
// 半截 JSON 也不能当正文吐给用户。
func TestStreamTraeUpstreamTruncatedCallNotLeaked(t *testing.T) {
	raw := strings.Join([]string{
		`event: output`,
		`data: {"response":` + jsonQuote(traeFnBegin+`[{"name":"write","parameters":{"file_path":"D:\\a.py","content":"#!/usr/bin/env py`) + `}`,
		``, // 没有 done，直接 EOF
	}, "\n")

	rw := httptest.NewRecorder()
	res := &upstreamResult{streaming: true, stream: &cleanEOFReader{data: []byte(raw)}}
	(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)

	if got := sseContents(t, rw.Body.String()); got != "" {
		t.Fatalf("半截工具调用泄漏进正文: %q", got)
	}
}
