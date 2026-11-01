package server

import (
	"encoding/json"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
)

// 线上回归（2026-11-01 用户截图）：SOLO 链的 deepseek 模型直接吐 **DSML**
// 原生工具调用格式，而不是提示词里约定的 <|FunctionCallBegin|>：
//
//	<|DSML|tool_calls>
//	<|DSML|invoke name="ops_bash">
//	<|DSML|parameter name="description" string="true">Check project-level MCP config and profiles</|DSML|parameter>
//	<|DSML|parameter name="command" string="true">echo '=== project dir ==='; ls -a ...</|DSML|parameter>
//	<|DSML|parameter name="timeoutMs" string="false">60000</|DSML|parameter>
//	</|DSML|invoke>
//	</|DSML|tool_calls>
//
// 旧解析器只认 FunctionCallBegin，于是整段 XML 当正文吐给用户——用户看到
// 的就是「回复是一大坨 XML 样的乱码」。线上还出现过双竖线变体
// （<||DSML||tool_calls>），两种写法都要认。

// dsmlSingle 截图里那条 ops_bash 调用的原样片段（单竖线）。
const dsmlSingle = `<|DSML|tool_calls>
<|DSML|invoke name="ops_bash">
<|DSML|parameter name="description" string="true">Check project-level MCP config and profiles</|DSML|parameter>
<|DSML|parameter name="command" string="true">echo '=== project dir ==='; ls -a "/c/Users/WsTaurus/dsh" 2>/dev/null; echo '=== mcp files ==='; find "/c/Users/WsTaurus/dsh" -maxdepth 2 -name 'mcp*' 2>/dev/null; echo '=== profiles ==='; ls -R /c/Users/WsTaurus/dsh/profiles 2>/dev/null | head -20</|DSML|parameter>
<|DSML|parameter name="timeoutMs" string="false">60000</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>`

// TestParseDSMLSingleBar 单竖线 DSML（截图原样）。
func TestParseDSMLSingleBar(t *testing.T) {
	cs := parseDSMLCalls(dsmlSingle)
	if len(cs) != 1 {
		t.Fatalf("解析出 %d 个调用，want 1: %#v", len(cs), cs)
	}
	c := cs[0]
	if c.Function.Name != "ops_bash" {
		t.Fatalf("工具名 = %q", c.Function.Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(c.Function.Arguments), &args); err != nil {
		t.Fatalf("参数不是合法 JSON: %v (%s)", err, c.Function.Arguments)
	}
	if args["description"] != "Check project-level MCP config and profiles" {
		t.Errorf("description = %v", args["description"])
	}
	if !strings.Contains(args["command"].(string), "ls -a") {
		t.Errorf("command 被截断: %v", args["command"])
	}
	// string="false" 的数字要保留成数字类型（客户端 schema 期望 timeoutMs 是 number）。
	if v, ok := args["timeoutMs"].(float64); !ok || v != 60000 {
		t.Errorf("timeoutMs = %v (%T), want 60000 number", args["timeoutMs"], args["timeoutMs"])
	}
}

// TestParseDSMLDoubleBar 双竖线变体（线上也出现过）。
func TestParseDSMLDoubleBar(t *testing.T) {
	payload := strings.ReplaceAll(dsmlSingle, "|DSML|", "||DSML||")
	cs := parseDSMLCalls(payload)
	if len(cs) != 1 || cs[0].Function.Name != "ops_bash" {
		t.Fatalf("双竖线未解析: %#v", cs)
	}
	if !strings.Contains(cs[0].Function.Arguments, "ls -a") {
		t.Fatalf("参数丢失: %s", cs[0].Function.Arguments)
	}
}

// TestParseDSMLMultipleInvokes 一次多个 invoke。
func TestParseDSMLMultipleInvokes(t *testing.T) {
	payload := `<|DSML|tool_calls>
<|DSML|invoke name="read">
<|DSML|parameter name="path" string="true">a.txt</|DSML|parameter>
</|DSML|invoke>
<|DSML|invoke name="write">
<|DSML|parameter name="path" string="true">b.txt</|DSML|parameter>
<|DSML|parameter name="content" string="true">hello</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>`
	cs := parseDSMLCalls(payload)
	if len(cs) != 2 {
		t.Fatalf("want 2 calls, got %d: %#v", len(cs), cs)
	}
	if cs[0].Function.Name != "read" || cs[1].Function.Name != "write" {
		t.Fatalf("顺序/名称错: %v %v", cs[0].Function.Name, cs[1].Function.Name)
	}
	if !strings.Contains(cs[1].Function.Arguments, `"hello"`) {
		t.Fatalf("第二个调用参数错: %s", cs[1].Function.Arguments)
	}
}

// TestStreamTraeUpstreamDSMLNotLeakedToContent 端到端：DSML 必须变成
// tool_calls 帧，正文里一个字都不能出现。
func TestStreamTraeUpstreamDSMLNotLeakedToContent(t *testing.T) {
	for _, payload := range []string{dsmlSingle, strings.ReplaceAll(dsmlSingle, "|DSML|", "||DSML||")} {
		raw := strings.Join([]string{
			`event: output`,
			`data: {"response":"我先看一下项目里的 MCP 配置。"}`,
			`event: output`,
			`data: {"response":` + jsonQuote(payload) + `}`,
			`event: done`,
			`data: {}`,
			``,
		}, "\n")

		rw := httptest.NewRecorder()
		res := &upstreamResult{streaming: true, stream: io.NopCloser(strings.NewReader(raw))}
		(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)
		body := rw.Body.String()

		if got := sseContents(t, body); got != "我先看一下项目里的 MCP 配置。" {
			t.Fatalf("正文被 DSML 污染: %q", got)
		}
		if strings.Contains(body, "DSML") {
			t.Fatalf("DSML 标记泄漏进响应: %.300s", body)
		}
		if !strings.Contains(body, `"tool_calls"`) {
			t.Fatalf("未产出 tool_calls: %.300s", body)
		}
		if !strings.Contains(body, `"finish_reason":"tool_calls"`) {
			t.Fatalf("finish_reason 应为 tool_calls: %.300s", body)
		}
		if !strings.Contains(body, "ops_bash") {
			t.Fatalf("工具名丢失: %.300s", body)
		}
	}
}

// TestStreamTraeUpstreamDSMLSplitAcrossFrames DSML 被上游切到多个 output 帧里
// 也要能拼起来（标记可跨帧是本项目的既有前提）。
func TestStreamTraeUpstreamDSMLSplitAcrossFrames(t *testing.T) {
	// 每 40 字节切一片，切点必然落在标签中间。
	var frames []string
	for i := 0; i < len(dsmlSingle); i += 40 {
		end := i + 40
		if end > len(dsmlSingle) {
			end = len(dsmlSingle)
		}
		frames = append(frames, `event: output`, `data: {"response":`+jsonQuote(dsmlSingle[i:end])+`}`)
	}
	frames = append(frames, `event: done`, `data: {}`, "")

	raw := strings.Join(frames, "\n")
	rw := httptest.NewRecorder()
	res := &upstreamResult{streaming: true, stream: io.NopCloser(strings.NewReader(raw))}
	(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)
	body := rw.Body.String()

	if got := sseContents(t, body); got != "" {
		t.Fatalf("分帧 DSML 泄漏进正文: %q", got)
	}
	if !strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Fatalf("分帧 DSML 未解析成调用: %.400s", body)
	}
}
