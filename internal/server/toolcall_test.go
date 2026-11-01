package server

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestAnthToolsFromOAI(t *testing.T) {
	raw := json.RawMessage("[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"w\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]")
	out := anthToolsFromOAI(raw)
	if len(out) == 0 {
		t.Fatal("expected anthropic tools")
	}
	var arr []map[string]any
	if err := json.Unmarshal(out, &arr); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(arr) != 1 || arr[0]["name"] != "get_weather" {
		t.Fatalf("bad tool: %#v", arr)
	}
	if _, ok := arr[0]["input_schema"].(map[string]any); !ok {
		t.Fatalf("input_schema missing: %#v", arr[0])
	}
	if _, has := arr[0]["function"]; has {
		t.Fatal("openai wrapper leaked")
	}
}

func TestAnthToolsFromOAIPassthrough(t *testing.T) {
	raw := json.RawMessage("[{\"name\":\"get_weather\",\"description\":\"w\",\"input_schema\":{\"type\":\"object\"}}]")
	out := string(anthToolsFromOAI(raw))
	if !strings.Contains(out, "input_schema") || strings.Contains(out, "\"function\"") {
		t.Fatalf("bad passthrough: %s", out)
	}
}

func TestAnthToolChoiceFromOAI(t *testing.T) {
	cases := []struct{ in, want string }{
		{"\"auto\"", "{\"type\":\"auto\"}"},
		{"\"required\"", "{\"type\":\"any\"}"},
		{"\"none\"", ""},
		{"{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}", "{\"name\":\"get_weather\",\"type\":\"tool\"}"},
	}
	for _, c := range cases {
		got := string(anthToolChoiceFromOAI(json.RawMessage(c.in)))
		if c.want == "" {
			if got != "" {
				t.Fatalf("in=%s want none got %s", c.in, got)
			}
			continue
		}
		var a, b map[string]any
		if json.Unmarshal([]byte(got), &a) != nil || json.Unmarshal([]byte(c.want), &b) != nil {
			t.Fatalf("bad json in=%s got=%s", c.in, got)
		}
		if a["type"] != b["type"] {
			t.Fatalf("in=%s got=%s want=%s", c.in, got, c.want)
		}
	}
}

func TestAnthMessagesFromOAIToolRoundTrip(t *testing.T) {
	in := []OAIMessage{
		{Role: "user", Content: json.RawMessage("\"北京天气\"")},
		{Role: "assistant", ToolCalls: json.RawMessage("[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"北京\\\"}\"}}]")},
		{Role: "tool", ToolCallID: "call_1", Content: json.RawMessage("\"晴 20C\"")},
	}
	msgs, sys := anthMessagesFromOAI(in)
	if sys != "" {
		t.Fatalf("unexpected system: %s", sys)
	}
	b, _ := json.Marshal(msgs)
	got := string(b)
	for _, want := range []string{"tool_use", "tool_result", "call_1", "get_weather"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %s in %s", want, got)
		}
	}
}

func TestTraeFnParserBasic(t *testing.T) {
	p := newTraeFnParser()
	out, calls := p.feed("<|FunctionCallBegin|>[{\"name\":\"get_weather\",\"parameters\":{\"city\":\"北京\"}}]<|FunctionCallEnd|>")
	if out != "" {
		t.Fatalf("expected empty text, got %q", out)
	}
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("bad calls: %#v", calls)
	}
	if calls[0].Function.Arguments != "{\"city\":\"北京\"}" {
		t.Fatalf("bad args: %s", calls[0].Function.Arguments)
	}
}

func TestTraeFnParserSplitAcrossChunks(t *testing.T) {
	p := newTraeFnParser()
	var text strings.Builder
	var calls []oaiToolCall
	chunks := []string{"好的，", "<|Func", "tionCallBe", "gin|>[{\"na", "me\":\"get_weather\",\"parameters\":{\"city\":\"上海\"}}]<|FunctionCall", "End|>", "完成了"}
	for _, c := range chunks {
		o, cs := p.feed(c)
		text.WriteString(o)
		calls = append(calls, cs...)
	}
	o, cs := p.flush()
	text.WriteString(o)
	calls = append(calls, cs...)
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("bad calls: %#v", calls)
	}
	if calls[0].Function.Arguments != "{\"city\":\"上海\"}" {
		t.Fatalf("bad args: %s", calls[0].Function.Arguments)
	}
	if text.String() != "好的，完成了" {
		t.Fatalf("bad text: %q", text.String())
	}
}

func TestTraeFnParserNoMarkerPassThrough(t *testing.T) {
	p := newTraeFnParser()
	var text strings.Builder
	for _, c := range []string{"普通", "回答", "没有工具"} {
		o, _ := p.feed(c)
		text.WriteString(o)
	}
	o, _ := p.flush()
	text.WriteString(o)
	if text.String() != "普通回答没有工具" {
		t.Fatalf("got %q", text.String())
	}
}

func TestParseTraeCallsSingleObject(t *testing.T) {
	cs := parseTraeCalls("{\"name\":\"f\",\"parameters\":{\"a\":1}}")
	if len(cs) != 1 || cs[0].Function.Name != "f" {
		t.Fatalf("bad: %#v", cs)
	}
}

// TestTraeFnParserFlushKeepsUnclosedPayload 回归「flush 吞正文」：
// 模型输出了 <|FunctionCallBegin|> 却**不是**在调工具（误发标记，载荷不像
// JSON 调用）时，flush 必须把载荷当正文补发，不能整段丢弃。
//
// 反例（必须丢弃）见 TestTraeFnParserFlushDropsTruncatedCall：
// 载荷是真工具调用（以 [ 或 { 开头）时绝不能当正文吐给用户。
func TestTraeFnParserFlushKeepsUnclosedPayload(t *testing.T) {
	p := newTraeFnParser()
	if out, _ := p.feed("前面的正文" + traeFnBegin + `我先看一下目录结构`); out == "" {
		t.Fatal("前面的正文应在 hold-back 之外被吐出")
	}
	tail, calls := p.flush()
	if len(calls) != 0 {
		t.Fatalf("残缺载荷不该被解析成调用: %#v", calls)
	}
	if tail != `我先看一下目录结构` {
		t.Fatalf("误发标记后的正文应补发，实际 %q", tail)
	}
}

// TestTraeFnParserFlushDropsTruncatedCall 真正被截断的工具调用（载荷以 [ 或
// { 开头）**不能**当正文吐给用户。
//
// 这是线上回归：模型用 write 工具写一个大文件，参数字节数超过 hold 上限，
// 旧实现把这段 JSON 当正文放行，客户端显示成
// 「已调用工具 + 一大坨 {"name":"write","parameters":{...}}」。
func TestTraeFnParserFlushDropsTruncatedCall(t *testing.T) {
	for _, payload := range []string{
		`[{"name":"get_wea`,
		`{"name":"write","parameters":{"file_path":"D:\\a.py","content":"#!/usr/bin/env py`,
	} {
		p := newTraeFnParser()
		p.feed(traeFnBegin + payload)
		tail, calls := p.flush()
		if len(calls) != 0 {
			t.Fatalf("残缺载荷不该被解析成调用: %#v", calls)
		}
		if tail != "" {
			t.Fatalf("截断的工具调用绝不能当正文吐出，实际 %q", tail)
		}
	}
}

// TestTraeFnParserFeedReleasesOverlongHold 模型误发未闭合标记时，
// 后续正文不能被无限 hold（观感同样是「回答停了」）。
// 但**仅当载荷不像调用时**才放行——见 TestTraeFnParserKeepsLargeCallPayload。
func TestTraeFnParserFeedReleasesOverlongHold(t *testing.T) {
	p := newTraeFnParser()
	p.feed(traeFnBegin)
	big := strings.Repeat("x", traeFnMaxHold+16)
	out, calls := p.feed(big)
	if len(calls) != 0 {
		t.Fatalf("不该产出调用: %#v", calls)
	}
	if out != big {
		t.Fatalf("超长且不像调用的载荷应整段放行，实际长度 %d", len(out))
	}
	if p.inCall {
		t.Fatal("放行后应复位 inCall")
	}
}

// TestTraeFnParserKeepsLargeCallPayload 大参数的工具调用必须完整解析。
//
// 线上回归：write 工具写整份源码（远超 traeFnMaxHold），旧实现按字节数
// 把载荷当正文吐了出去。判定必须结合「像不像调用」，只看长度就会误放行。
func TestTraeFnParserKeepsLargeCallPayload(t *testing.T) {
	body := strings.Repeat("x", 64<<10) // 64KB 源码
	payload := `[{"name":"write","parameters":{"file_path":"D:\\burp_mcp_bridge.py","content":"` + body + `"}}]`
	p := newTraeFnParser()
	var text strings.Builder
	// 分 4KB 片喂入，模拟上游分帧。
	raw := traeFnBegin + payload + traeFnEnd
	for i := 0; i < len(raw); i += 4096 {
		end := i + 4096
		if end > len(raw) {
			end = len(raw)
		}
		o, cs := p.feed(raw[i:end])
		text.WriteString(o)
		if len(cs) > 0 {
			if len(cs) != 1 || cs[0].Function.Name != "write" {
				t.Fatalf("bad calls: %#v", cs)
			}
			if len(cs[0].Function.Arguments) < 64<<10 {
				t.Fatalf("参数被截断: %d 字节", len(cs[0].Function.Arguments))
			}
			if text.String() != "" {
				t.Fatalf("工具调用不该泄漏进正文: %q", text.String())
			}
			return
		}
	}
	t.Fatal("大参数工具调用未被解析出来")
}

// TestParseTraeCallsStripsCDATA 模型把载荷包进 CDATA 时必须能解析，
// 否则客户端显示成「已调用工具 + <![CDATA[]]>」。
func TestParseTraeCallsStripsCDATA(t *testing.T) {
	for _, payload := range []string{
		`<![CDATA[[{"name":"ops_bash","parameters":{"cmd":"ls"}}]]]>`,
		`<![CDATA[{"name":"ops_bash","parameters":{"cmd":"ls"}}]]>`,
	} {
		cs := parseTraeCalls(payload)
		if len(cs) != 1 || cs[0].Function.Name != "ops_bash" {
			t.Fatalf("CDATA 载荷未解析: %q -> %#v", payload, cs)
		}
		if cs[0].Function.Arguments != `{"cmd":"ls"}` {
			t.Fatalf("参数 = %q", cs[0].Function.Arguments)
		}
	}
}

// TestParseTraeCallsUnquotesStringifiedArgs 上游把 parameters 写成字符串化
// JSON 时要解包，否则客户端拿到的是带转义的字符串而不是对象。
func TestParseTraeCallsUnquotesStringifiedArgs(t *testing.T) {
	cs := parseTraeCalls(`[{"name":"f","parameters":"{\"a\":1}"}]`)
	if len(cs) != 1 {
		t.Fatalf("未解析: %#v", cs)
	}
	if cs[0].Function.Arguments != `{"a":1}` {
		t.Fatalf("参数未解包: %q", cs[0].Function.Arguments)
	}
}

// TestParseTraeCallsDropsInvalidArgs 参数不是合法 JSON 时整条调用作废，
// 而不是把脏参数丢给客户端（那会渲染成占位符）。
func TestParseTraeCallsDropsInvalidArgs(t *testing.T) {
	cs := parseTraeCalls(`[{"name":"f","parameters":not-json-at-all}]`)
	if len(cs) != 0 {
		t.Fatalf("非法参数应丢弃该调用: %#v", cs)
	}
}

func TestToolChoiceIsNone(t *testing.T) {
	if !toolChoiceIsNone(json.RawMessage("\"none\"")) {
		t.Fatal("none not detected")
	}
	if toolChoiceIsNone(json.RawMessage("\"auto\"")) {
		t.Fatal("auto misdetected")
	}
	if toolChoiceIsNone(nil) {
		t.Fatal("nil misdetected")
	}
}

func TestTraeToolPromptContainsSchema(t *testing.T) {
	raw := json.RawMessage("[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"查天气\",\"parameters\":{\"type\":\"object\"}}}]")
	p := traeToolPrompt(raw)
	if !strings.Contains(p, "get_weather") || !strings.Contains(p, "<|FunctionCallBegin|>") {
		t.Fatalf("bad prompt: %s", p)
	}
}

func TestBuildTraePayloadInjectsToolPrompt(t *testing.T) {
	tools, _ := json.Marshal([]any{map[string]any{"type": "function", "function": map[string]any{"name": "get_weather", "description": "查天气", "parameters": map[string]any{"type": "object"}}}})
	req := &OAIRequest{Messages: []OAIMessage{{Role: "user", Content: json.RawMessage("\"北京天气\"")}}, Tools: tools, ToolChoice: json.RawMessage("\"auto\"")}
	raw := buildTraePayload(req, "seed_m8")
	s := string(raw)
	for _, want := range []string{"get_weather", "<|FunctionCallBegin|>", "<|FunctionCallResult|>"} {
		if !strings.Contains(s, want) {
			t.Fatalf("payload missing %s:\n%s", want, s)
		}
	}
	if strings.Contains(s, "tools") {
		t.Fatalf("raw tools field leaked: %s", s)
	}
}

func TestBuildTraePayloadNoToolsWhenNone(t *testing.T) {
	tools, _ := json.Marshal([]any{map[string]any{"type": "function", "function": map[string]any{"name": "get_weather", "parameters": map[string]any{"type": "object"}}}})
	req := &OAIRequest{Messages: []OAIMessage{{Role: "user", Content: json.RawMessage("\"hi\"")}}, Tools: tools, ToolChoice: json.RawMessage("\"none\"")}
	raw := buildTraePayload(req, "seed_m8")
	if strings.Contains(string(raw), "<|FunctionCallBegin|>") {
		t.Fatalf("tool prompt injected despite tool_choice=none: %s", raw)
	}
}

func TestRenderTraeCallsAndResult(t *testing.T) {
	calls := []oaiToolCall{{ID: "call_1", Type: "function", Function: oaiToolCallFnSpec{Name: "get_weather", Arguments: "{\"city\":\"北京\"}"}}}
	r := renderTraeCalls(calls)
	if !strings.Contains(r, "<|FunctionCallBegin|>") || !strings.Contains(r, "get_weather") || !strings.Contains(r, "city") {
		t.Fatalf("bad calls: %s", r)
	}
	res := renderTraeResult("call_1", "晴 20C")
	if !strings.Contains(res, "<|FunctionCallResult|>") || !strings.Contains(res, "晴 20C") {
		t.Fatalf("bad result: %s", res)
	}
}

func TestBuildTraePayloadToolResultMarker(t *testing.T) {
	req := &OAIRequest{Messages: []OAIMessage{
		{Role: "user", Content: json.RawMessage("\"北京天气\"")},
		{Role: "assistant", ToolCalls: json.RawMessage("[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{}\"}}]")},
		{Role: "tool", ToolCallID: "call_1", Content: json.RawMessage("\"晴 20C\"")},
	}}
	s := string(buildTraePayload(req, "seed_m8"))
	if !strings.Contains(s, "<|FunctionCallResult|>") {
		t.Fatalf("tool result not rendered: %s", s)
	}
}
