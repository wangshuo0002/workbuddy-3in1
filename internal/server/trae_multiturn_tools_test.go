package server

import (
	"encoding/json"
	"strings"
	"testing"
)

// 多轮工具链回归（2026-11-01 用户反馈「话没说完就停了」）。
//
// 线上现象：用户让模型「配置 Burp MCP，写一个 python 桥接脚本」。模型回了
// 三句就结束（用时 2 分 59 秒，正文极短）。根因不是流式问题，而是提示词：
//
//	只要历史里出现**任意一条** role:"tool"，buildTraePayload / buildSoloPayload
//	就注入「严禁再次输出任何 <|FunctionCallBegin|> 标记」。
//
// agent 场景是多轮工具链（读文件 → 写文件 → 跑测试 → 再改）。第 1 次工具调用
// 之后，模型在后续每一轮都被永久禁言调工具，任务做不下去只能草草收尾。
//
// 正确语义：禁止**重复调用同一个**工具，而不是禁止所有工具。

const multiTurnTools = `"tools":[{"type":"function","function":{"name":"ops_bash","parameters":{"type":"object"}}},` +
	`{"type":"function","function":{"name":"write","parameters":{"type":"object"}}}]`

// TestTraePayloadAllowsFurtherToolsAfterToolResult 已有工具结果时，
// 仍必须允许调用**其它**工具（多轮链路不能断）。
func TestTraePayloadAllowsFurtherToolsAfterToolResult(t *testing.T) {
	raw := `{"messages":[` +
		`{"role":"user","content":"配置 Burp MCP 并写桥接脚本"},` +
		`{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"ops_bash","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":"done"}` +
		`],` + multiTurnTools + `}`

	var req OAIRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}
	body := string(buildTraePayload(&req, "seed_m8"))

	// 必须还保留工具说明（有工具可用才谈得上继续调）。
	if !strings.Contains(body, traeFnBegin) {
		t.Fatalf("工具说明丢失，模型无法再调工具:\n%s", body)
	}
	if !strings.Contains(body, "ops_bash") || !strings.Contains(body, "write") {
		t.Fatalf("工具清单丢失:\n%s", body)
	}
	// 不能出现绝对禁令。
	if strings.Contains(body, "严禁再次输出") {
		t.Fatalf("出现「严禁再次输出」绝对禁令，会切断多轮工具链:\n%s", body)
	}
	// 防重复调用的指令仍然要在（抑制调用环）。
	if !strings.Contains(body, "不要重复调用") {
		t.Fatalf("缺少防重复调用指令，模型会陷入调用环:\n%s", body)
	}
}

// TestSoloPayloadAllowsFurtherToolsAfterToolResult SOLO 链同理。
func TestSoloPayloadAllowsFurtherToolsAfterToolResult(t *testing.T) {
	raw := `{"messages":[` +
		`{"role":"user","content":"配置 Burp MCP 并写桥接脚本"},` +
		`{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"ops_bash","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":"done"}` +
		`],` + multiTurnTools + `}`

	var req OAIRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}
	body := string(buildSoloPayload(&req, "glm-5.3-flashx"))

	if strings.Contains(body, "严禁再次输出") {
		t.Fatalf("SOLO 链仍带绝对禁令，会切断多轮工具链:\n%s", body)
	}
	if !strings.Contains(body, "不要重复调用") {
		t.Fatalf("SOLO 链缺少防重复调用指令:\n%s", body)
	}
}
