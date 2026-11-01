package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strings"
)

// 工具调用（function calling）跨协议适配。
//
// 背景（2026-10-07 三家反代实测）：
//
//	WorkBuddy(CodeBuddy) 上游原生支持 OpenAI tools，直接透传，无需本文件。
//	ZCode 上游是 Anthropic Messages：tools 必须是 {name,description,input_schema}，
//	      tool_choice 必须转成字典（{type:auto} / {type:any} / {type:tool,name}）。
//	      原样下发 OpenAI 的 {type:function,function:{...}} 会被上游 400 拒绝：
//	      "body.tools.0.name: Field required; body.tool_choice: Input should be a valid dictionary"。
//	Trae 上游 /api/ide/v1/chat 会静默忽略 tools 字段（既不报错也不调用，直接按
//	      自己的知识编造答案）。它的工具调用靠提示词注入 + 内联标记
//	      <|FunctionCallBegin|>[{"name":..,"parameters":{..}}]<|FunctionCallEnd|>，
//	      网关需要 ① 请求侧把工具定义注入提示词；② 响应侧把标记翻译成 tool_calls。

// ---- OpenAI 工具协议类型 ----

// oaiTool 是 OpenAI tools[] 元素。
type oaiTool struct {
	Type     string        `json:"type"`
	Function oaiToolFnSpec `json:"function"`
}

// oaiToolFnSpec 是 tools[].function。
type oaiToolFnSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// oaiToolCall 是响应 message.tool_calls[] 元素。
type oaiToolCall struct {
	Index    *int              `json:"index,omitempty"`
	ID       string            `json:"id"`
	Type     string            `json:"type"`
	Function oaiToolCallFnSpec `json:"function"`
}

// oaiToolCallFnSpec 是 tool_calls[].function。
type oaiToolCallFnSpec struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// marshalNoHTML 序列化时关闭 < > & 的 HTML 转义。
// Trae 的工具标记 <|FunctionCallBegin|> 若被转成 \u003c...\u003e，虽然 JSON 语义等价，
// 但注入提示词后模型看到的是转义形态，实测会降低标记遵循率，故这里显式关闭。
func marshalNoHTML(v any) []byte {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil
	}
	b := buf.Bytes()
	if n := len(b); n > 0 && b[n-1] == '\n' {
		b = b[:n-1]
	}
	return b
}

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return b
}

// asInt 把 JSON 解出来的数字（float64 / json.Number）安全转 int。
func asInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	}
	return 0
}

// parseOAITools 解析 OpenAI tools（[{type:function,function:{...}}]）。
// 非 function 类型（如内置 web_search）与无名工具一律丢弃：三家上游都不认。
func parseOAITools(raw json.RawMessage) []oaiTool {
	if len(raw) == 0 {
		return nil
	}
	var arr []oaiTool
	if json.Unmarshal(raw, &arr) != nil {
		return nil
	}
	out := make([]oaiTool, 0, len(arr))
	for _, t := range arr {
		if t.Type == "function" && t.Function.Name != "" {
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// parseOAIToolCalls 解析请求消息里的 assistant tool_calls。
func parseOAIToolCalls(raw json.RawMessage) []oaiToolCall {
	if len(raw) == 0 {
		return nil
	}
	var arr []oaiToolCall
	if json.Unmarshal(raw, &arr) != nil {
		return nil
	}
	return arr
}

// ---- ZCode：OpenAI tools → Anthropic tools ----

// anthToolsFromOAI 把工具定义转成 Anthropic 形态。
//
// 已是 Anthropic 形态（顶层没有 function、直接带 name / input_schema）的条目
// 原样保留，兼容调用方直接下发 Anthropic 风格 tools 的老行为（实测该形态上游 200）。
func anthToolsFromOAI(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var arr []map[string]any
	if json.Unmarshal(raw, &arr) != nil {
		return nil
	}
	out := make([]map[string]any, 0, len(arr))
	for _, t := range arr {
		fn, hasFn := t["function"].(map[string]any)
		if !hasFn {
			if _, ok := t["name"]; ok {
				out = append(out, t)
			}
			continue
		}
		name, _ := fn["name"].(string)
		if name == "" {
			continue
		}
		nt := map[string]any{"name": name}
		if v, ok := fn["description"]; ok {
			nt["description"] = v
		}
		if v, ok := fn["parameters"]; ok {
			nt["input_schema"] = v
		} else {
			nt["input_schema"] = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, nt)
	}
	if len(out) == 0 {
		return nil
	}
	return mustJSON(out)
}

// toolChoiceIsNone 判断 OpenAI tool_choice 是否为 none（等于禁用工具）。
func toolChoiceIsNone(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return strings.EqualFold(strings.TrimSpace(s), "none")
	}
	return false
}

// anthToolChoiceFromOAI 把 OpenAI tool_choice 转成 Anthropic 形态。
// 返回 nil 表示不下发该字段（交给上游默认 auto）。
func anthToolChoiceFromOAI(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch strings.ToLower(strings.TrimSpace(s)) {
		case "auto":
			return json.RawMessage(`{"type":"auto"}`)
		case "required", "any":
			return json.RawMessage(`{"type":"any"}`)
		default:
			return nil
		}
	}
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return nil
	}
	if t, _ := m["type"].(string); strings.EqualFold(t, "function") {
		if fn, ok := m["function"].(map[string]any); ok {
			if name, _ := fn["name"].(string); name != "" {
				return mustJSON(map[string]any{"type": "tool", "name": name})
			}
		}
		return nil
	}
	if t, _ := m["type"].(string); t == "auto" || t == "any" || t == "tool" {
		return raw
	}
	return nil
}

// ---- Trae：提示词注入 + 内联标记解析 ----

const (
	traeFnBegin = "<|FunctionCallBegin|>"
	traeFnEnd   = "<|FunctionCallEnd|>"
	// traeFnResult 是回灌给模型的工具结果包装（Trae chat_history 只有 user/assistant）。
	traeFnResult = "<|FunctionCallResult|>"

	// DSML 是 DeepSeek 系模型的**原生**工具调用格式。
	//
	// 提示词里约定的是 <|FunctionCallBegin|>，但上游模型不一定听话：SOLO 链的
	// deepseek-v4.1-flash 实测直接吐自己的 DSML 标记：
	//
	//	<|DSML|tool_calls>
	//	<|DSML|invoke name="ops_bash">
	//	<|DSML|parameter name="command" string="true">ls -a</|DSML|parameter>
	//	</|DSML|invoke>
	//	</|DSML|tool_calls>
	//
	// 线上实测还出现过双竖线变体（<||DSML||tool_calls>）。不认这两种写法，
	// 整段标记就会当正文吐给用户——现象就是「回复是一大坨 XML 样的乱码」。
	dsmlToolCallsBegin  = "<|DSML|tool_calls>"
	dsmlToolCallsEnd    = "</|DSML|tool_calls>"
	dsmlToolCallsBegin2 = "<||DSML||tool_calls>"
	dsmlToolCallsEnd2   = "</||DSML||tool_calls>"
)

// traeCallFormat 一种「内联工具调用标记」格式。
type traeCallFormat struct {
	begin, end string
	parse      func(string) []oaiToolCall
}

// traeCallFormats 所有支持的调用格式，按识别顺序排列。
//
// 多种格式并存是必要的：模型的原生格式由上游决定，网关只能尽量兼容。
// 新格式往这里加即可，解析状态机与 hold-back 逻辑无需改动。
var traeCallFormats = []traeCallFormat{
	{begin: traeFnBegin, end: traeFnEnd, parse: parseTraeCalls},
	{begin: dsmlToolCallsBegin, end: dsmlToolCallsEnd, parse: parseDSMLCalls},
	{begin: dsmlToolCallsBegin2, end: dsmlToolCallsEnd2, parse: parseDSMLCalls},
}

// traeCallKeep hold-back 需要保留的最大字节数（最长 begin 长度 - 1）。
func traeCallKeep() int {
	keep := 0
	for _, f := range traeCallFormats {
		if n := len(f.begin) - 1; n > keep {
			keep = n
		}
	}
	return keep
}

// traeNoRepeatDirective 针对「刚执行完工具」的收尾指令。
//
// 历史版本写的是「严禁再次输出 <|FunctionCallBegin|>」——只要历史里出现过
// **任意一条** role:"tool"，之后每一轮都会被这条禁令约束，模型只能草草收尾。
// agent 场景是多轮工具链（读文件 → 改文件 → 跑测试 → 再改），第 1 次调用工具
// 之后就被永久禁言，表现就是「话没说完就停 / 回答只有一两句」。
//
// 正确语义是「别重复调用同一个工具」，而不是「别再调工具」：与 traeToolPrompt
// 里「不要重复调用同一工具」的措辞一致，也给后续轮次留出继续调其它工具的空间。
const traeNoRepeatDirective = "\n\n注意：上面这些工具的调用结果已在对话中给出，不要重复调用**同一个**工具。若任务还需要其它工具，可以继续按格式调用；若已经可以作答，请直接给出最终的自然语言回答。"

// traeToolPrompt 生成注入 Trae 上下文的工具说明。
//
// 上游把 tools 字段整个忽略（实测带 tools 仍回自然语言编造），唯一可行的通路
// 是把「有哪些工具 + 用什么格式调用」写进提示词。格式与官方 IDE 客户端一致。
func traeToolPrompt(raw json.RawMessage) string {
	tools := parseOAITools(raw)
	if len(tools) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("【工具调用能力】你可以调用下列工具来完成任务。\n\n可用工具：\n")
	for _, tl := range tools {
		b.WriteString("- ")
		b.WriteString(tl.Function.Name)
		if tl.Function.Description != "" {
			b.WriteString("：")
			b.WriteString(tl.Function.Description)
		}
		b.WriteString("\n")
		if len(tl.Function.Parameters) > 0 {
			b.WriteString("  参数 JSON Schema：")
			b.WriteString(string(tl.Function.Parameters))
			b.WriteString("\n")
		}
	}
	b.WriteString("\n需要调用工具时，你的整条回复必须且只能包含如下格式，不要有任何解释或多余文字：\n")
	b.WriteString(traeFnBegin)
	b.WriteString(`[{"name":"工具名","parameters":{参数对象}}]`)
	b.WriteString(traeFnEnd)
	b.WriteString("\n一次可调用多个工具，用 JSON 数组表示。不需要调用工具时，用自然语言正常回答。\n")
	b.WriteString("工具执行结果会以如下格式回传给你，收到后请据此给出最终回答，不要重复调用同一工具：\n")
	b.WriteString(traeFnResult)
	b.WriteString(`[{"name":"工具名","result":结果}]`)
	b.WriteString(traeFnEnd)
	return b.String()
}

// renderTraeCalls 把 assistant 历史里的 tool_calls 还原成上游认识的标记文本，
// 否则模型看不到自己上一轮调用过什么，多轮工具链会断。
func renderTraeCalls(calls []oaiToolCall) string {
	type rc struct {
		Name       string          `json:"name"`
		Parameters json.RawMessage `json:"parameters"`
	}
	arr := make([]rc, 0, len(calls))
	for _, c := range calls {
		args := strings.TrimSpace(c.Function.Arguments)
		if args == "" {
			args = "{}"
		}
		arr = append(arr, rc{Name: c.Function.Name, Parameters: json.RawMessage(args)})
	}
	return traeFnBegin + string(mustJSON(arr)) + traeFnEnd
}

// hasToolResult 判断消息历史里是否已有工具执行结果。
func hasToolResult(msgs []OAIMessage) bool {
	for _, m := range msgs {
		if strings.EqualFold(m.Role, "tool") {
			return true
		}
	}
	return false
}

// renderTraeResult 把 OpenAI role:"tool" 的工具结果渲染成 Trae 认识的结果标记。
// 上游 chat_history 没有 tool 角色，直接塞裸文本模型认不出这是工具返回值，
// 会重复调用同一工具；加标记并在提示词里说明格式后，模型能正确续答。
func renderTraeResult(toolCallID, text string) string {
	payload := map[string]any{"result": text}
	if toolCallID != "" {
		payload["id"] = toolCallID
	}
	return traeFnResult + string(mustJSON(payload)) + traeFnEnd
}

// traeFnParser 从 Trae 的正文流里剥离 <|FunctionCallBegin|>...<|FunctionCallEnd|>。
//
// 标记可能跨 output 帧（上游把长回复切成多个 response 帧），因此按 hold-back
// 状态机处理：只吐出「不可能再构成标记前缀」的正文，把可能残缺的尾部留住。
type traeFnParser struct {
	pending string
	inCall  bool
	// format 当前正在解析的调用格式（inCall 为真时有效）。
	format *traeCallFormat
	calls  []oaiToolCall
}

// traeFnMaxHold 单个工具标记内最多缓存多少字节。
//
// 模型并非总按约定输出闭合标记：实测有「只发了 <|FunctionCallBegin|> 就断流」
// 和「复述提示词里的标记」两种情况。此时 inCall 会一直为真，后续正文全部被
// 状态机当成调用载荷吞掉——客户端只看到标记之前那半句，观感就是「回答没写完
// 就停了」。超过上限且**载荷不像工具调用**时即判定为误发标记，当正文放行。
//
// 判定必须带「像不像调用」这一层：工具参数完全可能很大（write 一个文件就是
// 几十 KB 源码），若仅凭字节数放行，这些参数会被当成正文吐给用户
// （现象：回复里出现一大坨 {"name":"write","parameters":{...}）。
const traeFnMaxHold = 8 << 10

// traeFnHardLimit 单个工具标记的绝对上限（超过即放弃该调用，丢弃载荷）。
//
// 防御上游永远不发结束标记：载荷可能无限增长，不能无上限地留在内存里。
// 取 4MB —— 远超任何合理工具参数，撞到它基本可以断定不是合法调用。
const traeFnHardLimit = 4 << 20

// traeFnLooksLikeCall 判断标记内的载荷是否「像」工具调用载荷。
//
// 工具调用是 JSON 数组/对象或 DSML invoke，正文回答不会以这些形态开头。
// 用于在标记残缺时区分「模型误发了标记」与「真的在调工具但被截断」——
// 前者当正文放出，后者绝不能当正文（否则用户会看到半截 JSON/XML）。
func traeFnLooksLikeCall(payload string) bool {
	s := strings.TrimSpace(payload)
	return strings.HasPrefix(s, "[") || strings.HasPrefix(s, "{") ||
		strings.Contains(s, "DSML") && strings.Contains(s, "invoke")
}

func newTraeFnParser() *traeFnParser { return &traeFnParser{} }

// indexAnyFormat 在 s 中找出最早出现的某个调用格式起始标记。
func indexAnyFormat(s string) (int, *traeCallFormat) {
	best, bestF := -1, (*traeCallFormat)(nil)
	for i := range traeCallFormats {
		f := &traeCallFormats[i]
		j := strings.Index(s, f.begin)
		if j < 0 {
			continue
		}
		if best < 0 || j < best {
			best, bestF = j, f
		}
	}
	return best, bestF
}

// feed 吃进一段正文，返回可安全吐出的正文与本次新完成的工具调用。
func (p *traeFnParser) feed(chunk string) (string, []oaiToolCall) {
	if p == nil || chunk == "" {
		return "", nil
	}
	p.pending += chunk
	var out strings.Builder
	var fresh []oaiToolCall
	for {
		if p.inCall {
			f := p.format
			if f == nil {
				p.inCall = false
				continue
			}
			i := strings.Index(p.pending, f.end)
			if i < 0 {
				// 未闭合。三种可能：真在调工具（参数很大，继续等）、
				// 模型误发标记（当正文放行）、恶意/异常超长（丢弃）。
				switch {
				case len(p.pending) > traeFnHardLimit:
					// 超过硬上限：放弃该调用，载荷丢弃（不吐给用户）。
					p.pending = ""
					p.inCall = false
					p.format = nil
				case len(p.pending) > traeFnMaxHold && !traeFnLooksLikeCall(p.pending):
					// 明显不是调用载荷 → 误发标记，当正文放行。
					out.WriteString(p.pending)
					p.pending = ""
					p.inCall = false
					p.format = nil
				}
				break
			}
			payload := p.pending[:i]
			p.pending = p.pending[i+len(f.end):]
			p.inCall = false
			p.format = nil
			if cs := f.parse(payload); len(cs) > 0 {
				fresh = append(fresh, cs...)
				p.calls = append(p.calls, cs...)
			}
			continue
		}
		j, f := indexAnyFormat(p.pending)
		if j >= 0 && f != nil {
			out.WriteString(p.pending[:j])
			p.pending = p.pending[j+len(f.begin):]
			p.inCall = true
			p.format = f
			continue
		}
		keep := traeCallKeep()
		if len(p.pending) > keep {
			// 切点必须落在 UTF-8 字符边界（见 utf8SafeCut）：正文多为中文，
			// 按字节硬切会把一个汉字劈成两半，前半段非法字节经 json.Marshal
			// 变成 \ufffd，客户端就是一片乱码。
			cut := utf8SafeCut(p.pending, len(p.pending)-keep)
			out.WriteString(p.pending[:cut])
			p.pending = p.pending[cut:]
		}
		break
	}
	return out.String(), fresh
}

// flush 收流：把仍被 hold 住的正文吐出；未闭合的调用尽力解析。
func (p *traeFnParser) flush() (string, []oaiToolCall) {
	if p == nil {
		return "", nil
	}
	if p.inCall {
		payload := p.pending
		f := p.format
		p.pending = ""
		p.inCall = false
		p.format = nil
		if f != nil {
			if cs := f.parse(payload); len(cs) > 0 {
				p.calls = append(p.calls, cs...)
				return "", cs
			}
		}
		// 解析失败：只有当载荷**不像**工具调用时才是「模型误发标记」，当正文
		// 补发。若像调用（JSON/DSML 开头），说明是中途被上游掐断的工具调用——
		// 绝不能把半截 JSON/XML 当正文吐给用户。
		if traeFnLooksLikeCall(payload) {
			return "", nil
		}
		return payload, nil
	}
	t := p.pending
	p.pending = ""
	return t, nil
}

// dsmlTag 匹配 <|DSML|xxx>、</|DSML|xxx>、<||DSML||xxx>、</||DSML||xxx>。
// 单双竖线两种写法都要认（线上两种都出现过）。第 2 组是标签名，
// 第 3 组是自闭合/属性区。
var dsmlTagRe = regexp.MustCompile(`<(/?)\s*\|{1,2}\s*DSML\s*\|{1,2}\s*([A-Za-z_][A-Za-z0-9_]*)([^>]*)>`)

// dsmlTag 一次标签匹配的解析结果。
type dsmlTag struct {
	name     string // 标签名（invoke / parameter / tool_calls …）
	closing  bool   // 是否 </...> 形式
	attrs    string // 属性区原文
	tagStart int    // 标签 '<' 的下标
	tagEnd   int    // 开标签 '>' 之后的下标
	textEnd  int    // 该标签「内容」的结束下标
}

// dsmlTags 扫描出全部标签，并计算每个标签的内容区间。
//
// 内容区间 = 本标签结束到「它自己的闭合标签」起点；没有闭合标签时取下一个
// 同级/任意标签起点（容错），再不行取整段末尾。不能简单地取「下一个标签」，
// 因为 parameter 的内容后面紧跟的正是它自己的闭合标签，那样会取到空串。
func dsmlTags(s string) []dsmlTag {
	locs := dsmlTagRe.FindAllStringSubmatchIndex(s, -1)
	out := make([]dsmlTag, 0, len(locs))
	for _, loc := range locs {
		name := ""
		if loc[4] >= 0 {
			name = s[loc[4]:loc[5]]
		}
		out = append(out, dsmlTag{
			name:     name,
			closing:  loc[2] >= 0 && loc[3] > loc[2] && s[loc[2]:loc[3]] == "/",
			attrs:    s[loc[6]:loc[7]],
			tagStart: loc[0],
			tagEnd:   loc[1],
		})
	}
	for i := range out {
		if out[i].closing {
			out[i].textEnd = out[i].tagEnd
			continue
		}
		// 找同名闭合标签；找不到就退到下一个标签起点，再不行取末尾。
		end := -1
		for j := i + 1; j < len(out); j++ {
			if out[j].closing && out[j].name == out[i].name {
				end = out[j].tagStart
				break
			}
		}
		if end < 0 {
			for j := i + 1; j < len(out); j++ {
				end = out[j].tagStart
				break
			}
		}
		if end < 0 {
			end = len(s)
		}
		out[i].textEnd = end
	}
	return out
}

// parseDSMLCalls 把 DSML 的 tool_calls 块解析成标准 tool_calls。
//
// 输入是 begin/end 之间的内容（即若干 <|DSML|invoke>…</|DSML|invoke>），
// 也容忍整块（含 tool_calls 外层标签）。
//
// DSML 用「标签 + 属性」表达结构，与 JSON 完全无关，因此不能复用
// parseTraeCalls，必须按标签逐个取参数：
//
//	<|DSML|invoke name="ops_bash">
//	  <|DSML|parameter name="command" string="true">ls -a</|DSML|parameter>
//	  <|DSML|parameter name="timeoutMs" string="false">60000</|DSML|parameter>
//	</|DSML|invoke>
func parseDSMLCalls(payload string) []oaiToolCall {
	tags := dsmlTags(payload)
	var out []oaiToolCall
	for _, tg := range tags {
		if tg.closing || tg.name != "invoke" {
			continue
		}
		name := dsmlAttr(tg.attrs, "name")
		if name == "" {
			continue
		}
		body := payload[tg.tagEnd:tg.textEnd]
		params := parseDSMLParams(body)
		idx := len(out)
		out = append(out, oaiToolCall{
			Index: &idx,
			ID:    "call_" + newID(""),
			Type:  "function",
			Function: oaiToolCallFnSpec{
				Name:      name,
				Arguments: params,
			},
		})
	}
	return out
}

// parseDSMLParams 从 invoke 体内取全部 parameter，拼成 JSON 对象字符串。
//
// 类型按 string 属性决定：string="true" 取值原样作字符串，否则尝试按
// JSON 字面量（数字/布尔/对象/数组）解析，失败则仍作字符串——宁可退化成
// 字符串也不能丢参数，客户端 schema 校验会纠正类型。
func parseDSMLParams(body string) string {
	obj := map[string]any{}
	for _, tg := range dsmlTags(body) {
		if tg.closing || tg.name != "parameter" {
			continue
		}
		name := dsmlAttr(tg.attrs, "name")
		if name == "" {
			continue
		}
		raw := strings.TrimSpace(body[tg.tagEnd:tg.textEnd])
		if strings.Contains(tg.attrs, `string="true"`) || strings.Contains(tg.attrs, `string='true'`) {
			obj[name] = raw
			continue
		}
		var v any
		if json.Unmarshal([]byte(raw), &v) == nil {
			obj[name] = v
		} else {
			obj[name] = raw
		}
	}
	b := marshalNoHTML(obj)
	if len(b) == 0 {
		return "{}"
	}
	return string(b)
}

// dsmlAttr 从标签属性串里取 attr="value"（也接受单引号）。
func dsmlAttr(attrs, name string) string {
	for _, quote := range []byte{'"', '\''} {
		needle := name + "=" + string(quote)
		if i := strings.Index(attrs, needle); i >= 0 {
			rest := attrs[i+len(needle):]
			if j := strings.IndexByte(rest, quote); j >= 0 {
				return rest[:j]
			}
		}
	}
	return ""
}

//
// 模型偶尔会把调用载荷按 XML 习惯包进 CDATA（提示词里出现过 <|...|> 这类
// 标记，模型容易类比到 XML 写法）。不剥掉的话 json.Unmarshal 必然失败，
// 该工具调用会被丢弃，客户端侧表现为「已调用工具」但参数是 `<![CDATA[]]>`
// 这类占位内容。
func traeStripCDATA(s string) string {
	s = strings.TrimSpace(s)
	for {
		if !strings.HasPrefix(s, "<![CDATA[") {
			return s
		}
		s = strings.TrimPrefix(s, "<![CDATA[")
		s = strings.TrimSuffix(strings.TrimSpace(s), "]]>")
		s = strings.TrimSpace(s)
	}
}

// parseTraeCalls 解析标记内载荷：期望 [{"name":..,"parameters":{..}}]，
// 也容忍单对象 {"name":..,"parameters":{..}} 与 name/arguments 混写。
func parseTraeCalls(payload string) []oaiToolCall {
	payload = traeStripCDATA(payload)
	if payload == "" {
		return nil
	}
	type rawCall struct {
		Name       string          `json:"name"`
		Parameters json.RawMessage `json:"parameters"`
		Arguments  json.RawMessage `json:"arguments"`
	}
	var arr []rawCall
	if json.Unmarshal([]byte(payload), &arr) != nil {
		var one rawCall
		if json.Unmarshal([]byte(payload), &one) != nil || one.Name == "" {
			return nil
		}
		arr = []rawCall{one}
	}
	out := make([]oaiToolCall, 0, len(arr))
	for i, c := range arr {
		if c.Name == "" {
			continue
		}
		args := "{}"
		switch {
		case len(c.Parameters) > 0:
			args = string(c.Parameters)
		case len(c.Arguments) > 0:
			args = string(c.Arguments)
		}
		// 参数必须是合法 JSON 对象：上游给字符串化的 JSON（"{\"a\":1}"）时
		// 解包一层；仍不是合法 JSON 就让这个调用整条作废——把脏参数发给客户端
		// 会让它渲染成 <![CDATA[]]> 之类的占位符，用户只看到「已调用工具」加
		// 一堆无意义符号，比明确不出调用更让人困惑。
		if unquoted, ok := traeUnquoteJSONObject(args); ok {
			args = unquoted
		} else if !json.Valid([]byte(args)) {
			continue
		}
		idx := i
		out = append(out, oaiToolCall{
			Index: &idx,
			ID:    "call_" + newID(""),
			Type:  "function",
			Function: oaiToolCallFnSpec{
				Name:      c.Name,
				Arguments: args,
			},
		})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// traeUnquoteJSONObject 把「字符串化的 JSON 对象」解包成真正的 JSON 对象文本。
// 只处理能解析成对象的情况；其余返回 false 交给调用方判断。
func traeUnquoteJSONObject(args string) (string, bool) {
	s := strings.TrimSpace(args)
	if !strings.HasPrefix(s, `"`) {
		return "", false
	}
	var inner string
	if json.Unmarshal([]byte(s), &inner) != nil {
		return "", false
	}
	inner = strings.TrimSpace(inner)
	if !strings.HasPrefix(inner, "{") || !json.Valid([]byte(inner)) {
		return "", false
	}
	return inner, true
}

// emitToolCallChunks 把一个 tool_call 列表写成 OpenAI SSE 增量帧。
// 每次带 role（首帧才发 role，由 sentRole 控制）与完整 arguments，
// 语义上等价于 OpenAI 的「一次给出整个 function call」。
//
// index 必须全局递增：工具调用可能分多批到达（模型一轮里先调两个、再调一个），
// 每批都从 0 开始会被客户端当成「同一个调用的增量」而互相覆盖。
func emitToolCallChunks(w io.Writer, flusher http.Flusher, id, model string, created int64, sentRole *bool, calls []oaiToolCall) {
	emitToolCallChunksFrom(w, flusher, id, model, created, sentRole, calls, 0)
}

// emitToolCallChunksFrom 同 emitToolCallChunks，但 index 从 start 开始。
func emitToolCallChunksFrom(w io.Writer, flusher http.Flusher, id, model string, created int64, sentRole *bool, calls []oaiToolCall, start int) {
	for i := range calls {
		idx := start + i
		one := calls[i]
		one.Index = &idx
		writeSSE(w, OAIChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []OAIChoice{{Index: 0, Delta: &OAIDelta{Role: roleOnce(sentRole), ToolCalls: mustJSON([]oaiToolCall{one})}}}})
		if flusher != nil {
			flusher.Flush()
		}
	}
}
