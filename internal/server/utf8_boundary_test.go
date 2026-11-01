package server

import (
	"encoding/json"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"unicode/utf8"
)

// UTF-8 边界回归。
//
// 背景：Trae 的正文经两层 hold-back 状态机（traeFnParser 与 stopMatcher），
// 两者的保留量都按**字节**算（traeFnBegin 是 ASCII，keep 取 len-1；stop 取
// len(stop)-1）。而正文几乎全是中文，于是切点经常落在某个多字节字符中间：
// 前半段是非法 UTF-8，经 json.Marshal 编码后变成 \ufffd，客户端看到的
// 就是一片 "�"（实测："我是小王一跑在���王这台机器上"）。
//
// 修法：任何按字节算出的切点都先经 utf8SafeCut 回退到字符边界——最多多 hold
// 3 个字节，但绝不把字符劈开。本文件把这一点钉成不变量。

// jsonQuote 把文本编码成 JSON 字符串字面量。
func jsonQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// TestUTF8SafeCut 纯函数：切点必须回退到 rune 起始字节。
func TestUTF8SafeCut(t *testing.T) {
	s := "a我" // 1 + 3 字节
	cases := map[int]int{
		-1: 0, 0: 0, 1: 1, 2: 1, 3: 1, 4: 4, 99: 4,
	}
	for in, want := range cases {
		if got := utf8SafeCut(s, in); got != want {
			t.Errorf("utf8SafeCut(%q, %d) = %d, want %d", s, in, got, want)
		}
	}
}

// TestStopMatcherHoldsUTF8Boundary stopMatcher 的 hold-back 不能吐出半个汉字。
func TestStopMatcherHoldsUTF8Boundary(t *testing.T) {
	for _, stops := range [][]string{{"\n\n"}, {"STOP"}, {"\n\n", "STOPSEQ"}} {
		m := newStopMatcher(stops)
		var out strings.Builder
		for _, c := range []string{"我是小王一", "跑在大王这台机器上", "的本地助手"} {
			e, _ := m.feed(c)
			if !utf8.ValidString(e) {
				t.Fatalf("stops=%v 吐出非法 UTF-8: %q", stops, e)
			}
			out.WriteString(e)
		}
		tail := m.flush()
		if !utf8.ValidString(tail) {
			t.Fatalf("stops=%v flush 吐出非法 UTF-8: %q", stops, tail)
		}
		out.WriteString(tail)
		if out.String() != "我是小王一跑在大王这台机器上的本地助手" {
			t.Fatalf("stops=%v 内容丢失: %q", stops, out.String())
		}
	}
}

// TestTraeFnParserHoldsUTF8Boundary traeFnParser 的 hold-back 同理。
func TestTraeFnParserHoldsUTF8Boundary(t *testing.T) {
	p := newTraeFnParser()
	var out strings.Builder
	for _, c := range []string{"我是小王一", "跑在大王这台机器上", "的本地助手，底层是 WorkBuddy"} {
		o, _ := p.feed(c)
		if !utf8.ValidString(o) {
			t.Fatalf("feed 吐出非法 UTF-8: %q", o)
		}
		out.WriteString(o)
	}
	tail, _ := p.flush()
	if !utf8.ValidString(tail) {
		t.Fatalf("flush 吐出非法 UTF-8: %q", tail)
	}
	out.WriteString(tail)
	if out.String() != "我是小王一跑在大王这台机器上的本地助手，底层是 WorkBuddy" {
		t.Fatalf("内容丢失: %q", out.String())
	}
}

// TestStreamTraeUpstreamNoMojibake 端到端：无论上游在两段正文的哪个字符边界
// 切开，客户端收到的 SSE 里都不能出现 \ufffd，且正文必须逐字还原。
func TestStreamTraeUpstreamNoMojibake(t *testing.T) {
	const text = "我是小王一跑在大王这台机器上的本地助手，底层是 WorkBuddy Agent 框架（Claude 系模型驱动），带一整套工具链。"
	for cut := 0; cut <= len(text); cut++ {
		if cut < len(text) && !utf8.RuneStart(text[cut]) {
			continue
		}
		for _, stops := range [][]string{nil, {"\n\n"}, {"STOP"}} {
			raw := strings.Join([]string{
				`event: output`,
				`data: {"response":` + jsonQuote(text[:cut]) + `}`,
				`event: output`,
				`data: {"response":` + jsonQuote(text[cut:]) + `}`,
				`event: done`,
				`data: {}`,
				``,
			}, "\n")
			res := &upstreamResult{streaming: true, stream: io.NopCloser(strings.NewReader(raw))}
			rr := httptest.NewRecorder()
			(&Handler{}).streamTraeUpstream(rr, res, "trae/seed_m8", stops)
			body := rr.Body.String()
			if strings.Contains(body, `\ufffd`) {
				t.Fatalf("cut=%d stops=%v 产出乱码:\n%s", cut, stops, body)
			}
			if got := sseContents(t, body); got != text {
				t.Fatalf("cut=%d stops=%v content=%q", cut, stops, got)
			}
		}
	}
}

// TestTraePipelineNeverEmitsInvalidUTF8 管线不变量：按任意字节步长切分输入
// （模拟上游帧边界不可控），fnParser → stopMatcher 的每一段输出都必须是
// 合法 UTF-8，且拼回来的正文一字不差。
func TestTraePipelineNeverEmitsInvalidUTF8(t *testing.T) {
	const text = "我是小王一的本地助手，底层是 WorkBuddy Agent 框架，带一整套工具链和 MCP 连接。"
	for step := 1; step <= len(text); step++ {
		p := newTraeFnParser()
		m := newStopMatcher([]string{"\n\n"})
		var emitted []string
		add := func(s, from string) {
			if s == "" {
				return
			}
			if !utf8.ValidString(s) {
				t.Fatalf("step=%d %s 吐出非法 UTF-8: %q", step, from, s)
			}
			emitted = append(emitted, s)
		}
		for i := 0; i < len(text); i += step {
			end := i + step
			if end > len(text) {
				end = len(text)
			}
			content, _ := p.feed(text[i:end])
			out, _ := m.feed(content)
			add(out, "matcher")
		}
		content, _ := p.flush()
		out, _ := m.feed(content)
		add(out, "matcher(flush)")
		add(m.flush(), "matcher.flush")
		if got := strings.Join(emitted, ""); got != text {
			t.Fatalf("step=%d 内容丢失/串序: %q", step, got)
		}
	}
}
