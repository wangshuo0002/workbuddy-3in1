package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// jsonUnmarshalStrict 只是 json.Unmarshal 的薄封装，测试里避免到处导入。
func jsonUnmarshalStrict(s string, v any) error { return json.Unmarshal([]byte(s), v) }

// errAfterReader 先吐出 data，随后以 err 结束——模拟上游把 SSE 连接掐断
// （HTTP 分块传输未正常结束 → transport 返回 unexpected EOF）。
type errAfterReader struct {
	data []byte
	err  error
}

func (r *errAfterReader) Read(p []byte) (int, error) {
	if len(r.data) > 0 {
		n := copy(p, r.data)
		r.data = r.data[n:]
		return n, nil
	}
	return 0, r.err
}

func (*errAfterReader) Close() error { return nil }

// sseContents 把 SSE 响应体里所有 delta.content 按顺序拼起来。
// 正文会被上游分成多帧（状态机的 hold-back 与 UTF-8 边界都会切段），
// 直接对整段响应体做子串匹配会误判「内容丢失」。
func sseContents(t *testing.T, body string) string {
	t.Helper()
	var sb strings.Builder
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") || line == "data: [DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := jsonUnmarshalStrict(strings.TrimSpace(strings.TrimPrefix(line, "data:")), &chunk); err != nil {
			t.Fatalf("非法 SSE 帧 %q: %v", line, err)
		}
		for _, c := range chunk.Choices {
			sb.WriteString(c.Delta.Content)
		}
	}
	return sb.String()
}

// TestStreamTraeUpstreamReportsInterruptedAsLength 回归「回答没写完就停」的直接成因：
// 上游既没发 event: done 也没发 event: error，却带着读错误断流时，旧实现照样
// 写出 finish_reason=stop + [DONE]，客户端于是把半截回答当成正常完成，既不会
// 续写也不会重试（面板上账号、用量全是正常的）。现在必须报 length 并置位
// res.interrupted，让上层把这次请求记为流错误。
func TestStreamTraeUpstreamReportsInterruptedAsLength(t *testing.T) {
	raw := strings.Join([]string{
		`event: output`,
		`data: {"response":"工作目录是空的，我来基于 PRD 撰写"}`,
		``,
	}, "\n")
	res := &upstreamResult{streaming: true, stream: &errAfterReader{
		data: []byte(raw),
		err:  errors.New("unexpected EOF"),
	}}
	rr := httptest.NewRecorder()
	(&Handler{}).streamTraeUpstream(rr, res, "trae/seed_m8", nil)

	body := rr.Body.String()
	if !res.interrupted {
		t.Fatal("上游断流未被标记为 interrupted")
	}
	if !strings.Contains(body, `"finish_reason":"length"`) {
		t.Fatalf("finish_reason 应为 length（旧实现谎报 stop）: %s", body)
	}
	if strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("断流时不该上报 stop: %s", body)
	}
	// 已经收到的正文必须照常交付，不能被判定逻辑吞掉。
	if got := sseContents(t, body); got != "工作目录是空的，我来基于 PRD 撰写" {
		t.Fatalf("已收到的正文丢失或串序: %q", got)
	}
	if !strings.Contains(body, "[DONE]") {
		t.Fatalf("终止帧缺失: %s", body)
	}
}

// TestStreamTraeUpstreamNormalDoneStaysStop 反例：正常收到 done 就是 stop，
// 不能被中断判定误伤（否则客户端每次都会以为回答被截断）。
func TestStreamTraeUpstreamNormalDoneStaysStop(t *testing.T) {
	raw := strings.Join([]string{
		`event: output`,
		`data: {"response":"完整回答"}`,
		`event: done`,
		`data: {"finish_reason":"stop"}`,
		``,
	}, "\n")
	res := &upstreamResult{streaming: true, stream: io.NopCloser(strings.NewReader(raw))}
	rr := httptest.NewRecorder()
	(&Handler{}).streamTraeUpstream(rr, res, "trae/seed_m8", nil)

	body := rr.Body.String()
	if res.interrupted {
		t.Fatal("正常收尾被误判为中断")
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("正常收尾应为 stop: %s", body)
	}
}

// TestStreamTraeUpstreamKeepsFramesAfterDone 回归「done 提前收流」：
// 上游把长回答分多帧发时，done 可能先到，紧跟其后的 output 仍是正文。
// 旧实现在 done 处直接 break，剩下的正文整段丢失。
func TestStreamTraeUpstreamKeepsFramesAfterDone(t *testing.T) {
	raw := strings.Join([]string{
		`event: output`,
		`data: {"response":"第一段"}`,
		`event: done`,
		`data: {}`,
		`event: output`,
		`data: {"response":"done 之后的第二段"}`,
		``,
	}, "\n")
	res := &upstreamResult{streaming: true, stream: io.NopCloser(strings.NewReader(raw))}
	rr := httptest.NewRecorder()
	(&Handler{}).streamTraeUpstream(rr, res, "trae/seed_m8", nil)

	body := rr.Body.String()
	if got := sseContents(t, body); got != "第一段done 之后的第二段" {
		t.Fatalf("done 之后的正文被丢弃: %q", got)
	}
	if res.interrupted {
		t.Fatal("收到 done 的流不该被判为中断")
	}
}

// TestCollectTraeUpstreamInterruptedFinishLength 非流式聚合同样不能虚报成功。
func TestCollectTraeUpstreamInterruptedFinishLength(t *testing.T) {
	raw := strings.Join([]string{
		`event: output`,
		`data: {"response":"半截回答"}`,
		``,
	}, "\n")
	res := &upstreamResult{streaming: true, stream: &errAfterReader{
		data: []byte(raw),
		err:  errors.New("connection reset by peer"),
	}}
	rr := httptest.NewRecorder()
	(&Handler{}).collectTraeUpstream(rr, res, "trae/seed_m8", nil)

	if !res.interrupted {
		t.Fatal("聚合路径未标记 interrupted")
	}
	if !strings.Contains(rr.Body.String(), `"finish_reason":"length"`) {
		t.Fatalf("聚合响应的 finish_reason 应为 length: %s", rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "半截回答") {
		t.Fatalf("已聚合的正文丢失: %s", rr.Body.String())
	}
}

// TestChanReaderCloseInterruptsRead Close 必须真的让 Read 返回 EOF。
// 旧实现 Close 是空函数：下游提前收流后读循环仍在 <-ch 上永久阻塞，
// 上游 goroutine 也跟着卡在发送上，连接与 goroutine 一起泄漏。
func TestChanReaderCloseInterruptsRead(t *testing.T) {
	ch := make(chan string) // 无人发送：只有 Close 能解开 Read
	r := &chanReader{ch: ch}
	done := make(chan error, 1)
	go func() {
		buf := make([]byte, 8)
		_, err := r.Read(buf)
		done <- err
	}()
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != io.EOF {
			t.Fatalf("Close 后 Read 应返回 io.EOF，实际 %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Close 未解开阻塞的 Read")
	}
}
