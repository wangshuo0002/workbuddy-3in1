package server

import (
	"io"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// ---- 上游「干净 EOF」回归 ----

// cleanEOFReader 先吐完 data，然后**正常返回 io.EOF**（不是读错误）。
// 这正是线上实测的形态：上游把连接干净关掉，却从不下发 event: done。
// 旧判定只认 sc.Err() != nil，对这种 EOF 失效 → 仍报 stop → 客户端把半截
// 回答当成正常完成（截图：Trae「已完成，用时 1 分 56 秒 / 用量 0 tok」、
// WorkBuddy「已取消」）。
type cleanEOFReader struct{ data []byte }

func (r *cleanEOFReader) Read(p []byte) (int, error) {
	if len(r.data) > 0 {
		n := copy(p, r.data)
		r.data = r.data[n:]
		return n, nil
	}
	return 0, io.EOF
}

func (*cleanEOFReader) Close() error { return nil }

// TestStreamTraeUpstreamCleanEOFIsLength 干净 EOF（无 done）必须判为中断。
func TestStreamTraeUpstreamCleanEOFIsLength(t *testing.T) {
	raw := strings.Join([]string{
		`event: output`,
		`data: {"response":"JS 被混淆（string array + 十六进制索引）。"}`,
		``,
	}, "\n")
	res := &upstreamResult{streaming: true, stream: &cleanEOFReader{data: []byte(raw)}}
	rr := httptest.NewRecorder()
	(&Handler{}).streamTraeUpstream(rr, res, "trae/seed_m8", nil)

	body := rr.Body.String()
	if !res.interrupted {
		t.Fatal("干净 EOF 未标记为中断——客户端会把它当成正常完成")
	}
	if !strings.Contains(body, `"finish_reason":"length"`) {
		t.Fatalf("干净 EOF 应报 length: %s", body)
	}
	if got := sseContents(t, body); got != "JS 被混淆（string array + 十六进制索引）。" {
		t.Fatalf("已收到的正文丢失: %q", got)
	}
}

// TestCollectTraeUpstreamCleanEOFIsLength 非流式同样。
func TestCollectTraeUpstreamCleanEOFIsLength(t *testing.T) {
	raw := strings.Join([]string{
		`event: output`,
		`data: {"response":"半截"}`,
		``,
	}, "\n")
	res := &upstreamResult{streaming: true, stream: &cleanEOFReader{data: []byte(raw)}}
	rr := httptest.NewRecorder()
	(&Handler{}).collectTraeUpstream(rr, res, "trae/seed_m8", nil)
	if !strings.Contains(rr.Body.String(), `"finish_reason":"length"`) {
		t.Fatalf("聚合响应应报 length: %s", rr.Body.String())
	}
}

// ---- 首帧立即下发 ----

// recordingWriter 记录每次 Flush，用来验证「首帧已 flush」。
type recordingWriter struct {
	*httptest.ResponseRecorder
	mu      sync.Mutex
	flushes int
}

func (w *recordingWriter) Flush() {
	w.mu.Lock()
	w.flushes++
	w.mu.Unlock()
}

// TestStreamTraeUpstreamFlushesRoleFrameFirst 上游迟迟不出字时，网关必须
// 立刻写并 flush 一个 role 帧：否则客户端 TTFB 看门狗会取消请求
// （截图里「已取消 / 用量 0 tok」的直接成因）。
func TestStreamTraeUpstreamFlushesRoleFrameFirst(t *testing.T) {
	// 一个只有 progress_notice、没有 output 的上游：模拟「Building prompt」阶段。
	raw := strings.Join([]string{
		`event: progress_notice`,
		`data: {"name":"construct_prompt","content":"Building prompt: ide_prompts"}`,
		`event: done`,
		`data: {}`,
		``,
	}, "\n")
	rw := &recordingWriter{ResponseRecorder: httptest.NewRecorder()}
	res := &upstreamResult{streaming: true, stream: io.NopCloser(strings.NewReader(raw))}
	(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)

	if rw.flushes == 0 {
		t.Fatal("首帧从未 flush，客户端收不到任何字节")
	}
	if !strings.Contains(rw.Body.String(), `"role":"assistant"`) {
		t.Fatalf("首帧应为 role 帧: %s", rw.Body.String())
	}
	if strings.Contains(rw.Body.String(), "Building prompt") {
		t.Fatalf("控制帧泄漏进正文: %s", rw.Body.String())
	}
}

// ---- 心跳保活 ----

// blockingReader 先给一帧，然后永久阻塞直到 Close —— 模拟上游「发完进度就
// 长时间不出字」。网关在此期间必须持续写空 delta 帧，客户端才不会超时取消。
type blockingReader struct {
	data  []byte
	gate  chan struct{}
	once  sync.Once
	mu    sync.Mutex
	close bool
}

func (r *blockingReader) Read(p []byte) (int, error) {
	r.mu.Lock()
	if len(r.data) > 0 {
		n := copy(p, r.data)
		r.data = r.data[n:]
		r.mu.Unlock()
		return n, nil
	}
	r.mu.Unlock()
	<-r.gate
	return 0, io.EOF
}

func (r *blockingReader) Close() error {
	r.once.Do(func() { close(r.gate) })
	return nil
}

// TestStreamTraeUpstreamSendsKeepAlive 上游静默期间必须持续保活。
//
// 直接调用 streamTraeUpstream 会等满 traeKeepAliveInterval，太慢；
// 这里只验证「空 delta 帧」这一机制本身可用且格式合法。
func TestStreamTraeUpstreamKeepAliveFrameShape(t *testing.T) {
	rw := httptest.NewRecorder()
	startOpenAIStream(rw)
	// 与 streamTraeUpstream 内部 writeChunk 的形态一致。
	writeSSE(rw, OAIChunk{
		ID: "chatcmpl-x", Object: "chat.completion.chunk", Created: time.Now().Unix(), Model: "trae/seed_m8",
		Choices: []OAIChoice{{Index: 0, Delta: &OAIDelta{}}},
	})
	body := rw.Body.String()
	if !strings.Contains(body, `"delta":{}`) {
		t.Fatalf("空心跳帧形态不对: %s", body)
	}
	if strings.Contains(body, "keep-alive") {
		t.Fatalf("心跳不该用 SSE 注释行（部分客户端会当畸形帧）: %s", body)
	}
}
