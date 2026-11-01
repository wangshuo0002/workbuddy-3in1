package server

import (
	"io"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// ---- 上游静默期间的保活 ----

// idleUpstreamReader 永不返回数据，阻塞到被取消（Close 或 ctx 取消）为止。
// 模拟上游「连接建立后长时间静默」：Trae 的 Building prompt 阶段实测可达 1~2 分钟。
type idleUpstreamReader struct {
	gate chan struct{}
	once sync.Once
}

func newIdleUpstream() *idleUpstreamReader {
	return &idleUpstreamReader{gate: make(chan struct{})}
}

func (r *idleUpstreamReader) Read([]byte) (int, error) {
	<-r.gate
	return 0, io.EOF
}

func (r *idleUpstreamReader) Close() error {
	r.once.Do(func() { close(r.gate) })
	return nil
}

// flushCountingWriter 记录 flush 次数（httptest.ResponseRecorder 虽自带
// Flush，但无法计数）。
type flushCountingWriter struct {
	*httptest.ResponseRecorder
	mu sync.Mutex
	n  int
}

func (w *flushCountingWriter) Flush() {
	w.mu.Lock()
	w.n++
	w.mu.Unlock()
}

func (w *flushCountingWriter) flushes() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.n
}

// withKeepAliveInterval 临时缩短心跳窗口（测试用）。
func withKeepAliveInterval(t *testing.T, d time.Duration) {
	t.Helper()
	old := traeKeepAliveInterval
	traeKeepAliveInterval = d
	t.Cleanup(func() { traeKeepAliveInterval = old })
}

// TestStreamTraeUpstreamKeepAliveWhileIdle 上游静默期间必须持续下发空 delta 帧。
//
// 这是「话没说完就停了」最直接的一条：Trae 上游在出字前可能整分钟只发
// progress_notice，网关旧实现一个字节都不写，客户端空闲看门狗判定「响应停滞」
// 后主动取消（截图：Trae「已取消」/ WorkBuddy「用户已取消」，用量 0 tok）。
func TestStreamTraeUpstreamKeepAliveWhileIdle(t *testing.T) {
	withKeepAliveInterval(t, 20*time.Millisecond)
	up := newIdleUpstream()
	rw := &flushCountingWriter{ResponseRecorder: httptest.NewRecorder()}
	res := &upstreamResult{streaming: true, stream: up, cancel: func() { _ = up.Close() }}

	done := make(chan struct{})
	go func() {
		(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)
		close(done)
	}()

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if strings.Count(rw.Body.String(), `"delta":{}`) >= 2 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	body := rw.Body.String()
	if n := strings.Count(body, `"delta":{}`); n < 2 {
		t.Fatalf("上游静默期间只发了 %d 个心跳帧（客户端会取消）: %q", n, body)
	}
	if strings.Contains(body, "keep-alive") {
		t.Fatalf("心跳不该用 SSE 注释行（部分客户端当畸形帧）: %q", body)
	}
	if rw.flushes() < 3 {
		t.Fatalf("flush 次数不足（%d），字节会滞留在缓冲里", rw.flushes())
	}

	_ = up.Close()
	<-done
	if !res.interrupted {
		t.Fatal("全程没有 output 就结束，应判为中断")
	}
}

// TestStreamTraeUpstreamIdleCapAborts 上游持续静默超过上限时必须放弃，
// 不能靠心跳让用户无限等待。
func TestStreamTraeUpstreamIdleCapAborts(t *testing.T) {
	withKeepAliveInterval(t, 10*time.Millisecond)
	oldCap := traeUpstreamIdleCap
	traeUpstreamIdleCap = 40 * time.Millisecond
	t.Cleanup(func() { traeUpstreamIdleCap = oldCap })

	up := newIdleUpstream()
	var cancelled bool
	var mu sync.Mutex
	rw := httptest.NewRecorder()
	res := &upstreamResult{streaming: true, stream: up, cancel: func() {
		mu.Lock()
		cancelled = true
		mu.Unlock()
		_ = up.Close()
	}}

	done := make(chan struct{})
	go func() {
		(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("上游假死时未在 idleCap 内放弃")
	}
	mu.Lock()
	got := cancelled
	mu.Unlock()
	if !got {
		t.Fatal("未取消上游请求")
	}
	if !res.interrupted {
		t.Fatal("假死收尾应判为中断")
	}
	if !strings.Contains(rw.Body.String(), `"finish_reason":"length"`) {
		t.Fatalf("假死应收尾为 length: %s", rw.Body.String())
	}
}

// TestStreamTraeUpstreamFirstFlushBeforeAnyOutput 首帧必须在读上游之前
// 就写出并 flush：客户端 TTFB 看门狗不等上游预热。
func TestStreamTraeUpstreamFirstFlushBeforeAnyOutput(t *testing.T) {
	raw := strings.Join([]string{
		`event: progress_notice`,
		`data: {"name":"construct_prompt","content":"Building prompt: ide_prompts"}`,
		`event: done`,
		`data: {}`,
		``,
	}, "\n")
	rw := &flushCountingWriter{ResponseRecorder: httptest.NewRecorder()}
	res := &upstreamResult{streaming: true, stream: io.NopCloser(strings.NewReader(raw))}
	(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)

	if !strings.Contains(rw.Body.String(), `"role":"assistant"`) {
		t.Fatalf("首帧应为 role 帧: %s", rw.Body.String())
	}
	if rw.flushes() == 0 {
		t.Fatal("首帧从未 flush")
	}
	if !strings.Contains(rw.Body.String(), `"finish_reason":"stop"`) {
		t.Fatalf("有 done 的流应为 stop: %s", rw.Body.String())
	}
}
