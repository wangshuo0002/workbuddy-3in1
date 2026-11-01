package server

import (
	"bufio"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// slowTraeUpstream 模拟真实 Trae：先静默 silent 时长（只发 progress_notice），
// 之后才开始出字并正常 done。用来验证「静默期不取消 + 心跳持续 + 正文完整」。
type slowTraeUpstream struct {
	silent    time.Duration
	startedAt time.Time
	queue     []string
	buf       []byte
	stage     int
}

func newSlowTraeUpstream(silent time.Duration) *slowTraeUpstream {
	return &slowTraeUpstream{silent: silent, startedAt: time.Now(), queue: []string{
		"event: progress_notice\n",
		`data: {"name":"construct_prompt","content":"Building prompt: ide_prompts"}` + "\n",
		"\n",
	}}
}

// tail 静默期结束后要发的内容（只追加一次）。
func (u *slowTraeUpstream) tail() []string {
	return []string{
		"event: output\n",
		`data: {"response":"JS 被混淆（string array + 十六进制索引）。"}` + "\n",
		"\n",
		"event: output\n",
		`data: {"response":"先分流：哪些文件是明文、哪些被混淆。"}` + "\n",
		"\n",
		"event: token_usage\n",
		`data: {"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}` + "\n",
		"\n",
		"event: done\n",
		"data: {}\n",
		"\n",
	}
}

func (u *slowTraeUpstream) Read(p []byte) (int, error) {
	if len(u.buf) > 0 {
		n := copy(p, u.buf)
		u.buf = u.buf[n:]
		return n, nil
	}
	if len(u.queue) == 0 {
		if u.stage == 0 {
			// 进度帧发完：进入静默期，之后换成 tail 内容。
			if elapsed := time.Since(u.startedAt); elapsed < u.silent {
				time.Sleep(u.silent - elapsed)
			}
			u.queue = u.tail()
			u.stage = 1
		} else {
			return 0, io.EOF
		}
	}
	u.buf = []byte(u.queue[0])
	u.queue = u.queue[1:]
	return u.Read(p)
}

func (*slowTraeUpstream) Close() error { return nil }

// TestStreamTraeUpstreamSurvivesSilentUpstream 端到端：上游静默远超心跳间隔，
// 网关必须持续发心跳（客户端不取消），静默结束后正文完整送达且 finish=stop。
func TestStreamTraeUpstreamSurvivesSilentUpstream(t *testing.T) {
	withKeepAliveInterval(t, 20*time.Millisecond)
	up := newSlowTraeUpstream(300 * time.Millisecond)
	rw := &flushCountingWriter{ResponseRecorder: httptest.NewRecorder()}
	res := &upstreamResult{streaming: true, stream: up, cancel: func() {}}

	done := make(chan struct{})
	go func() {
		(&Handler{}).streamTraeUpstream(rw, res, "trae/seed_m8", nil)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("流未在预期内结束")
	}

	body := rw.Body.String()
	// 静默 300ms / 心跳 20ms ≈ 15 次，断言至少 5 次（宽松，防抖动）。
	if n := strings.Count(body, `"delta":{}`); n < 5 {
		t.Fatalf("静默期只发了 %d 个心跳帧，客户端会取消。body=%q", n, body)
	}
	if got := sseContents(t, body); got != "JS 被混淆（string array + 十六进制索引）。先分流：哪些文件是明文、哪些被混淆。" {
		t.Fatalf("正文不完整: %q", got)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("正常 done 应为 stop: %s", body)
	}
	if res.interrupted {
		t.Fatal("正常收尾被误判为中断")
	}
	if strings.Contains(body, "Building prompt") {
		t.Fatalf("控制帧泄漏: %s", body)
	}
}

// TestStreamTraeUpstreamRealHTTPThroughServer 走真实 HTTP 栈：
// 验证字节真的被 flush 到连接上（httptest.ResponseRecorder 不校验这点）。
func TestStreamTraeUpstreamRealHTTPThroughServer(t *testing.T) {
	if testing.Short() {
		t.Skip("依赖真实网络读时序")
	}
	withKeepAliveInterval(t, 20*time.Millisecond)
	up := newSlowTraeUpstream(250 * time.Millisecond)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		res := &upstreamResult{streaming: true, stream: up, cancel: func() {}}
		(&Handler{}).streamTraeUpstream(w, res, "trae/seed_m8", nil)
	}))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	// 读第一帧：必须在静默期结束前就到达（即首帧已 flush）。
	br := bufio.NewReader(resp.Body)
	first := make(chan string, 1)
	go func() {
		line, _ := br.ReadString('\n')
		first <- line
	}()
	select {
	case line := <-first:
		if !strings.Contains(line, "data:") {
			t.Fatalf("首帧应立刻到达: %q", line)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("首帧未及时 flush（客户端会判响应停滞）")
	}
	rest, _ := io.ReadAll(br)
	if !strings.Contains(string(rest), `"finish_reason":"stop"`) {
		t.Fatalf("收尾帧缺失: %s", rest)
	}
}
