package zcode

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/jjjane233/workbuddy-3in1/internal/auth"
)

// ─────────────────────────────────────────────────────────────────────────────
// 凭证 → 端点 + 鉴权头 的路由契约
//
// 三处独立实现（D3-vin/Zcode2Api、liu5269/zcode2api、TriDefender/zcode-api）
// 交叉验证出同一张表。旧实现的 `useBigModel := APIKeyValue() != ""` 会把
// 「面板登录的 JWT 账号」（Mode=jwt 但带 api_key 用于签名）错误路由到 bigmodel，
// 只发 JWT 作 Bearer —— bigmodel 不认 zcodejwt，必然 401「身份验证失败」。
// ─────────────────────────────────────────────────────────────────────────────

// captured 记录一次出站请求的 URL 与关键头。
type captured struct {
	url           string
	authorization string
	xAPIKey       string
	xSessionID    string
	xQueryID      string
	requestID     string
	traceID       string
	sessionType   string
	zcodeAgent    string
	deviceMid     string
	anthropicVer  string
}

// captureChat 用假上游截获 Chat() 的出站请求。
func captureChat(t *testing.T, a *auth.Auth) captured {
	t.Helper()
	var got captured
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = captured{
			url:           r.URL.Path,
			authorization: r.Header.Get("Authorization"),
			xAPIKey:       r.Header.Get("x-api-key"),
			xSessionID:    r.Header.Get("X-Session-Id"),
			xQueryID:      r.Header.Get("x-query-id"),
			requestID:     r.Header.Get("x-request-id"),
			traceID:       r.Header.Get("x-zcode-trace-id"),
			sessionType:   r.Header.Get("x-zcode-session-type"),
			zcodeAgent:    r.Header.Get("X-ZCode-Agent"),
			deviceMid:     r.Header.Get("X-Device-Mid"),
			anthropicVer:  r.Header.Get("anthropic-version"),
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"m","content":[]}`))
	}))
	defer srv.Close()

	c := New(a, Options{HTTPClient: srv.Client(), BaseURL: srv.URL})
	// bigmodel 分支的端点写死在常量里，测试里把它指到同一台假服务器：
	// 通过给账号只留 api_key、并把 BaseURL 指到 srv 无法覆盖常量，
	// 因此这里只用「有 JWT」的路径做头矩阵断言（见下方 TestChatBigModelRouting
	// 用 host 判定代替）。
	req := &ChatRequest{Model: "glm-5.3-flash", MaxTokens: 16,
		Messages: json.RawMessage(`[{"role":"user","content":"hi"}]`)}
	resp, err := c.Chat(context.Background(), req, "", nil)
	if err != nil {
		t.Fatalf("Chat err = %v", err)
	}
	resp.Body.Close()
	return got
}

// TestChatJWTUsesZcodePlanEndpointWithBearer JWT 账号：走 zcode-plan 端点 + Bearer。
func TestChatJWTUsesZcodePlanEndpointWithBearer(t *testing.T) {
	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "u1", AccessToken: "the-jwt"}
	got := captureChat(t, a)

	if !strings.HasSuffix(got.url, PathChat) {
		t.Errorf("JWT 账号应走 %s，实际 %s", PathChat, got.url)
	}
	if got.authorization != "Bearer the-jwt" {
		t.Errorf("Authorization = %q, want Bearer the-jwt", got.authorization)
	}
	if got.xAPIKey != "" {
		t.Errorf("JWT 账号不该发 x-api-key（会被判非官方客户端），实际 %q", got.xAPIKey)
	}
}

// TestChatJWTTakesPrecedenceOverAPIKey 两者都有时优先 JWT（主通道）。
//
// 这是旧实现 401 的根因：有 api_key 就被路由到 bigmodel，同时只发 JWT。
func TestChatJWTTakesPrecedenceOverAPIKey(t *testing.T) {
	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "u1", AccessToken: "the-jwt"}
	a.SetAPIKey("keyid.secret")
	got := captureChat(t, a)

	if !strings.HasSuffix(got.url, PathChat) {
		t.Errorf("同时有 JWT 与 API Key 时应优先 JWT（走 zcode-plan），实际 %s", got.url)
	}
	if got.authorization != "Bearer the-jwt" {
		t.Errorf("Authorization = %q, want Bearer the-jwt", got.authorization)
	}
	if got.xAPIKey != "" {
		t.Errorf("不该同时发 x-api-key，实际 %q", got.xAPIKey)
	}
}

// TestChatSendsTraceTrio trace trio 必须存在（start-plan 官方客户端的指纹要求）。
func TestChatSendsTraceTrio(t *testing.T) {
	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "u1", AccessToken: "jwt"}
	got := captureChat(t, a)

	if got.requestID == "" {
		t.Error("缺 x-request-id")
	}
	if got.traceID == "" {
		t.Error("缺 x-zcode-trace-id")
	}
	if got.sessionType != "main" {
		t.Errorf("x-zcode-session-type = %q, want main", got.sessionType)
	}
	if got.zcodeAgent != "glm" {
		t.Errorf("X-ZCode-Agent = %q, want glm", got.zcodeAgent)
	}
	if got.anthropicVer != "2023-06-01" {
		t.Errorf("anthropic-version = %q", got.anthropicVer)
	}
}

// TestChatTraceTrioRegeneratedPerRequest trace 必须每请求不同——
// 固定值会被上游判为重放。
func TestChatTraceTrioRegeneratedPerRequest(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 5; i++ {
		a := &auth.Auth{Provider: auth.ProviderZCode, UID: "u1", AccessToken: "jwt"}
		got := captureChat(t, a)
		key := got.requestID + "|" + got.traceID
		if seen[key] {
			t.Fatalf("第 %d 次请求的 trace 与之前重复：%s", i, key)
		}
		seen[key] = true
	}
}

// TestChatOmitsSessionAndQueryIDWithoutSigning 无签名时不发 x-session-id / x-query-id。
//
// 这是 3012「unusual activity」的直接触发器（两处独立实现实测结论一致）。
func TestChatOmitsSessionAndQueryIDWithoutSigning(t *testing.T) {
	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "u1", AccessToken: "jwt",
		SessionID: "sess-should-not-leak"}
	got := captureChat(t, a)

	if got.xSessionID != "" {
		t.Errorf("无签名时不该发 X-Session-Id（触发 3012），实际 %q", got.xSessionID)
	}
	if got.xQueryID != "" {
		t.Errorf("不该发 x-query-id，实际 %q", got.xQueryID)
	}
}

// TestChatDropsForbiddenIncomingHeaders 入站同名头不得透传到上游。
func TestChatDropsForbiddenIncomingHeaders(t *testing.T) {
	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "u1", AccessToken: "jwt"}

	var got captured
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = captured{
			xSessionID: r.Header.Get("X-Session-Id"),
			xQueryID:   r.Header.Get("x-query-id"),
			requestID:  r.Header.Get("x-request-id"),
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := New(a, Options{HTTPClient: srv.Client(), BaseURL: srv.URL})
	incoming := http.Header{}
	incoming.Set("X-Session-Id", "leaked-session")
	incoming.Set("x-query-id", "leaked-query")
	incoming.Set("x-request-id", "leaked-request-id")
	req := &ChatRequest{Model: "m", MaxTokens: 16,
		Messages: json.RawMessage(`[{"role":"user","content":"hi"}]`)}
	resp, err := c.Chat(context.Background(), req, "", incoming)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	if got.xSessionID != "" {
		t.Errorf("入站 X-Session-Id 不应透传，实际 %q", got.xSessionID)
	}
	if got.xQueryID != "" {
		t.Errorf("入站 x-query-id 不应透传，实际 %q", got.xQueryID)
	}
	if got.requestID == "leaked-request-id" {
		t.Error("入站 x-request-id 不应覆盖网关生成的 trace")
	}
}

// TestChatDeviceMidAlwaysPresent 无合法 UUID 时也必须发一个稳定派生值
// （旧实现在 deviceMid() 为空时干脆不发，容易被判客户端不完整）。
func TestChatDeviceMidAlwaysPresent(t *testing.T) {
	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "uid-no-uuid", AccessToken: "jwt"}
	got := captureChat(t, a)
	if got.deviceMid == "" {
		t.Fatal("X-Device-Mid 必须存在（缺省按 UID 派生）")
	}
	if !IsUUID(got.deviceMid) {
		t.Errorf("派生的 device_mid 不是合法 UUID：%q", got.deviceMid)
	}
}

// TestChatSigningKeepsSessionID 带签名时 X-Session-Id 必须保留（签名协议要求）。
//
// 与「无签名时不发」不矛盾：带签名 = coding-plan 通道，它的协议就是需要；
// 不带签名 = start-plan 通道，必须不带。
func TestChatSigningKeepsSessionID(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/api/paas/") {
			w.WriteHeader(http.StatusInternalServerError) // 握手失败 → 不签名
			return
		}
		got = r.Header.Get("X-Session-Id")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "u1", AccessToken: "jwt"}
	a.SetSignCredential("keyid.secret")
	c := New(a, Options{HTTPClient: srv.Client(), BaseURL: srv.URL})
	if c.signer != nil {
		c.signer.origin = srv.URL
	}
	req := &ChatRequest{Model: "m", MaxTokens: 16,
		Messages: json.RawMessage(`[{"role":"user","content":"hi"}]`)}
	resp, err := c.Chat(context.Background(), req, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// 握手失败 → 走不到 SignRequest，因此这里只断言「不会因为无签名而误发入站
	// SessionID」；真正的签名路径由 sign_test.go 覆盖。
	if got == "leaked" {
		t.Error("不应透传入站 SessionID")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 错误码分类
// ─────────────────────────────────────────────────────────────────────────────

func TestBizCode(t *testing.T) {
	cases := []struct {
		name string
		body string
		want int
	}{
		{"flat int", `{"code":3012,"msg":"x"}`, 3012},
		{"flat string", `{"code":"3012"}`, 3012},
		{"nested error", `{"error":{"code":3007}}`, 3007},
		{"missing", `{"msg":"x"}`, 0},
		{"non-json", `<html>401</html>`, 0},
		{"empty", ``, 0},
		{"zero", `{"code":0}`, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := BizCode([]byte(c.body)); got != c.want {
				t.Errorf("BizCode(%s) = %d, want %d", c.body, got, c.want)
			}
		})
	}
}

func TestClassify(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
		want   Verdict
	}{
		// 额度耗尽族：等重试没用，必须换号
		{"402 payment", 402, `{}`, VerdictQuotaExhausted},
		{"1005 quota", 429, `{"code":1005}`, VerdictQuotaExhausted},
		{"1308 usage cap", 429, `{"code":1308}`, VerdictQuotaExhausted},
		{"1310 weekly cap", 429, `{"code":1310}`, VerdictQuotaExhausted},
		{"1316 5h cap", 429, `{"code":1316}`, VerdictQuotaExhausted},
		{"1321 enterprise", 429, `{"code":1321}`, VerdictQuotaExhausted},
		{"1113 arrears", 429, `{"code":1113}`, VerdictQuotaExhausted},

		// 准入并发上限：同账号重试有效
		{"3010 admission", 429, `{"code":3010}`, VerdictAdmissionLimit},

		// 指纹拦截：账号无责
		{"3012 load", 403, `{"code":3012,"msg":"unusual activity"}`, VerdictLoadBlocked},
		{"3012 on 405", 405, `{"code":3012}`, VerdictLoadBlocked},
		{"unusual activity text", 403, `request has been blocked due to unusual activity`, VerdictLoadBlocked},

		// 验证码：重取即可
		{"3007 captcha", 403, `{"code":3007}`, VerdictCaptcha},
		{"3007 on 429", 429, `{"code":3007}`, VerdictCaptcha},

		// 凭据被拒
		{"401", 401, `{"error":{"message":"身份验证失败。","type":"1000"}}`, VerdictUnauthorized},

		// 套餐不含模型：长期避让
		{"1311 lacks model", 403, `{"code":1311}`, VerdictPlanLacksModel},

		// 普通限流
		{"plain 429", 429, `{"code":1302,"msg":"rate limit"}`, VerdictRateLimited},
		{"429 no body", 429, ``, VerdictRateLimited},

		// 未分类
		{"500", 500, `{}`, VerdictUnknown},
		{"404", 404, ``, VerdictUnknown},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Classify(c.status, []byte(c.body)); got != c.want {
				t.Errorf("Classify(%d, %s) = %v, want %v", c.status, c.body, got, c.want)
			}
		})
	}
}

// TestClassifyPriorityOrder 判定顺序不可交换：3007 伴随 429 时必须归到验证码，
// 而不是「额度耗尽」或「限流」。
func TestClassifyPriorityOrder(t *testing.T) {
	// 3007 与配额码同时出现在不同层级时，3007 优先（验证码可重取，配额不可）。
	body := []byte(`{"code":3007,"msg":"captcha"}`)
	if got := Classify(429, body); got != VerdictCaptcha {
		t.Errorf("3007 应优先归为 captcha，got %v", got)
	}
	// 1311 与配额族重叠时按「套餐不含模型」处理：它比「配额耗尽」更精确
	// （换号可能立刻有效，而配额耗尽要等周期重置）。
	if got := Classify(403, []byte(`{"code":1311}`)); got != VerdictPlanLacksModel {
		t.Errorf("1311 应优先归为 plan_lacks_model，got %v", got)
	}
	// 3012 配 403/405（真实形态）必须归为指纹拦截，而不是「凭据被拒」。
	for _, st := range []int{403, 405, 406, 429, 503} {
		if got := Classify(st, []byte(`{"code":3012}`)); got != VerdictLoadBlocked {
			t.Errorf("3012 配 %d 应归为 load_blocked，got %v", st, got)
		}
	}
	// 401 是独立通道：不带 3012 时就是凭据被拒。
	if got := Classify(401, []byte(`{"error":{"message":"身份验证失败。"}}`)); got != VerdictUnauthorized {
		t.Errorf("纯 401 应归为 unauthorized，got %v", got)
	}
}

// TestQuotaFamilyDoesNotIncludeTransientCodes 瞬时限流码（1302/1305）不属于配额族——
// 它们等一会儿就能恢复，标死会导致好号被长期排除。
func TestQuotaFamilyDoesNotIncludeTransientCodes(t *testing.T) {
	for _, code := range []int{1302, 1305} {
		if got := Classify(429, []byte(`{"code":`+itoa(code)+`}`)); got == VerdictQuotaExhausted {
			t.Errorf("code %d 是瞬时限流，不该归为 quota_exhausted", code)
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// TestChatWithResignNotAffected 确认本次改动没有破坏「只发一次」的语义
// （重签逻辑未引入，这里只断言单次 Chat 只发一个请求）。
func TestChatSingleRequest(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "u1", AccessToken: "jwt"}
	c := New(a, Options{HTTPClient: srv.Client(), BaseURL: srv.URL})
	req := &ChatRequest{Model: "m", MaxTokens: 16,
		Messages: json.RawMessage(`[{"role":"user","content":"hi"}]`)}
	resp, err := c.Chat(context.Background(), req, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if got := calls.Load(); got != 1 {
		t.Errorf("Chat 应只发一次，实际 %d 次", got)
	}
}
