package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jjjane233/workbuddy-3in1/internal/auth"
	"github.com/jjjane233/workbuddy-3in1/internal/pool"
)

// ZCode 错误处置的端到端契约。
//
// 背景：旧实现按 status 逐分支处置，四类完全不同的问题塌缩成两句话：
//   额度耗尽 / 并发超限 / 验证码失效 / 指纹被拦 → 全被当成「429 限流」或「401 未授权」，
// 且都去冷却账号（其中两类根本不该罚号）。这些用例把「分类 → 处置」钉死。

// zcodeStatusHTTP 构造只接管 ZCode 端点的假上游。
func zcodeStatusHTTP(status int, header http.Header, body string) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		h := http.Header{"Content-Type": []string{"application/json"}}
		for k, vs := range header {
			h[k] = vs
		}
		return &http.Response{
			StatusCode: status,
			Header:     h,
			Body:       io.NopCloser(strings.NewReader(body)),
		}, nil
	})}
}

// zcodeChatHandler 构造只含一个 ZCode 号的 handler（JWT 账号，走 zcode-plan 端点）。
func zcodeChatHandler(t *testing.T, hc *http.Client) (*Handler, *pool.Pool) {
	t.Helper()
	a := &auth.Auth{
		Provider:    auth.ProviderZCode,
		UID:         "zc1",
		AccessToken: "jwt",
		Source:      "bigmodel",
	}
	p := testPoolWith(a)
	h := NewHandler(Config{
		Pool:         p,
		Upstream:     newFakeUpstream(t, func(string) (int, string, bool) { return 200, sseOK, true }),
		UpstreamHTTP: hc,
		MaxRotate:    1,
	})
	return h, p
}

// zcodeChatCall 发一次 zcode 模型请求，返回响应码与解析后的错误体。
func zcodeChatCall(t *testing.T, h *Handler) (int, map[string]any) {
	t.Helper()
	body := `{"model":"zcode:glm-5.3-flash","messages":[{"role":"user","content":"hi"}],"stream":false}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

// errMessage 取 OpenAI 错误体的 message。
func errMessage(out map[string]any) string {
	e, _ := out["error"].(map[string]any)
	if e == nil {
		return ""
	}
	s, _ := e["message"].(string)
	return s
}

// TestZCodeQuotaExhaustedBlocksModelNotAccount 额度耗尽必须「按模型标死」，
// 而不是冷却整个账号——账号对其他模型仍可能可用。
func TestZCodeQuotaExhaustedBlocksModelNotAccount(t *testing.T) {
	h, p := zcodeChatHandler(t, zcodeStatusHTTP(429, nil, `{"code":1308,"msg":"usage cap reached"}`))
	zcodeChatCall(t, h)

	st, _ := p.Status("zc1")
	// 关键断言：账号级**不冷却**（旧实现会 600s 软冷却，白占一个可用号）。
	if st.Cooling {
		t.Errorf("额度耗尽不该冷却整个账号: kind=%q remaining=%ds", st.CoolKind, st.CoolRemaining)
	}
	// 但该模型必须被标死（选号侧 healthyForModel 会避开它）。
	blocked := false
	for _, m := range st.RateLimitedModels {
		if strings.Contains(m.Model, "glm-5.3-flash") {
			blocked = true
			break
		}
	}
	if !blocked {
		t.Errorf("该模型应被标死（modelCooldowns），实际 RateLimitedModels=%+v", st.RateLimitedModels)
	}
}

// TestZCode402AlsoBlocksModel 402 与配额族同处置。
func TestZCode402AlsoBlocksModel(t *testing.T) {
	h, p := zcodeChatHandler(t, zcodeStatusHTTP(402, nil, `{"error":{"message":"insufficient balance"}}`))
	zcodeChatCall(t, h)

	st, _ := p.Status("zc1")
	if st.Cooling {
		t.Errorf("402 不该冷却整个账号: kind=%q", st.CoolKind)
	}
	if len(st.RateLimitedModels) == 0 {
		t.Error("402 应把该模型标死")
	}
}

// TestZCode401KeepsUpstreamBody 401 必须保留上游原文（运维唯一可见处）。
func TestZCode401KeepsUpstreamBody(t *testing.T) {
	h, p := zcodeChatHandler(t, zcodeStatusHTTP(401,
		nil, `{"error":{"message":"身份验证失败。","type":"1000"}}`))
	_, out := zcodeChatCall(t, h)

	if msg := errMessage(out); !strings.Contains(msg, "身份验证失败") {
		t.Errorf("错误 message 未带上游原文: %q", msg)
	}
	st, _ := p.Status("zc1")
	if !strings.Contains(st.Reason, "身份验证失败") {
		t.Errorf("面板 reason 未带上游原文: %q", st.Reason)
	}
	// 401 是真凭据问题，仍按硬冷却（等重登/重同步）。
	if st.CoolKind != "hard_credit" {
		t.Errorf("401 cool_kind = %q, want hard_credit", st.CoolKind)
	}
}

// TestZCode429KeepsUpstreamBodyAndFollowsRetryAfter 普通 429：保留原文 + 跟随 Retry-After。
func TestZCode429KeepsUpstreamBodyAndFollowsRetryAfter(t *testing.T) {
	hdr := http.Header{"Retry-After": []string{"30"}}
	h, p := zcodeChatHandler(t, zcodeStatusHTTP(429, hdr,
		`{"code":1302,"msg":"rate limit exceeded, please retry in 30s"}`))
	_, out := zcodeChatCall(t, h)

	if msg := errMessage(out); !strings.Contains(msg, "rate limit exceeded") {
		t.Errorf("429 错误 message 未带上游原文: %q", msg)
	}
	st, _ := p.Status("zc1")
	if !st.Cooling {
		t.Fatal("普通 429 应冷却账号")
	}
	// Retry-After: 30s → 冷却剩余应在 30s 附近（允许误差）。
	if st.CoolRemaining > 60 || st.CoolRemaining < 25 {
		t.Errorf("冷却剩余 = %ds, want ≈30s（应跟随 Retry-After）", st.CoolRemaining)
	}
}

// TestZCode429WithoutRetryAfterFallsBackToSoftBase 无 Retry-After 时回落软冷却基数。
func TestZCode429WithoutRetryAfterFallsBackToSoftBase(t *testing.T) {
	h, p := zcodeChatHandler(t, zcodeStatusHTTP(429, nil, `{"code":1305,"msg":"too many requests"}`))
	zcodeChatCall(t, h)

	st, _ := p.Status("zc1")
	if !st.Cooling {
		t.Fatal("应冷却账号")
	}
	// 默认 soft_rate 600s。
	if st.CoolRemaining < 500 || st.CoolRemaining > 700 {
		t.Errorf("冷却剩余 = %ds, want ≈600s（软冷却基数）", st.CoolRemaining)
	}
}

// TestZCode3012LoadBlockedDoesNotPunishAccount 指纹拦截（3012）不该罚账号。
//
// 旧实现把它当 403/405 处理并 noteUpstreamHard（硬冷却到次日 04:00），
// 一个完全健康的号会被白白封 8~20 小时。
func TestZCode3012LoadBlockedDoesNotPunishAccount(t *testing.T) {
	h, p := zcodeChatHandler(t, zcodeStatusHTTP(403,
		nil, `{"code":3012,"msg":"request has been blocked due to unusual activity."}`))
	zcodeChatCall(t, h)

	st, _ := p.Status("zc1")
	if st.CoolKind == "hard_credit" {
		t.Errorf("3012 指纹拦截不该硬冷却账号（换端点/换号即可）: reason=%q", st.Reason)
	}
	if st.Disabled {
		t.Error("3012 不该禁用账号")
	}
}

// TestZCodeCaptchaInvalidatesCacheWithoutCooldown 验证码失效：作废缓存，不罚号。
func TestZCodeCaptchaInvalidatesCacheWithoutCooldown(t *testing.T) {
	h, p := zcodeChatHandler(t, zcodeStatusHTTP(403, nil, `{"code":3007,"msg":"captcha failed"}`))
	zcodeChatCall(t, h)

	st, _ := p.Status("zc1")
	if st.Cooling {
		t.Errorf("验证码失效不该冷却账号（重取验证码即可）: kind=%q remaining=%ds",
			st.CoolKind, st.CoolRemaining)
	}
	if st.Disabled {
		t.Error("验证码失效不该禁用账号")
	}
}

// TestZCode1311PlanLacksModelBlocksModel 套餐不含模型：按模型长期避让，不冷却账号。
func TestZCode1311PlanLacksModelBlocksModel(t *testing.T) {
	h, p := zcodeChatHandler(t, zcodeStatusHTTP(403, nil, `{"code":1311,"msg":"plan lacks model"}`))
	zcodeChatCall(t, h)

	st, _ := p.Status("zc1")
	if st.Cooling {
		t.Errorf("套餐不含模型不该冷却账号: kind=%q", st.CoolKind)
	}
	if len(st.RateLimitedModels) == 0 {
		t.Error("该模型应被标死（长期避让）")
	}
}

// TestZCode3010AdmissionLimitSoftCooldown 3010 准入并发上限：短冷却后换号。
func TestZCode3010AdmissionLimitSoftCooldown(t *testing.T) {
	h, p := zcodeChatHandler(t, zcodeStatusHTTP(429, nil, `{"code":3010,"msg":"admission limit"}`))
	_, out := zcodeChatCall(t, h)

	if msg := errMessage(out); !strings.Contains(msg, "admission") {
		t.Errorf("错误 message 应指明准入上限: %q", msg)
	}
	st, _ := p.Status("zc1")
	// 3010 走软冷却（600s 基数），不是硬冷却。
	if st.CoolKind == "hard_credit" {
		t.Error("3010 不该硬冷却")
	}
}

// TestZCodeFallbackSuccessDoesNotCooldown 3012 回退成功时绝不能给账号上冷却。
//
// 旧代码在「fallback ok」分支调 noteUpstreamSoft，等于每次成功回退都把好号罚 600s。
func TestZCodeFallbackSuccessDoesNotCooldown(t *testing.T) {
	var calls int
	hc := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		// 第一次（zcode.z.ai zcode-plan）回 405+3012；第二次（bigmodel anthropic）回 200。
		if calls == 1 {
			return &http.Response{
				StatusCode: http.StatusMethodNotAllowed,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"code":3012,"msg":"unusual activity"}`)),
			}, nil
		}
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"id":"m1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`)),
		}, nil
	})}

	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "zc1", AccessToken: "jwt", Source: "bigmodel"}
	a.SetAPIKey("keyid.secret")
	a.SetSignCredential("keyid.secret")
	p := testPoolWith(a)
	h := NewHandler(Config{
		Pool:         p,
		Upstream:     newFakeUpstream(t, func(string) (int, string, bool) { return 200, sseOK, true }),
		UpstreamHTTP: hc,
		MaxRotate:    1,
	})

	zcodeChatCall(t, h)

	st, _ := p.Status("zc1")
	if st.Cooling {
		t.Errorf("回退成功不该冷却账号: kind=%q reason=%q remaining=%ds",
			st.CoolKind, st.Reason, st.CoolRemaining)
	}
	if st.SuccessCount == 0 {
		t.Error("回退成功应记一次成功")
	}
}

// TestZCodeUsesInjectedHTTPClient Config.UpstreamHTTP 必须真的被 ZCode 链路用上。
//
// 旧实现漏传，导致部署方配置的出口代理对 ZCode 对话无效、测试也无法注入假上游。
func TestZCodeUsesInjectedHTTPClient(t *testing.T) {
	var hits int
	hc := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		hits++
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"id":"m1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`)),
		}, nil
	})}

	a := &auth.Auth{Provider: auth.ProviderZCode, UID: "zc1", AccessToken: "jwt"}
	p := testPoolWith(a)
	h := NewHandler(Config{
		Pool:         p,
		Upstream:     newFakeUpstream(t, func(string) (int, string, bool) { return 200, sseOK, true }),
		UpstreamHTTP: hc,
		MaxRotate:    1,
	})
	zcodeChatCall(t, h)

	if hits == 0 {
		t.Fatal("注入的 UpstreamHTTP 未被使用（说明 ZCode 链路仍在自建 http.Client）")
	}
}

// 确保 time 被使用。
var _ = time.Second
