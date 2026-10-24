package upstream

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/jjjane233/workbuddy-3in1/internal/auth"
)

// TestWithTokenRefreshRetryOn401 401 时强制刷新 token 并重试一次（成功即自愈）。
// 现场：余额/签到这类周期任务没有 chat 的换号轮转，token 被上游作废后会一直
// 拿同一个死 token 打同一端点——日志里只剩刷屏的 401（"balance PENG: upstream
// client (http 401)"），账号永远不恢复。
func TestWithTokenRefreshRetryOn401(t *testing.T) {
	var refreshCalls int
	c := testClient(func(r *http.Request) (*http.Response, error) {
		if strings.HasSuffix(r.URL.Path, "/v2/plugin/auth/token/refresh") {
			refreshCalls++
			return jsonResp(200, `{"code":0,"data":{"accessToken":"fresh","refreshToken":"rt2","expiresIn":3600}}`), nil
		}
		return nil, errors.New("unexpected path: " + r.URL.Path)
	})
	a := &auth.Auth{AccessToken: "at", RefreshToken: "rt1", ExpiresAt: 1}

	var calls int
	err := c.WithTokenRefreshRetry(a, func() error {
		calls++
		if calls == 1 {
			return &Error{Kind: ErrUnauthorized, Status: 401, Msg: "401 Authorization Required"}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("retry should succeed: %v", err)
	}
	if refreshCalls != 1 {
		t.Errorf("refreshCalls=%d want 1", refreshCalls)
	}
	if calls != 2 {
		t.Errorf("fn calls=%d want 2（首次 401 + 刷新后重试）", calls)
	}
	if a.AccessToken != "fresh" {
		t.Errorf("accessToken=%q want fresh", a.AccessToken)
	}
}

// TestWithTokenRefreshRetryNon401Passthrough 非 401 错误原样返回、不刷新
// （429/5xx 各有自己的冷却/重试策略，不得在此叠加）。
func TestWithTokenRefreshRetryNon401Passthrough(t *testing.T) {
	var refreshCalls int
	c := testClient(func(r *http.Request) (*http.Response, error) {
		refreshCalls++
		return jsonResp(200, `{"code":0,"data":{"accessToken":"fresh"}}`), nil
	})
	a := &auth.Auth{AccessToken: "at", RefreshToken: "rt1"}
	want := &Error{Kind: ErrSoftRate, Status: 429, Msg: "rate limit"}
	got := c.WithTokenRefreshRetry(a, func() error { return want })
	if got != want {
		t.Errorf("err=%v want 原样返回", got)
	}
	if refreshCalls != 0 {
		t.Errorf("refreshCalls=%d want 0（非 401 不刷新）", refreshCalls)
	}
}

// TestWithTokenRefreshRetryRefreshFails 刷新也失败 → 返回原始 401 并带上刷新原因
// （调用方据此禁用并要求重登）。
func TestWithTokenRefreshRetryRefreshFails(t *testing.T) {
	c := testClient(func(r *http.Request) (*http.Response, error) {
		return jsonResp(401, `{"code":9999,"msg":"bad token"}`), nil
	})
	a := &auth.Auth{AccessToken: "at", RefreshToken: "rt1"}
	err := c.WithTokenRefreshRetry(a, func() error {
		return &Error{Kind: ErrUnauthorized, Status: 401, Msg: "401 Authorization Required"}
	})
	if err == nil {
		t.Fatal("want error")
	}
	if !IsUnauthorized(err) {
		t.Errorf("err=%v 应仍可识别为 401（errors.Is/As 链保留）", err)
	}
	if !strings.Contains(err.Error(), "token refresh failed") {
		t.Errorf("err=%v 应带刷新失败原因", err)
	}
}

// TestIsUnauthorized 只认 ErrUnauthorized（12153 session_dead 不算）。
func TestIsUnauthorized(t *testing.T) {
	if !IsUnauthorized(&Error{Kind: ErrUnauthorized, Status: 401}) {
		t.Error("ErrUnauthorized 应命中")
	}
	if IsUnauthorized(&Error{Kind: ErrSessionDead, Status: 401}) {
		t.Error("session_dead 不是 unauthorized（终态不同：一个禁用待重登，一个刷新可救）")
	}
	if IsUnauthorized(errors.New("plain")) || IsUnauthorized(nil) {
		t.Error("非 *Error / nil 不应命中")
	}
}
