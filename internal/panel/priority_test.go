package panel

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/jjjane233/workbuddy-3in1/internal/auth"
	"github.com/jjjane233/workbuddy-3in1/internal/pool"
)

// priorityTestPanel 构造带真实池的面板（api_key 固定，便于直接调用 handler）。
func priorityTestPanel(t *testing.T) *Panel {
	t.Helper()
	p := pool.New("")
	p.Add(&auth.Auth{UID: "u1"})
	p.Add(&auth.Auth{UID: "u2"})
	return New(Config{Version: "test", APIKey: "k", Pool: p})
}

func postPriority(t *testing.T, p *Panel, uid, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/panel/api/accounts/"+uid+"/priority", bytes.NewReader([]byte(body)))
	req.Header.Set("Authorization", "Bearer k")
	req.SetPathValue("uid", uid)
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)
	return rec
}

// TestAccountPriorityEndpoint 端点写入生效，并回读钳位后的值。
func TestAccountPriorityEndpoint(t *testing.T) {
	p := priorityTestPanel(t)
	rec := postPriority(t, p, "u1", `{"priority":80}`)
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var out struct {
		OK       bool `json:"ok"`
		Priority int  `json:"priority"`
		Max      int  `json:"max"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if !out.OK || out.Priority != 80 || out.Max != pool.PriorityMax {
		t.Fatalf("resp=%+v", out)
	}
	if n, _ := p.cfg.Pool.Priority("u1"); n != 80 {
		t.Fatalf("池内 priority=%d want 80", n)
	}
}

// TestAccountPriorityClamps 越界输入被钳位并如实回读（不报错）。
func TestAccountPriorityClamps(t *testing.T) {
	p := priorityTestPanel(t)
	rec := postPriority(t, p, "u1", `{"priority":999}`)
	if rec.Code != 200 {
		t.Fatalf("越界输入应被钳位而非报错，status=%d", rec.Code)
	}
	var out struct {
		Priority int `json:"priority"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if out.Priority != pool.PriorityMax {
		t.Fatalf("priority=%d want %d", out.Priority, pool.PriorityMax)
	}
	// 负数同样归零（清除）。
	rec = postPriority(t, p, "u1", `{"priority":-7}`)
	if rec.Code != 200 {
		t.Fatalf("status=%d", rec.Code)
	}
	if n, _ := p.cfg.Pool.Priority("u1"); n != 0 {
		t.Fatalf("负数应归 0，got %d", n)
	}
}

// TestAccountPriorityUnknownUID 未知账号返回 404。
func TestAccountPriorityUnknownUID(t *testing.T) {
	p := priorityTestPanel(t)
	rec := postPriority(t, p, "nope", `{"priority":10}`)
	if rec.Code != 404 {
		t.Fatalf("status=%d want 404", rec.Code)
	}
}

// TestAccountPriorityBadJSON 非法 JSON 返回 400（而不是静默按 0 处理）。
func TestAccountPriorityBadJSON(t *testing.T) {
	p := priorityTestPanel(t)
	rec := postPriority(t, p, "u1", `{oops`)
	if rec.Code != 400 {
		t.Fatalf("status=%d want 400", rec.Code)
	}
}

// TestAccountPriorityRequiresAuth 端点必须走 api_key 鉴权。
func TestAccountPriorityRequiresAuth(t *testing.T) {
	p := priorityTestPanel(t)
	req := httptest.NewRequest("POST", "/panel/api/accounts/u1/priority", bytes.NewReader([]byte(`{"priority":10}`)))
	req.SetPathValue("uid", "u1")
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)
	if rec.Code != 401 {
		t.Fatalf("未带 key 应 401，got %d", rec.Code)
	}
}

// TestOverviewExposesPriority 面板 overview 必须带上 priority 字段，
// 否则前端 chip 永远显示「+权重」。
func TestOverviewExposesPriority(t *testing.T) {
	p := priorityTestPanel(t)
	p.cfg.Pool.SetPriority("u1", 65)
	req := httptest.NewRequest("GET", "/panel/api/overview", nil)
	req.Header.Set("Authorization", "Bearer k")
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d", rec.Code)
	}
	var out struct {
		Accounts []struct {
			UID      string `json:"uid"`
			Priority int    `json:"priority"`
		} `json:"accounts"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, a := range out.Accounts {
		if a.UID == "u1" {
			found = true
			if a.Priority != 65 {
				t.Fatalf("overview priority=%d want 65", a.Priority)
			}
		}
	}
	if !found {
		t.Fatal("overview 未返回 u1")
	}
}
