package zcode

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// ChatRequest 是 ZCode 对话所需的最小请求（内部结构，与 server 层解耦）。
type ChatRequest struct {
	Model       string          `json:"model"`
	Messages    json.RawMessage `json:"messages"`
	System      json.RawMessage `json:"system,omitempty"`
	MaxTokens   int             `json:"max_tokens"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Stop        json.RawMessage `json:"stop_sequences,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Tools       json.RawMessage `json:"tools,omitempty"`
	ToolChoice  json.RawMessage `json:"tool_choice,omitempty"`
	Metadata    json.RawMessage `json:"metadata,omitempty"`
	// Thinking 是 Anthropic 原生思考开关（{type:"enabled"|"disabled",
	// budget_tokens:N}），由客户端**原样透传**时使用。
	//
	// 注意：GLM-5.3 系列把档位写进 thinking.type 会被上游 400 拒绝 ——
	// 实测（2026-10-08，open.bigmodel.cn/api/anthropic/v1/messages）：
	//
	//	thinking:{type:"low"|"high"|"max"|"disabled"} → 400 code 1210
	//	  「该模型始终思考，不支持关闭思考；请使用 low、high 或 max。」
	//
	// 也就是说这句 400 文案说的"low、high、max"指的是 **reasoning_effort**
	// （见下 OutputConfig），不是 thinking.type 的取值。旧实现把档位写进
	// thinking.type，等于每次带档位都必然 400。
	Thinking json.RawMessage `json:"thinking,omitempty"`
	// OutputConfig 是档位的**真正载体**：{"effort":"low|high|max"}。
	//
	// 逆向自 ZCode 官方客户端配置（zcode.z.ai/api/v1/client/configs 的
	// builtinModels[].reasoning.levels.*.anthropic.set）：
	//
	//	{"path":["output_config","effort"],"value":"low"|"high"|"max"}
	//
	// 实测剂量效应（同一 prompt，GLM-5.3，thinking 块字符数）：
	//	无 effort（默认 max）→ 1485 / effort=low → 0（几乎不思考）
	//	effort=high → 71 / effort=max → 1941
	// 非法值 → 400「reasoning_effort 参数值非法，可选值为：
	//	none、minimal、low、medium、high、xhigh、max」。
	OutputConfig json.RawMessage `json:"output_config,omitempty"`
}

// Chat 发起一次 Anthropic Messages 请求。
//
// 【路由决策：凭证形态 → 端点 + 鉴权头】三处独立实现（D3-vin/Zcode2Api
// upstream.ts buildAuthHeaders、liu5269/zcode2api agent.py build_request、
// TriDefender/zcode-api upstream.ts）给出完全一致的三分表：
//
//	有 JWT（start-plan）      → zcode.z.ai/.../zcode-plan/anthropic/v1/messages
//	                            Authorization: Bearer <jwt>
//	有 API Key（coding-plan） → open.bigmodel.cn/api/anthropic/v1/messages
//	                            Authorization: Bearer <key>  （同时可带 x-api-key）
//	两者都有                  → **优先 JWT**（它是主通道，API Key 只是换钥产物）
//
// 旧实现的判定是 `useBigModel := c.a.APIKeyValue() != ""`，于是「面板登录的
// JWT 账号」（Mode=jwt 但带 api_key 用于签名）会被错误地路由到 bigmodel 端点，
// 同时只发 JWT 作为 Bearer —— bigmodel 不认 zcodejwt，必然
// 401「身份验证失败」。这正是线上一直复现的 401。
//
// verifyParam 为阿里云无痕验证参数（可为空）；
// extra 为需要透传或叠加的头部。
func (c *Client) Chat(ctx context.Context, req *ChatRequest, verifyParam string, extra http.Header) (*http.Response, error) {
	if req.MaxTokens <= 0 {
		req.MaxTokens = 8192
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	// 端点与鉴权头由同一处决定，避免「选 A 端点配 B 鉴权」这类错配。
	jwt := c.a.TokenValue()
	apiKey := c.a.APIKeyValue()
	useBigModel := jwt == "" && apiKey != ""
	endpoint := c.baseURL + PathChat
	if useBigModel {
		endpoint = BaseURLBigModel + PathChatBigModel
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	h := httpReq.Header
	if useBigModel {
		c.bigModelHeaders(h)
	} else {
		c.commonHeaders(h)
		c.authHeader(h)
		h.Set("X-ZCode-Agent", "glm")
		// trace trio 只属于对话面（billing/monitor 不带），且必须每请求重新生成。
		c.traceHeaders(h)
	}
	h.Set("Content-Type", "application/json")
	h.Set("anthropic-version", "2023-06-01")
	if req.Stream {
		h.Set("Accept", "text/event-stream")
	}
	if verifyParam != "" {
		h.Set("X-Aliyun-Captcha-Verify-Param", verifyParam)
		h.Set("X-Aliyun-Captcha-Verify-Region", c.captchaRegion())
	}
	// 透传入站头（不覆盖既有上游鉴权/版本头）。
	for k, vs := range extra {
		lk := strings.ToLower(k)
		if lk == "authorization" || lk == "x-api-key" || lk == "content-type" ||
			lk == "content-length" || lk == "host" || lk == "accept-encoding" ||
			// trace trio 由本函数生成：客户端的同名头若透传进来会覆盖掉，
			// 导致多请求共用同一 trace → 被上游判为重放。
			lk == "x-request-id" || lk == "x-zcode-trace-id" || lk == "x-zcode-session-type" ||
			// 3012 的两个直接触发器，任何来源都不许带上游。
			lk == "x-session-id" || lk == "x-query-id" {
			continue
		}
		for _, v := range vs {
			h.Set(k, v)
		}
	}
	// 客户端签名（若配置了凭据）。
	if cred := c.a.SignCredential(); cred != "" {
		// 注意：这里传空 origin，让 NewSigner 用 SignHandshakeOrigin。
		// 切勿把 c.baseURL（zcode.z.ai）当握手宿主，那个路径实测 404。
		sg, serr := NewSigner(cred, "", c.appVer, c.hc)
		if serr == nil {
			session := c.a.SessionID
			if session == "" {
				session = c.a.UID
			}
			// 签名协议本身要求 X-Session-Id（payload 里有它），这里由
			// SignRequest 写入 —— 与「start-plan 不该带 x-session-id」不矛盾：
			// 带签名 = coding-plan 通道，它的协议就是需要；不带签名 = start-plan
			// 通道，必须不带。
			_ = sg.SignRequest(ctx, httpReq, session, c.appVer)
		}
	}
	// 无签名时不发 X-Session-Id（旧实现会发，这是 3012 的直接触发器之一）。
	resp, err := c.hc.Do(httpReq)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// PathChatAnthropic 是 BigModel / Z.AI 的 Anthropic 兼容入口。
const PathChatAnthropic = "/api/anthropic/v1/messages"

// BaseURLZAI 是 Z.AI 开放平台（与 open.bigmodel.cn 等价的另一条通路）。
const BaseURLZAI = "https://api.z.ai"

// ChatViaKey 用「API Key 形态凭据」直接打 BigModel / Z.AI 的 Anthropic 端点。
//
// 为什么需要独立入口：zcode.z.ai 的 /api/v1/zcode-plan/anthropic 会对
// 非官方客户端返回 405 + code 3012「unusual activity」，无论带不带签名、
// 带什么 model / header 都一样（2026-10-06 全矩阵实测）；而同一把
// "<keyId>.<secret>" 凭据在 open.bigmodel.cn / api.z.ai 的 Anthropic
// 端点上实测 200 且出正文。凭据从 auth 的 SignCredential/APIKey 取。
//
// preferZAI=true 时先打 api.z.ai，失败再打 open.bigmodel.cn；否则反过来。
func (c *Client) ChatViaKey(ctx context.Context, req *ChatRequest, key string, preferZAI bool) (*http.Response, error) {
	if key == "" {
		return nil, fmt.Errorf("zcode: empty api key for anthropic fallback")
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = 8192
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	bases := []string{BaseURLBigModel, BaseURLZAI}
	if preferZAI {
		bases = []string{BaseURLZAI, BaseURLBigModel}
	}
	var lastErr error
	for _, base := range bases {
		hreq, err := http.NewRequestWithContext(ctx, "POST", base+PathChatAnthropic, bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		h := hreq.Header
		// 实测两种写法都能过：x-api-key 是 Anthropic 原生，Bearer 是平台原生。
		h.Set("x-api-key", key)
		h.Set("Authorization", "Bearer "+key)
		h.Set("Content-Type", "application/json")
		h.Set("anthropic-version", "2023-06-01")
		h.Set("Accept", "application/json")
		h.Set("User-Agent", "ZCode/"+c.appVer)
		h.Set("X-ZCode-App-Version", c.appVer)
		if req.Stream {
			h.Set("Accept", "text/event-stream")
		}
		resp, err := c.hc.Do(hreq)
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		resp.Body.Close()
		lastErr = fmt.Errorf("zcode: %s%s -> %d %s", base, PathChatAnthropic,
			resp.StatusCode, truncate2(string(body), 160))
	}
	return nil, lastErr
}

// truncate2 保留 helper 为局部私有，避免与 api.go 的 truncate 语义混淆。
func truncate2(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// captchaRegion 返回验证码区域（默认 cn）。
func (c *Client) captchaRegion() string {
	if c.a.Region != "" {
		return c.a.Region
	}
	return "cn"
}

// IsCaptchaError 判定响应体是否为无痕验证失败。
func IsCaptchaError(body string) bool {
	l := strings.ToLower(body)
	return strings.Contains(l, "captcha") || strings.Contains(l, "verify") ||
		strings.Contains(l, "3007") || strings.Contains(l, "人机")
}

// Token 返回账号 token（面板用）。
func (c *Client) Token() string { return c.a.TokenValue() }

// Ensure interface { io.Closer } usage placeholder.
var _ = io.Discard
var _ = fmt.Sprintf
