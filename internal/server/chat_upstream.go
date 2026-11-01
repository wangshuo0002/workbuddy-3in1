package server

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/jjjane233/workbuddy-3in1/internal/auth"
	"github.com/jjjane233/workbuddy-3in1/internal/pool"
	"github.com/jjjane233/workbuddy-3in1/internal/reqlog"
	"github.com/jjjane233/workbuddy-3in1/internal/trae"
	"github.com/jjjane233/workbuddy-3in1/internal/upstream"
	"github.com/jjjane233/workbuddy-3in1/internal/zcode"
)

// 三合一网关的 Trae / ZCode 出口（Phase 4b）。
//
// CodeBuddy 链路一行未改：chatCompletions 判定 provider 后，非 workbuddy 才转发到这里，
// 因此存量行为完全不受影响。本文件的结构与调用口径对齐 gw2api/internal/server/chat.go，
// 差异只在「池与冷却」换用 workbuddy 自己的 API（见 noteUpstream* 三个适配函数）。

// 上游（Trae/ZCode）单请求最多换号次数的默认值与硬上限。
//
// 默认值历史就是 6（与 gw2api 的 maxAccountAttempts 同值）；上硬上限是为了防止
// 面板误配把单请求放大成无限重试风暴。真正生效的值由 Config.UpstreamMaxAttempts
// 经 livecfg 热改，读取口是 (*Handler).maxUpstreamAttempts()。
const (
	defaultUpstreamMaxAttempts = 6
	maxUpstreamMaxAttempts     = 32
)

// upstreamResult 单次上游尝试的结果。
type upstreamResult struct {
	streaming bool
	// 非流式
	status int
	body   []byte
	// 流式
	stream io.ReadCloser
	// usage 非流式响应的 token 用量（流式由转换函数边写边解析，回填在返回值里）。
	usage upstreamUsage
	// 控制
	retry  bool
	errMsg string
	// cancel 释放本次 Trae 上游请求的 context。排队预检命中时立即调用，
	// 避免上游连接继续占用账号并发槽。
	cancel context.CancelFunc
	// interrupted 上游流在**未发出收尾帧**的情况下提前结束（EOF / 读错误 /
	// 未收到 event: done）。此时客户端已经收到 200 与部分正文，无法重试，
	// 唯一正确做法是把「流被截断」明确告知客户端并记为流错误——
	// 旧实现对此一无所知，仍写出 finish_reason=stop + [DONE]，于是客户端把
	// 半截回答当成正常完成（面板显示「已完成」但内容停在半句），
	// 这也是「对话没输出完就停止」最隐蔽的一条成因。
	interrupted bool
}

// ---- 池与冷却适配层 ----
//
// workbuddy 的池没有 gw2api 的 CooldownSoft/CooldownHard/SetLastErrMsg：
//   - 软冷却用 Cooldown(CoolSoft, ...)（固定时长，不叠指数）
//   - 硬冷却用 CooldownUntilTomorrow4AM（与 CodeBuddy 的余额耗尽同语义）
//   - 错误详情不进池（本仓库存 reqlog），只记时间戳，避免两套口径漂移

func (h *Handler) noteUpstreamSoft(uid, reason string) {
	h.cfg.Pool.Cooldown(uid, pool.CoolSoft, h.softCooldown(), reason)
}

func (h *Handler) noteUpstreamHard(uid, reason string) {
	h.cfg.Pool.CooldownUntilTomorrow4AM(uid, reason)
}

// ---- 对外入口 ----

// chatUpstreamProvider 把 Trae / ZCode 请求代理到对应上游。
// raw 是原始 body：Trae 的 buildTraePayload 需要它做 chat_history 直通。
func (h *Handler) chatUpstreamProvider(w http.ResponseWriter, r *http.Request, provider, bare string, raw []byte, st *chatStat) {
	var req OAIRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request", "invalid JSON: "+err.Error())
		return
	}
	// 模型名保持客户端原始串（含 provider 前缀，如 "zcode:glm-4.7"）：newChatStat
	// 已从 body 解析好，这里不再用剥离前缀的 bare 覆盖——否则归档里 trae/zcode 的
	// "glm-4.7" 会与 workbuddy 的同名模型无法区分。
	// 请求体里的 OpenAI stop 在这里解析出来（string 或 []string 都认）。
	// Trae 私有协议没有该字段、WorkBuddy 上游会忽略它，两条链路都由网关侧
	// 自行截断补齐语义（见 stopseq.go）；ZCode 会把它映射成 Anthropic 的
	// stop_sequences 交给上游，网关层只做兜底（同一份 matcher 对已停止的
	// 流是幂等的，不会重复截断）。
	stops := parseStopSequences(req.Stop)
	switch provider {
	case providerTrae:
		h.chatTraeUpstream(w, r, &req, bare, raw, stops, st)
	default:
		h.chatZCodeUpstream(w, r, &req, bare, stops, st)
	}
}

// ---- ZCode（Anthropic Messages 上游，OpenAI 协议对外） ----

func (h *Handler) chatZCodeUpstream(w http.ResponseWriter, r *http.Request, req *OAIRequest, model string, stops []string, st *chatStat) {
	tried := map[string]bool{}
	var lastErr string
	for attempt := 0; attempt < h.maxUpstreamAttempts(); attempt++ {
		acct := h.cfg.Pool.PickExcludingForScope(tried, model, "", providerZCode)
		if acct == nil {
			break
		}
		tried[acct.UID] = true
		if !h.cfg.Pool.Acquire(acct.UID) {
			continue // 在途满：换下一个号
		}
		st.noteAccount(acct)
		started := time.Now()
		res, err := h.tryZCodeOnce(r.Context(), acct, req, model, captureHeaders(r))
		h.cfg.Pool.Release(acct.UID)
		st.attempts++ // 每次真实发出的上游调用都计入，与 CodeBuddy 链路口径一致
		if res == nil {
			// 传输错误没有 usage：计入失败尝试，让用量页看得见重试放大。
			h.recordUpstreamUsage(acct, model, upstreamUsage{}, time.Since(started), false)
			lastErr = errString(err)
			continue
		}
		if res.retry {
			// 401/402/403/429/405 等：同样计为失败尝试（与本仓 CodeBuddy 链路同口径）。
			h.recordUpstreamUsage(acct, model, res.usage, time.Since(started), false)
			lastErr = res.errMsg
			continue
		}
		if res.streaming {
			// 上游（Anthropic 风格）可能不回 stop 语义，网关层再兜一层。
			if len(stops) > 0 {
				res.stream = NewStopFilter(res.stream, stops)
			}
			u := h.streamZCodeUpstream(w, res, model)
			// 一次折算、两处消费（归档 + 台账），与 Trae 链路同构；
			// 详见 zcode_credit_rate.go（ZCode 响应不含 credit，用当日真值分摊）。
			u = zcodeCreditUsage(acct.UID, model, u)
			st.noteStreamUsage(u, "stop")
			st.status = http.StatusOK
			st.outcome = reqlog.OutcomeSuccess
			h.recordUpstreamUsage(acct, model, u, time.Since(started), true)
			return
		}
		res.usage = zcodeCreditUsage(acct.UID, model, res.usage)
		h.recordUpstreamUsage(acct, model, res.usage, time.Since(started), true)
		st.noteUpstreamBody(res.body)
		// 非流式的 token 由上面从响应体解析，折算积分来自 usage 结构，单独并入归档
		// （否则同步请求的 credit 永远是「上游未提供」，只有流式才有值）。
		st.noteCredit(res.usage)
		st.status = http.StatusOK
		st.outcome = reqlog.OutcomeSuccess
		writeJSON(w, http.StatusOK, anthToOAIMessageWithStop(res.body, model, stops))
		return
	}
	// 注意这里的 code 不能一律用 no_available_account。
	//
	// 上游拒绝（405 + code 3012 "unusual activity"）与「账号池空了」
	// 是两回事：账号存在且余额正常，只是对话端点被风控拦了。
	// 用 no_available_account 会让人去狂添账号，而真正的排查方向
	// （客户端签名 / 风控）被盖掉。
	code, msg := "no_available_account", "no available zcode account: "+lastErr
	if zcode.IsBlocked(lastErr) {
		code = "upstream_blocked"
		msg = "上游拒绝了本次请求（风控/客户端签名）：" + lastErr
	} else if zcode.IsModelNotEntitled(lastErr) {
		code = "model_not_entitled"
		msg = "该模型不在本账号套餐授权范围内：" + lastErr
	}
	st.status = http.StatusServiceUnavailable
	st.outcome = reqlog.OutcomeHTTPError
	writeOpenAIError(w, http.StatusServiceUnavailable, code, msg)
}

func (h *Handler) tryZCodeOnce(ctx context.Context, a *auth.Auth, req *OAIRequest, model string, incoming http.Header) (*upstreamResult, error) {
	// 客户端与上游共用的 HTTP 客户端（Config.UpstreamHTTP 的注释明确写了它服务
	// Trae/ZCode 链路）。此前漏传，导致：无视部署方配置的出口代理、无法在测试里
	// 注入假上游、超时口径与 Trae 链路脱节。
	cli := zcode.New(a, zcode.Options{
		Timeout:     5 * time.Minute,
		HTTPClient:  h.cfg.UpstreamHTTP,
		MonitorHost: zcodeMonitorHostFor(a),
	})

	var verifyParam string
	if h.cfg.Captcha != nil {
		if vp, err := h.cfg.Captcha.Get(ctx); err == nil {
			verifyParam = vp
		}
	}

	creq := oaiToZCodeRequest(req, model)
	resp, err := cli.Chat(ctx, creq, verifyParam, incoming)
	if err != nil {
		h.cfg.Pool.NoteError(a.UID)
		h.noteUpstreamSoft(a.UID, "zcode chat error: "+err.Error())
		return nil, err
	}

	// ---- 统一分类 ----
	//
	// 旧实现按 status 逐分支处置，后果是「四类完全不同的问题塌缩成两句话」：
	//   额度耗尽 / 并发超限 / 验证码失效 / 指纹被拦 → 全被当成「429 限流」或「401 未授权」，
	// 且都去冷却账号（其中两类根本不该罚号，冷却只会让可用号变少）。
	// 现在一次读完 body、交给 zcode.Classify 归类，再按类处置。
	//
	// 200 直接放行，不做任何 body 预读（流式响应不能被消费）。
	if resp.StatusCode == http.StatusOK {
		h.cfg.Pool.NoteSuccess(a.UID)
		if req.Stream {
			return &upstreamResult{streaming: true, stream: resp.Body}, nil
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
		resp.Body.Close()
		return &upstreamResult{status: http.StatusOK, body: body, usage: usageFromZCodeBody(body)}, nil
	}

	// 非 200：读一次 body（上限 8KB），后面所有分支共用，且原文进日志/错误信息。
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
	resp.Body.Close()
	b := string(body)
	detail := truncate(strings.TrimSpace(b), 300)
	verdict := zcode.Classify(resp.StatusCode, body)

	// 3012/3009 指纹拦截：先尝试换到 BigModel/Z.AI 的 Anthropic 端点（同一把凭据），
	// 成功就当作成功——这条回落链路旧实现只在 405/406 时试，而实际拦截也常以 403 出现。
	if verdict == zcode.VerdictLoadBlocked {
		if key := zcodeFallbackKey(a); key != "" {
			if alt, aerr := cli.ChatViaKey(ctx, creq, key, a.Source == "zai"); aerr == nil && alt != nil {
				// 回退成功 = 账号完全可用。**只记日志，不碰冷却**：
				// 旧代码在这里调 noteUpstreamSoft("fallback ok")，等于每次成功回退
				// 都把好号罚 600s——成功路径给自己上冷却，是纯 bug。
				logUpstream("zcode %s: %s -> anthropic fallback ok", a.Display(), verdict)
				return h.finishZCodeResponse(ctx, a, req, creq, alt), nil
			} else if aerr != nil {
				logUpstream("zcode %s: %s fallback failed: %v", a.Display(), verdict, aerr)
			}
		}
	}

	switch verdict {
	case zcode.VerdictQuotaExhausted:
		// 额度耗尽：**按模型标死 + 换号**，不冷却账号。
		// 与「限流」的本质区别：等一会儿重试没用，要等周期重置；而账号对其他模型
		// 仍可能可用，所以只能标 (账号, 模型) 而不是整号。
		h.cfg.Pool.BlockModelBackoff(a.UID, model, "zcode quota exhausted")
		return &upstreamResult{retry: true,
			errMsg: "zcode quota exhausted (" + verdict.String() + "): " + detail}, nil

	case zcode.VerdictAdmissionLimit:
		// 准入并发上限（3010）：账号健康、额度充足，只是该模型在途太多。
		// 官方客户端是「等 1s，再等 2s，同账号重试」——这里简化为短退避后换号，
		// 因为网关的换号成本远低于客户端等待，且能立刻找到空闲号。
		h.noteUpstreamSoft(a.UID, "3010 准入并发上限: "+detail)
		return &upstreamResult{retry: true, errMsg: "zcode 3010 admission limit: " + detail}, nil

	case zcode.VerdictCaptcha:
		// 人机验证失效：作废缓存，下次请求重取。**不罚号**（账号没问题）。
		if h.cfg.Captcha != nil {
			h.cfg.Captcha.Invalidate()
		}
		return &upstreamResult{retry: true, errMsg: "zcode captcha rejected: " + detail}, nil

	case zcode.VerdictPlanLacksModel:
		// 套餐不含该模型：长期避让 (账号, 模型)，换号可能有效（别的号套餐不同）。
		h.cfg.Pool.BlockModelBackoff(a.UID, model, "zcode plan lacks model")
		return &upstreamResult{retry: true,
			errMsg: "zcode plan lacks model (" + verdict.String() + "): " + detail}, nil

	case zcode.VerdictUnauthorized:
		// 凭据被拒——这才是真的需要换凭证/重登。
		// 保留上游原文：三种成因（JWT 过期 / secret 轮换后失效 / 端点选错）处置完全不同，
		// 旧实现丢 body 只留一句「401 未授权」，线上完全无法定位。
		h.noteUpstreamHard(a.UID, "401 未授权: "+detail)
		return &upstreamResult{retry: true, errMsg: "zcode 401 unauthorized: " + detail}, nil

	case zcode.VerdictRateLimited:
		// 普通限流：跟随 Retry-After（与 CodeBuddy 链路同口径），没有才回落软冷却基数。
		reason := "429 限流"
		if detail != "" {
			reason += ": " + detail
		}
		if d, ok := upstream.ParseRetryAfter(resp.Header); ok {
			h.cfg.Pool.CooldownSoftRate(a.UID, h.softCooldown(), time.Now().Add(d), reason)
		} else {
			h.noteUpstreamSoft(a.UID, reason)
		}
		return &upstreamResult{retry: true, errMsg: "zcode 429 rate limited: " + detail}, nil
	}

	// 未分类（4xx/5xx 其他）：喂熔断计数，原文透传。
	h.cfg.Pool.NoteError(a.UID)
	return &upstreamResult{retry: true,
		errMsg: fmt.Sprintf("zcode http %d: %s", resp.StatusCode, detail)}, nil
}

// zcodeFallbackKey 返回可用于 BigModel / Z.AI Anthropic 端点的 key。
//
// 优先级：
//
//  1. SignCredential —— "<keyId>.<secret>"，实测在 open.bigmodel.cn /
//     api.z.ai 的 /api/anthropic/v1/messages 上直接 200；
//  2. APIKey —— 面板手填的 api_key（同样是 key 形态）。
//
// 返回空串表示两条路都没有可用凭据，此时调用方应把原始 3012 透出。
func zcodeFallbackKey(a *auth.Auth) string {
	if a == nil {
		return ""
	}
	if k := a.SignCredential(); k != "" {
		return k
	}
	return a.APIKeyValue()
}

// finishZCodeResponse 把 fallback 拿到的 Anthropic 响应整理成 upstreamResult。
func (h *Handler) finishZCodeResponse(ctx context.Context, a *auth.Auth, req *OAIRequest,
	creq *zcode.ChatRequest, resp *http.Response) *upstreamResult {
	defer resp.Body.Close()
	h.cfg.Pool.NoteSuccess(a.UID)
	if req.Stream {
		return &upstreamResult{streaming: true, stream: resp.Body}
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	return &upstreamResult{status: http.StatusOK, body: body, usage: usageFromZCodeBody(body)}
}

// usageFromZCodeBody 从 Anthropic Messages 非流式响应体取 usage。
func usageFromZCodeBody(body []byte) upstreamUsage {
	var wrap struct {
		Usage json.RawMessage `json:"usage"`
	}
	if json.Unmarshal(body, &wrap) != nil {
		return upstreamUsage{}
	}
	return parseZCodeUsage(wrap.Usage)
}

// streamZCodeUpstream 将 Anthropic SSE 转换为 OpenAI SSE 并写回客户端。
// 返回本次流观测到的 usage（Anthropic 把 input 放在 message_start、
// output 放在末尾 message_delta，必须整段读完才能拿到终值）。
func (h *Handler) streamZCodeUpstream(w http.ResponseWriter, res *upstreamResult, model string) upstreamUsage {
	defer res.stream.Close()
	flusher := startOpenAIStream(w)
	id := "chatcmpl-" + newID("")
	created := time.Now().Unix()

	sc := bufio.NewScanner(res.stream)
	sc.Buffer(make([]byte, 0, 64*1024), 8<<20)
	sentRole := false
	var got upstreamUsage
	// Anthropic 的 tool_use 分片：content_block_start 给 id/name，紧随的一串
	// input_json_delta 拼出 JSON 参数，content_block_stop 收口。按块下标累积。
	type pendingTool struct {
		id   string
		name string
		args strings.Builder
	}
	tools := map[int]*pendingTool{}
	var toolOrder []int
	finish := "stop"
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var ev map[string]any
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		// usage 帧：message_start.message.usage / message_delta.usage。
		// 两处都取、后帧覆盖前帧（Anthropic 的 output_tokens 会随生成递增）。
		var ue struct {
			Usage   json.RawMessage `json:"usage"`
			Message struct {
				Usage json.RawMessage `json:"usage"`
			} `json:"message"`
		}
		if json.Unmarshal([]byte(data), &ue) == nil {
			if len(ue.Message.Usage) > 0 {
				got = mergeZCodeUsage(got, parseZCodeUsage(ue.Message.Usage))
			}
			if len(ue.Usage) > 0 {
				got = mergeZCodeUsage(got, parseZCodeUsage(ue.Usage))
			}
		}
		typ, _ := ev["type"].(string)
		switch typ {
		case "content_block_start":
			idx := asInt(ev["index"])
			blk, _ := ev["content_block"].(map[string]any)
			if bt, _ := blk["type"].(string); bt == "tool_use" {
				pt := &pendingTool{}
				pt.id, _ = blk["id"].(string)
				pt.name, _ = blk["name"].(string)
				// 有些上游会在 start 帧直接给完整 input，先塞进去。
				if in, ok := blk["input"]; ok && in != nil {
					if raw, err := json.Marshal(in); err == nil && string(raw) != "{}" && string(raw) != "null" {
						pt.args.Write(raw)
					}
				}
				tools[idx] = pt
				toolOrder = append(toolOrder, idx)
			}
		case "content_block_delta":
			delta, _ := ev["delta"].(map[string]any)
			dt, _ := delta["type"].(string)
			if dt == "input_json_delta" {
				idx := asInt(ev["index"])
				if pt := tools[idx]; pt != nil {
					if pj, ok := delta["partial_json"].(string); ok {
						pt.args.WriteString(pj)
					}
				}
				continue
			}
			text, _ := delta["text"].(string)
			if text == "" {
				continue
			}
			writeSSE(w, OAIChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
				Choices: []OAIChoice{{Index: 0, Delta: &OAIDelta{Role: roleOnce(&sentRole), Content: text}}}})
			if flusher != nil {
				flusher.Flush()
			}
		case "message_delta":
			if d, ok := ev["delta"].(map[string]any); ok {
				if sr, _ := d["stop_reason"].(string); sr != "" {
					switch sr {
					case "max_tokens":
						finish = "length"
					case "tool_use":
						finish = "tool_calls"
					default:
						finish = "stop"
					}
				}
			}
		}
	}
	// 收口：把累积到的 tool_use 块转成 OpenAI 的 tool_calls 增量帧。
	if len(toolOrder) > 0 {
		for i, idx := range toolOrder {
			pt := tools[idx]
			if pt == nil || pt.name == "" {
				continue
			}
			args := strings.TrimSpace(pt.args.String())
			if args == "" {
				args = "{}"
			}
			cid := pt.id
			if cid == "" {
				cid = "call_" + newID("")
			}
			callIdx := i
			emitToolCallChunks(w, flusher, id, model, created, &sentRole, []oaiToolCall{{
				Index: &callIdx,
				ID:    cid,
				Type:  "function",
				Function: oaiToolCallFnSpec{
					Name:      pt.name,
					Arguments: args,
				},
			}})
		}
		if finish == "stop" {
			finish = "tool_calls"
		}
	}
	writeSSE(w, OAIChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []OAIChoice{{Index: 0, Delta: &OAIDelta{}, FinishReason: &finish}}})
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
	return got
}

// ---- Trae（SSE 上游，OpenAI 协议对外） ----

func (h *Handler) chatTraeUpstream(w http.ResponseWriter, r *http.Request, req *OAIRequest, model string, raw []byte, stops []string, st *chatStat) {
	tried := map[string]bool{}
	var lastErr string
	for attempt := 0; attempt < h.maxUpstreamAttempts(); attempt++ {
		acct := h.cfg.Pool.PickExcludingForScope(tried, model, "", providerTrae)
		if acct == nil {
			break
		}
		tried[acct.UID] = true
		if !h.cfg.Pool.Acquire(acct.UID) {
			continue
		}
		st.noteAccount(acct)
		started := time.Now()
		res, err := h.tryTraeOnce(r.Context(), acct, req, model, raw)
		h.cfg.Pool.Release(acct.UID)
		st.attempts++
		if res == nil {
			// Trae 是纯流式上游：建连失败即无 usage，计为失败尝试。
			h.recordUpstreamUsage(acct, model, upstreamUsage{}, time.Since(started), false)
			lastErr = errString(err)
			continue
		}
		if res.retry {
			if res.cancel != nil {
				res.cancel()
			}
			h.recordUpstreamUsage(acct, model, res.usage, time.Since(started), false)
			lastErr = res.errMsg
			continue
		}
		// 上游只有流式，但客户端要什么协议就给什么协议：
		//   stream:true  → 逐帧转成 OpenAI SSE；
		//   stream:false → 网关侧聚合后回一条 OpenAI 非流式 JSON。
		// 旧实现无论客户端要什么一律回 SSE，Content-Type 也是 text/event-stream，
		// 于是 stream:false 的调用方（IDE 插件 / agent / 批处理脚本）拿到流式体
		// 后解析失败——这是「Trae 接口不能用」最直接的一条。
		var u upstreamUsage
		if res.streaming {
			if req.Stream {
				u = h.streamTraeUpstream(w, res, model, stops)
			} else {
				u = h.collectTraeUpstream(w, res, model, stops)
			}
		}
		if res.cancel != nil {
			res.cancel()
		}
		// 一次折算、两处消费：归档（noteStreamUsage）与台账（recordUpstreamUsage）
		// 共用同一个估算值，避免两边分别算出现漂移。
		u = traeCreditUsage(model, u, true)
		h.recordUpstreamUsage(acct, model, u, time.Since(started), true)
		st.noteStreamUsage(u, "stop")
		st.status = http.StatusOK
		// 上游提前断流时不能记成功：面板上要能看出「这次回答是残缺的」，
		// 否则用户只能看到账号正常、用量正常，却找不到任何异常痕迹。
		st.outcome = reqlog.OutcomeSuccess
		if res.interrupted {
			st.outcome = reqlog.OutcomeStreamError
		}
		return
	}
	// 错误分类（与 traecode2api 口径一致）：模型不存在 / 权益不足要给出可诊断的
	// 状态码，不能一律包成 503 no_available_account —— 否则用户会去狂添账号，
	// 而真正的问题（模型名打错 / 该模型不在账号套餐里）被盖掉。
	if isTraeModelNotFound(lastErr) {
		st.status = http.StatusNotFound
		st.outcome = reqlog.OutcomeHTTPError
		writeOpenAIError(w, http.StatusNotFound, "model_not_found",
			"trae model is not available for this account: "+lastErr)
		return
	}
	if isTraeEntitlementError(lastErr) {
		st.status = http.StatusForbidden
		st.outcome = reqlog.OutcomeHTTPError
		writeOpenAIError(w, http.StatusForbidden, "model_not_entitled",
			"trae model endpoint is not enabled for this account: "+lastErr)
		return
	}
	st.status = http.StatusServiceUnavailable
	st.outcome = reqlog.OutcomeHTTPError
	writeOpenAIError(w, http.StatusServiceUnavailable, "no_available_account",
		"no available trae account: "+lastErr)
}

// isTraeModelNotFound 识别上游明确的模型不存在/未登记错误（4023 / model is unknown）。
func isTraeModelNotFound(msg string) bool {
	m := strings.ToLower(msg)
	return strings.Contains(m, "4023") || strings.Contains(m, "model is unknown") ||
		strings.Contains(m, "unknown model") || strings.Contains(m, "model not found")
}

// isTraeEntitlementError 识别端点或账号权益不足（4011 / 4001）。
func isTraeEntitlementError(msg string) bool {
	m := strings.ToLower(msg)
	return strings.Contains(m, "4011") || strings.Contains(m, "4001") ||
		strings.Contains(m, "not entitled") || strings.Contains(m, "config lookup")
}

func (h *Handler) tryTraeOnce(ctx context.Context, a *auth.Auth, req *OAIRequest, model string, raw []byte) (*upstreamResult, error) {
	attemptCtx, cancel := context.WithCancel(ctx)
	// 与 ZCode 链路对齐：共用部署方配置的上游客户端（出口代理 / 超时口径一致），
	// 并让测试可以注入假上游。此前漏传导致 Trae 只能走自建 transport。
	var hopts trae.Options
	if h.cfg.UpstreamHTTP != nil {
		hopts.HTTPClient = h.cfg.UpstreamHTTP
	}
	cli := trae.New(a, hopts)
	// 凭证过期则先换票（ExchangeToken 滚动刷新），失败不阻断——上游会用 401 回我们。
	_ = cli.EnsureFreshToken(attemptCtx, a, 10*time.Minute)
	// DLL 内置模型（glm-5.3-flashx / deepseek-v4.1-flash / step-5-preview 等）
	// 走独立的 SOLO 链 llm_utils_chat：/api/ide/v1/chat 不认这些 model_name
	// （实测 code 4023 model is unknown）。模型名原样透传，不降级 seed_m8。
	var dataCh <-chan string
	var errCh <-chan error
	var err error
	if useSoloModelChain(model) {
		dataCh, errCh, err = cli.SoloChatStream(attemptCtx, buildSoloPayload(req, model), nil)
	} else {
		dataCh, errCh, err = cli.ChatStream(attemptCtx, buildTraePayload(req, model), nil)
	}
	if err != nil {
		cancel()
		h.cfg.Pool.NoteError(a.UID)
		h.noteUpstreamSoft(a.UID, "trae chat error: "+err.Error())
		return nil, err
	}
	res := &upstreamResult{streaming: true, stream: &chanReader{ch: dataCh, err: errCh}, cancel: cancel}
	if queued, reader := preflightTraeStream(res.stream); queued {
		_ = reader.Close()
		h.noteUpstreamSoft(a.UID, "trae request queued; rotate account")
		return &upstreamResult{retry: true, errMsg: "trae request queued", cancel: cancel}, nil
	} else {
		res.stream = reader
	}
	return res, nil
}

// preflightTraeStream 读取 Trae SSE 的首段，直到看到首个 output 或明确的排队状态。
// 排队中的免费账号会先返回 progress/queue 事件并保持连接；若直接交给下游，
// 请求会长时间占住该账号。命中排队后返回 queued=true，由外层轮换账号。
// 已读取的首段会通过 prefixReadCloser 放回，正常 output 不会丢失。
func preflightTraeStream(src io.ReadCloser) (queued bool, out io.ReadCloser) {
	br := bufio.NewReaderSize(src, 64*1024)
	var prefix bytes.Buffer
	event := ""
	for {
		line, err := br.ReadString('\n')
		line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
		if err != nil && line == "" {
			return false, &prefixReadCloser{Reader: io.MultiReader(bytes.NewReader(prefix.Bytes()), br), src: src}
		}
		prefix.WriteString(line)
		prefix.WriteByte('\n')
		if strings.HasPrefix(line, "event:") {
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			if traeQueueEvent(event, "") {
				return true, src
			}
			continue
		}
		if !strings.HasPrefix(line, "data:") && !strings.HasPrefix(line, "{") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if event == "output" || event == "done" || traeUsageEvent(event, data) || traeDeltaAvailable(data) {
			return false, &prefixReadCloser{Reader: io.MultiReader(bytes.NewReader(prefix.Bytes()), br), src: src}
		}
		if traeQueueEvent(event, data) {
			return true, src
		}
		event = ""
		if err != nil {
			return false, &prefixReadCloser{Reader: io.MultiReader(bytes.NewReader(prefix.Bytes()), br), src: src}
		}
	}
}

func traeDeltaAvailable(data string) bool {
	_, _, ok := traeDelta(data)
	return ok
}

// traeQueueEvent 同时覆盖事件名和 JSON 文案。不同 Trae 版本分别使用
// queue/waiting 事件，或 progress_notice 中的 status/message/content 字段。
func traeQueueEvent(event, data string) bool {
	ev := strings.ToLower(strings.TrimSpace(event))
	// timing_cost 的 queue_timing 是耗时字段，不是排队状态；只有明确的
	// 队列事件，或 progress/error/status 类事件里的状态文案才触发轮换。
	if ev == "queue" || ev == "queued" || ev == "queueing" || ev == "waiting" || ev == "wait_queue" {
		return true
	}
	if ev != "progress_notice" && ev != "waiting" && ev != "status" && ev != "error" && ev != "message" {
		return false
	}
	s := strings.ToLower(strings.TrimSpace(data))
	for _, marker := range []string{
		"queued", "queueing", "waiting", "wait_queue", "queue position",
		"request queue", "排队", "排隊", "队列", "隊列", "等待处理", "等待中",
		`\"status\":\"queue\"`, `\"status\":\"waiting\"`,
		`\"state\":\"queue\"`, `\"state\":\"waiting\"`,
	} {
		if strings.Contains(s, marker) {
			return true
		}
	}
	return false
}

type prefixReadCloser struct {
	io.Reader
	src io.Closer
}

func (r *prefixReadCloser) Close() error { return r.src.Close() }

// traeDelta 从 Trae 的 output 事件载荷取出正文与思考链。
//
// 逆向结论（2026-10-07 线上抓包，真实帧如下）：
//
//	event: output
//	data: {"response":"OK","reasoning_content":"……","tool_calls":null,
//	       "multimodal_contents":null,"phase":null}
//
//	event: token_usage
//	data: {"name":"ide_prompts","prompt_tokens":95,"completion_tokens":49,
//	       "total_tokens":144,"reasoning_tokens":0,"cache_read_input_tokens":0,...}
//
//	event: progress_notice
//	data: {"name":"construct_prompt","content":"Building prompt: ide_prompts"}
//
// response 是正文，reasoning_content 是思考链（与正文分列，两者都在 output 帧里）。
//
// 返回值 ok=false 表示这一帧既不是正文也不是思考链（progress_notice / metadata /
// prompt / done 等控制帧）。调用方必须跳过，绝不能像旧实现那样对任意 data 做
// 「content/text 字段兜底」，否则会命中
// {"name":"construct_prompt","content":"Building prompt: ide_prompts"} 的 content，
// 把上游的进度提示当成回复正文吐给客户端——这是「回复开头挂着
// Building prompt: ide_prompts」的唯一来源。
func traeDelta(data string) (content, reasoning string, ok bool) {
	var m struct {
		Response         *string `json:"response"`
		ReasoningContent *string `json:"reasoning_content"`
	}
	if json.Unmarshal([]byte(data), &m) != nil {
		return "", "", false
	}
	if m.Response == nil && m.ReasoningContent == nil {
		return "", "", false
	}
	if m.Response != nil {
		content = *m.Response
	}
	if m.ReasoningContent != nil {
		reasoning = *m.ReasoningContent
	}
	return content, reasoning, true
}

// traePostDoneDrainLines event: done 之后最多继续读多少行。
//
// 旧实现见到 done 立刻收流：上游把长回答分成多个 output 帧时，只要 done 早到
// 一帧，剩下的正文就整段丢掉（观感「回答写到一半停了」）。但也不能无脑读到 EOF
// ——上游可能带着心跳把连接留住，客户端会一直等。故给有界窗口：done 之后的
// 残帧照收，行数到顶或超过 traePostDoneGrace 未结束即收流。
const traePostDoneDrainLines = 64

// traePostDoneGrace done 之后等待残帧的最长时间，超时即取消上游请求。
// 正常残帧是紧随 done 到达的，几百毫秒就够；给到 2s 是为了容忍上行拥塞，
// 同时避免 upstream 挂住连接时把客户端一起拖死。
const traePostDoneGrace = 2 * time.Second

// traeKeepAliveInterval 网关向客户端发送「空心跳帧」的间隔。
//
// Trae 上游在真正出字之前可能长时间只发 progress_notice（实测「Building
// prompt」阶段能持续 1~2 分钟）。此期间网关一个字节都不写，客户端的空闲
// 看门狗会判定「响应停滞」并主动取消（Trae 客户端显示「已取消」、
// WorkBuddy 客户端显示「用户已取消」，用量 0 tok）——用户看到的就是
// 「话还没说完就停了」。
//
// 心跳用**合法的 OpenAI 空 delta 帧**而不是 SSE 注释（": keep-alive"）：
// 注释行虽符合 SSE 规范，但不少 OpenAI 兼容客户端会把它当成畸形帧。
//
// 变量而非常量：测试需要把窗口缩短到毫秒级才能验证保活行为。
var traeKeepAliveInterval = 10 * time.Second

// traeUpstreamIdleCap 上游连续多久没有任何帧就判定为「假死」并主动取消。
//
// 心跳只解决「客户端不取消」，解决不了「上游永不回应」：若上游挂住连接，
// 客户端会一直收到心跳、一直转圈，用户等不到任何回答也等不到错误。
// 超过该上限即取消上游请求（Read 立刻返回），按中断收尾并上报 length。
var traeUpstreamIdleCap = 3 * time.Minute

// traeInterruptedTotal 累计观测到的「上游提前断流」次数（进程内计数）。
// 用于线上判断该现象是偶发还是常态；日志里与单次请求的中断明细一起看。
var traeInterruptedTotal atomic.Int64

// streamTraeUpstream 将 Trae 的 SSE 数据帧转换为 OpenAI SSE。
// 返回本次流观测到的 usage（上游在 event: token_usage 帧里给计数）。
//
// 事件分派严格按事件名（与 traecode2api 的 handleTraeEvent 同口径）：
//
//	output         → response 进正文；reasoning_content 丢弃（OpenAI 的 delta
//	                 没有思考链的标准槽位，塞进 content 会让正文混入思维过程）
//	token_usage    → 取计数，不进正文
//	error          → 把上游错误文案附在正文并收流
//	done           → 标记已收尾，继续吃掉紧随其后的残帧（有界），随后收流
//	其余（progress_notice / metadata / prompt）→ 纯控制帧，一律丢弃
//
// 三条「不能停」的保护：
//
//  1. **首帧立即下发**：上游首个 output 可能几十秒后才到，网关先把 role 帧
//     写出去并 flush，客户端的 TTFB 看门狗就不会误判。
//  2. **空闲心跳**：等待期每 traeKeepAliveInterval 写一个空 delta 帧，
//     避免客户端「响应停滞 → 取消」（截图上「已取消 / 用量 0 tok」的成因）。
//  3. **上游假死上限**：连续 traeUpstreamIdleCap 无任何帧即取消上游请求。
//
// 退出原因决定 finish_reason：正常收尾为 stop / tool_calls；上游在**没有 done
// 帧**的情况下结束（连接被掐 sc.Err() != nil，或上游「干净 EOF」直接关闭流）
// 则为 length，并把 res.interrupted 置位。这一点是「任务没完成、输出没写完
// 就停」的关键：旧实现一律上报 stop，客户端于是把半截回答当成正常完成。
func (h *Handler) streamTraeUpstream(w http.ResponseWriter, res *upstreamResult, model string, stops []string) upstreamUsage {
	defer res.stream.Close()
	flusher := startOpenAIStream(w)
	id := "chatcmpl-" + newID("")
	created := time.Now().Unix()

	sentRole := false
	var got upstreamUsage
	matcher := newStopMatcher(stops)
	// Trae 的工具调用是内联标记文本，先剥离再进 stop 匹配（标记不该出现在正文里）。
	fnp := newTraeFnParser()
	var toolCalls []oaiToolCall
	emittedLen := 0 // 已吐出正文的字节数（仅用于日志与「有没有输出」判定）
	// writeChunk 发一帧并立即 flush。所有写出都必须走它：不 flush 的字节
	// 会滞留在缓冲里，客户端的空闲看门狗看到的仍是「没有响应」。
	writeChunk := func(delta *OAIDelta, finish *string) {
		writeSSE(w, OAIChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []OAIChoice{{Index: 0, Delta: delta, FinishReason: finish}}})
		if flusher != nil {
			flusher.Flush()
		}
	}
	emit := func(s string) {
		if s == "" {
			return
		}
		emittedLen += len(s)
		writeChunk(&OAIDelta{Role: roleOnce(&sentRole), Content: s}, nil)
	}

	// 首帧：立刻把 role 帧写出去。上游在「Building prompt」阶段可能整分钟
	// 不发 output，客户端在此期间收不到任何字节就会超时取消。
	writeChunk(&OAIDelta{Role: roleOnce(&sentRole)}, nil)

	sc := bufio.NewScanner(res.stream)
	sc.Buffer(make([]byte, 0, 64*1024), 8<<20)
	pendingEvent := ""
	stop := false    // 硬停（error 帧 / stop 命中）
	sawDone := false // 收到 event: done
	sawError := false
	sawOutput := false // 收到过 output 帧（判定「上游到底有没有开始回答」）
	drain := 0
	// seenEvents 记录上游发过的事件名序列（去重计数），仅用于「无输出就结束」
	// 时的诊断日志：线上排查需要区分「上游只发 progress 就关流」与
	// 「上游发了 output 但我们判成控制帧」这两种完全不同的成因。
	var seenEvents []string
	noteEvent := func(name string) {
		if len(seenEvents) < 16 {
			seenEvents = append(seenEvents, name)
		}
	}
	// done 之后的看门狗：到点直接取消上游请求，让阻塞中的读立刻 EOF。
	// （chanReader 的 Close 只是本地信号，真正解开 Read 要靠 RST 连接。）
	var doneTimer *time.Timer
	defer func() {
		if doneTimer != nil {
			doneTimer.Stop()
		}
	}()
	armDoneTimer := func() {
		if doneTimer != nil || res.cancel == nil {
			return
		}
		doneTimer = time.AfterFunc(traePostDoneGrace, res.cancel)
	}

	// 行读取与「空闲保活 / 假死上限」并行：sc.Scan() 是阻塞的，等待期间
	// 什么都不会写，因此心跳必须由另一条 goroutine 定时触发。
	// 用 timer 而不是固定 ticker：每次真有帧到达就重置，避免一边正常出字
	// 一边白发心跳。
	//
	// 退出协议：主线关闭 stopCh 后 goroutine 立即返回（不再阻塞在发送上）。
	// 上游阻塞在 Scan 里时由 res.cancel() 打断——那正是「假死上限」的用法。
	// sc 只允许由该 goroutine 触碰（含 sc.Err()），否则与之构成数据竞争。
	type scanResult struct {
		line string
		err  error
		done bool
	}
	lines := make(chan scanResult, 8)
	stopCh := make(chan struct{})
	defer close(stopCh)
	var scanErr error
	go func() {
		defer close(lines)
		for sc.Scan() {
			select {
			case lines <- scanResult{line: sc.Text()}:
			case <-stopCh:
				return
			}
		}
		// Scan 结束：把读错误一并交出（sc 归本 goroutine 独占）。
		err := sc.Err()
		select {
		case lines <- scanResult{err: err, done: true}:
		case <-stopCh:
		}
	}()
	idleTimer := time.NewTimer(traeKeepAliveInterval)
	defer idleTimer.Stop()
	lastActivity := time.Now()
	for !stop {
		var line string
		select {
		case r, ok := <-lines:
			if !ok {
				// channel 已关闭且没收到 done 标记：上游流结束。
				stop = true
				continue
			}
			if r.done {
				scanErr = r.err
				stop = true
				continue
			}
			line = r.line
			lastActivity = time.Now()
			idleTimer.Reset(traeKeepAliveInterval)
		case <-idleTimer.C:
			// 上游这段时间没有新帧：给客户端发一个空 delta 保活，
			// 同时检查上游是否已经假死到该放弃的程度。
			if time.Since(lastActivity) >= traeUpstreamIdleCap {
				log.Printf("WARN trae upstream idle %.0fs model=%s emitted=%dB; abort",
					time.Since(lastActivity).Seconds(), model, emittedLen)
				if res.cancel != nil {
					res.cancel()
				}
				stop = true
				continue
			}
			if !sawDone {
				writeChunk(&OAIDelta{}, nil)
			}
			idleTimer.Reset(traeKeepAliveInterval)
			continue
		}
		if sawDone {
			drain++
			if drain > traePostDoneDrainLines {
				stop = true
				continue
			}
		}
		// event: 与 data: 成对出现；事件名先到，先暂存再与紧随的 data 配对。
		if strings.HasPrefix(line, "event:") {
			pendingEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		data := ""
		switch {
		case strings.HasPrefix(line, "data:"):
			data = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		case strings.HasPrefix(line, "{"):
			data = strings.TrimSpace(line)
		}
		if data == "" || data == "[DONE]" {
			pendingEvent = ""
			continue
		}
		// usage 帧不是正文。事件名缺失时按载荷特征兜底（含 prompt_tokens）。
		if traeUsageEvent(pendingEvent, data) {
			got = parseTraeUsage(data)
			pendingEvent = ""
			continue
		}
		name := pendingEvent
		pendingEvent = ""
		if name != "" {
			noteEvent(name)
		}
		switch name {
		case "done":
			// 不在这里收流：done 之后仍可能有 output 残帧（见 traePostDoneDrainLines）。
			// 只给有界窗口，到点由看门狗取消上游请求。
			sawDone = true
			armDoneTimer()
		case "error":
			var e struct {
				Message string `json:"message"`
			}
			_ = json.Unmarshal([]byte(data), &e)
			if e.Message != "" {
				emit("\n[trae error] " + e.Message)
			}
			sawError = true
			stop = true
		case "output":
			raw, _, ok := traeDelta(data)
			if !ok || raw == "" {
				continue
			}
			sawOutput = true
			// 先剥工具标记：命中则进 toolCalls，正文只剩标记之外的部分。
			content, calls := fnp.feed(raw)
			if len(calls) > 0 {
				toolCalls = append(toolCalls, calls...)
			}
			if content == "" {
				continue
			}
			// stop 截断（Trae 私有协议没有 stop 字段，只能网关侧补）。
			if matcher != nil {
				out, hit := matcher.feed(content)
				emit(out)
				if hit {
					stop = true
				}
				continue
			}
			emit(content)
		default:
			// 控制帧（progress_notice / metadata / prompt / 心跳）不进正文。
		}
	}
	// 收流：把 fnParser 与 stop matcher 各自 hold 住的尾巴补发（否则最后几字被吞）。
	if ftail, fcalls := fnp.flush(); true {
		if len(fcalls) > 0 {
			toolCalls = append(toolCalls, fcalls...)
		}
		if ftail != "" {
			if matcher != nil {
				out, _ := matcher.feed(ftail)
				emit(out)
			} else {
				emit(ftail)
			}
		}
	}
	if matcher != nil {
		emit(matcher.flush())
	}
	// 工具调用：OpenAI 规范的 tool_calls 增量帧，随后 finish_reason=tool_calls。
	// index 用真实序号（parseTraeCalls 已按数组顺序写好），否则多工具会被客户端
	// 当成同一个调用而互相覆盖。
	if len(toolCalls) > 0 {
		for i := range toolCalls {
			idx := i
			toolCalls[i].Index = &idx
		}
		emitToolCallChunks(w, flusher, id, model, created, &sentRole, toolCalls)
	}
	finish := "stop"
	if len(toolCalls) > 0 {
		finish = "tool_calls"
	}
	// 中断判定：没收到 done / error 就结束了 —— 不论是被掐（sc.Err() != nil）
	// 还是上游「干净 EOF」直接关流，都算残缺。
	//
	// 干净 EOF 这一支必须单独判：实测上游把连接正常关闭却从不下发 done 时
	// sc.Err() 为 nil，旧判定漏掉它，于是照样上报 stop，客户端把半截回答
	// 当成正常完成（截图症状）。
	//
	// 例外：已经拿到完整工具调用时保持 tool_calls——改报 length 会让客户端
	// 丢掉这些调用并重试，比「正文少一截」代价更大。
	if !sawDone && !sawError {
		res.interrupted = true
		traeInterruptedTotal.Add(1)
		if finish != "tool_calls" {
			finish = "length"
		}
		log.Printf("WARN trae stream ended without done model=%s emitted=%dB output_frames=%v tool_calls=%d finish=%s events=%v err=%v",
			model, emittedLen, sawOutput, len(toolCalls), finish, seenEvents, scanErr)
	}
	writeChunk(&OAIDelta{}, &finish)
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
	return got
}

// traeFinishFor 按「是否命中工具 / 流是否被上游提前掐断」给出 finish_reason。
// 中断时优先 length（OpenAI 语义里 length 表示「还能继续、但这次到此为止」，
// 客户端据此续写或重试；写成 stop 会让 agent 判定任务已完成并停下）。
// 已拿到完整工具调用时仍报 tool_calls（客户端要按它去执行工具）。
func traeFinishFor(hasTools, interrupted bool) string {
	if hasTools {
		return "tool_calls"
	}
	if interrupted {
		return "length"
	}
	return "stop"
}

// collectTraeUpstream 读完整条 Trae 流并聚合成一条 OpenAI 非流式响应。
//
// 上游只有 SSE，而 OpenAI 的 stream:false 语义要求「一次 JSON、一次 usage」。
// 网关在这里聚合（与 traecode2api 的 traeStream.collect 同口径），返回的
// Content-Type 是 application/json，调用方可直接 json.Unmarshal。
//
// 与流式不同，非流式没有「中间帧」可发：客户端拿到的是一条完整 JSON，
// HTTP 语义本身就允许服务端先思考再响应，故这里不做心跳（改动会破坏
// 纯 JSON 契约）。上游假死由 tryTraeOnce 的请求 context 与传输层超时兜底。
func (h *Handler) collectTraeUpstream(w http.ResponseWriter, res *upstreamResult, model string, stops []string) upstreamUsage {
	defer res.stream.Close()
	var content, reasoning strings.Builder
	var got upstreamUsage
	matcher := newStopMatcher(stops)
	// 工具标记解析（见 streamTraeUpstream 的同类说明）。
	fnp := newTraeFnParser()
	var toolCalls []oaiToolCall
	stopHit := false
	errMsg := ""
	sawDone := false
	sawError := false
	drain := 0
	sc := bufio.NewScanner(res.stream)
	sc.Buffer(make([]byte, 0, 64*1024), 8<<20)
	pendingEvent := ""
	stop := false
	for !stop && sc.Scan() {
		if sawDone {
			drain++
			if drain > traePostDoneDrainLines {
				break
			}
		}
		line := sc.Text()
		if strings.HasPrefix(line, "event:") {
			pendingEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		data := ""
		switch {
		case strings.HasPrefix(line, "data:"):
			data = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		case strings.HasPrefix(line, "{"):
			data = strings.TrimSpace(line)
		}
		if data == "" || data == "[DONE]" {
			pendingEvent = ""
			continue
		}
		if traeUsageEvent(pendingEvent, data) {
			got = parseTraeUsage(data)
			pendingEvent = ""
			continue
		}
		name := pendingEvent
		pendingEvent = ""
		switch name {
		case "done":
			sawDone = true
		case "error":
			var e struct {
				Message string `json:"message"`
			}
			if json.Unmarshal([]byte(data), &e) == nil {
				errMsg = e.Message
			}
			sawError = true
			stop = true
		case "output":
			raw, r, ok := traeDelta(data)
			if !ok {
				continue
			}
			if r != "" {
				reasoning.WriteString(r)
			}
			if stopHit {
				continue
			}
			// 先剥工具标记，命中则累积 toolCalls，正文只留其余部分。
			c, calls := fnp.feed(raw)
			if len(calls) > 0 {
				toolCalls = append(toolCalls, calls...)
			}
			if c == "" {
				continue
			}
			// stop 截断（Trae 上游没有该语义，网关侧补）。
			if matcher != nil {
				emit, hit := matcher.feed(c)
				content.WriteString(emit)
				if hit {
					stopHit = true
				}
				continue
			}
			content.WriteString(c)
		default:
			// 控制帧不进正文。
		}
	}
	if ftail, fcalls := fnp.flush(); true {
		if len(fcalls) > 0 {
			toolCalls = append(toolCalls, fcalls...)
		}
		if ftail != "" {
			if matcher != nil {
				emit, _ := matcher.feed(ftail)
				content.WriteString(emit)
			} else {
				content.WriteString(ftail)
			}
		}
	}
	if matcher != nil && !stopHit {
		content.WriteString(matcher.flush())
	}
	// 非流式同样不能虚报成功：上游没 done 也没 error 就结束（被掐或干净 EOF），
	// 说明聚合到的是半截回答，finish_reason 用 length 并记 WARN。
	interrupted := !sawDone && !sawError
	if interrupted {
		res.interrupted = true
		traeInterruptedTotal.Add(1)
		log.Printf("WARN trae collect interrupted model=%s collected=%dB tool_calls=%d err=%v",
			model, content.Len(), len(toolCalls), sc.Err())
	}
	msg := map[string]any{"role": "assistant", "content": content.String()}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	if reasoning.Len() > 0 {
		msg["reasoning_content"] = reasoning.String()
	}
	if errMsg != "" {
		msg["content"] = strings.TrimRight(content.String(), "\n") + "\n[trae error] " + errMsg
	}
	var prompt, completion int64
	if got.HasPrompt {
		prompt = got.PromptTokens
	}
	if got.HasCompletion {
		completion = got.Completion
	}
	var total int64
	if t, ok := got.total(); ok {
		total = t
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      "chatcmpl-" + newID(""),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": traeFinishFor(len(toolCalls) > 0, interrupted)}},
		"usage": map[string]any{
			"prompt_tokens":     prompt,
			"completion_tokens": completion,
			"total_tokens":      total,
		},
	})
	return got
}

// ---- 辅助（平移自 gw2api） ----

// startOpenAIStream 写出 SSE 响应头并返回 flusher（不支持 flush 时返回 nil）。
func startOpenAIStream(w http.ResponseWriter) http.Flusher {
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	f, _ := w.(http.Flusher)
	return f
}

// roleOnce 首次返回 "assistant"，其后返回空串（OpenAI 流式约定：role 只发一次）。
func roleOnce(sent *bool) string {
	if *sent {
		return ""
	}
	*sent = true
	return "assistant"
}

func writeSSE(w io.Writer, v any) {
	b, _ := json.Marshal(v)
	_, _ = io.WriteString(w, "data: ")
	_, _ = w.Write(b)
	_, _ = io.WriteString(w, "\n\n")
}

// newID 生成响应 id（chatcmpl- 后缀）。
func newID(prefix string) string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return prefix + hex.EncodeToString(b[:])
}

// extractTraeText 从 Trae SSE 数据帧中取文本。
//
// 字段优先级来自真实事件形态：output 事件的正文在 `response`
// （traecode2api 实测口径，见其 traeOutput.Response），思考链在
// `reasoning_content`；其余字段是历史形态兜底，避免上游换名后一片空白。
func extractTraeText(data string) string {
	var m map[string]any
	if json.Unmarshal([]byte(data), &m) != nil {
		return ""
	}
	if v, ok := m["response"].(string); ok && v != "" {
		return v
	}
	if v, ok := m["reasoning_content"].(string); ok && v != "" {
		return v
	}
	if d, ok := m["data"].(map[string]any); ok {
		if v, ok := d["response"].(string); ok && v != "" {
			return v
		}
		if v, ok := d["reasoning_content"].(string); ok && v != "" {
			return v
		}
	}
	for _, k := range []string{"content", "text", "delta"} {
		if v, ok := m[k].(string); ok && v != "" {
			return v
		}
	}
	if d, ok := m["delta"].(map[string]any); ok {
		if v, ok := d["content"].(string); ok {
			return v
		}
	}
	if msg, ok := m["message"].(map[string]any); ok {
		if v, ok := msg["content"].(string); ok {
			return v
		}
	}
	if d, ok := m["data"].(map[string]any); ok {
		if v, ok := d["content"].(string); ok {
			return v
		}
	}
	return ""
}

// previewOf 取最后一条 user 消息作为请求预览（日志用）。
func previewOf(req *OAIRequest) string {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if strings.EqualFold(req.Messages[i].Role, "user") {
			return truncate(strings.ReplaceAll(req.Messages[i].TextContent(), "\n", " "), 80)
		}
	}
	return ""
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	// 回退到 UTF-8 字符边界：错误体/上游文案常含中文，按字节硬切会劈开汉字。
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// captureHeaders 复制入站头（剔除与上游鉴权冲突的字段）。
func captureHeaders(r *http.Request) http.Header {
	out := http.Header{}
	drop := map[string]bool{
		"host": true, "content-length": true, "connection": true,
		"authorization": true, "x-api-key": true,
	}
	for k, vs := range r.Header {
		if drop[strings.ToLower(k)] {
			continue
		}
		for _, v := range vs {
			out.Add(k, v)
		}
	}
	return out
}

// anthToOAIMessageWithStop 在 anthToOAIMessage 基础上对正文做网关侧 stop 截断。
// stops 为空时与 anthToOAIMessage 逐字段一致（默认流量零改动）。
func anthToOAIMessageWithStop(body []byte, model string, stops []string) map[string]any {
	out := anthToOAIMessage(body, model)
	if len(stops) == 0 {
		return out
	}
	choices, _ := out["choices"].([]map[string]any)
	for _, c := range choices {
		msg, _ := c["message"].(map[string]any)
		if msg == nil {
			continue
		}
		s, _ := msg["content"].(string)
		if s == "" {
			continue
		}
		if cut, hit := truncateAtStop(s, stops); hit {
			msg["content"] = cut
			c["finish_reason"] = "stop"
		}
	}
	return out
}

// anthToOAIMessage 将 Anthropic Messages 响应转换为 OpenAI chat.completion（非流式用）。
// Anthropic content 只取 text 块；thinking 块不进正文；tool_use 块翻成 tool_calls。
func anthToOAIMessage(body []byte, model string) map[string]any {
	var anth struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Content []struct {
			Type     string          `json:"type"`
			Text     string          `json:"text"`
			Thinking string          `json:"thinking"`
			ID       string          `json:"id"`
			Name     string          `json:"name"`
			Input    json.RawMessage `json:"input"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	out := map[string]any{
		"id":      "chatcmpl-" + newID(""),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
	}
	if json.Unmarshal(body, &anth) != nil {
		out["choices"] = []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": string(body)},
			"finish_reason": "stop",
		}}
		return out
	}
	var sb strings.Builder
	toolCalls := make([]oaiToolCall, 0, 2)
	for _, c := range anth.Content {
		switch c.Type {
		case "tool_use":
			if c.Name == "" {
				continue
			}
			args := "{}"
			if len(c.Input) > 0 {
				args = string(c.Input)
			}
			id := c.ID
			if id == "" {
				id = "call_" + newID("")
			}
			toolCalls = append(toolCalls, oaiToolCall{
				ID:   id,
				Type: "function",
				Function: oaiToolCallFnSpec{
					Name:      c.Name,
					Arguments: args,
				},
			})
		default:
			if c.Text != "" {
				sb.WriteString(c.Text)
			}
		}
	}
	if anth.Model != "" {
		out["model"] = anth.Model
	}
	if anth.ID != "" {
		out["id"] = anth.ID
	}
	finish := "stop"
	switch anth.StopReason {
	case "max_tokens":
		finish = "length"
	case "tool_use":
		finish = "tool_calls"
	}
	msg := map[string]any{"role": "assistant", "content": sb.String()}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
		if finish == "stop" {
			// Anthropic 回 tool_use 时 stop_reason 已是 tool_use；上游若漏填，
			// 有 tool_calls 就必须收敛成 tool_calls，否则客户端不会去执行工具。
			finish = "tool_calls"
		}
	}
	out["choices"] = []map[string]any{{
		"index":         0,
		"message":       msg,
		"finish_reason": finish,
	}}
	out["usage"] = map[string]any{
		"prompt_tokens":     anth.Usage.InputTokens,
		"completion_tokens": anth.Usage.OutputTokens,
		"total_tokens":      anth.Usage.InputTokens + anth.Usage.OutputTokens,
	}
	return out
}
