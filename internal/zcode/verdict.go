// 上游错误的业务码分类。
//
// 为什么必须分类：ZCode 的 HTTP 状态码表达力极弱——同一批语义都压在 429/403/405 上，
// 真正的成因在响应体的业务 code 里。旧实现只看 status，于是「套餐额度用完」「并发超限」
// 「需要人机验证」「客户端指纹被拦」四类完全不同的问题，全都塌缩成一句「429 限流」或
// 「401 未授权」，处置也全错（都去冷却账号，而其中两类根本不该罚号）。
//
// 码表来源（两处独立实现交叉验证，结论一致）：
//
//	D3-vin/Zcode2Api  internal/upstream/quota.go  QuotaFamilyCodes
//	liu5269/zcode2api app/routes/gateway.py       _is_exhausted / _is_captcha_error
//
// 处置差异（本包只做判定，处置在 server 层）：
//
//	配额族 429  → 该账号在该模型上额度耗尽：按模型标死，换号（不冷却账号、不换出口）
//	3010        → 模型准入并发上限：账号仍可用，同账号短退避重试（1s/2s，官方客户端节奏）
//	普通 429    → 真·限流：跟随 Retry-After，没有才按软冷却基数
//	3007/验证码 → 人机验证失效：作废验证码缓存后同账号重试
//	3012/3009   → 客户端指纹/负载被拦：换出口或换号，账号本身不坏
//	401/403     → 凭据被拒：才是真的需要换凭证/重登
package zcode

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
)

// 上游业务码（只列本包用到的）。
const (
	// CodeQuotaExhausted 是「额度/余额类」的通用耗尽码（liu5269 用的主判据）。
	CodeQuotaExhausted = 1005
	// CodeCaptchaFail 是人机验证失败（阿里云无痕校验没过）。
	CodeCaptchaFail = 3007
	// CodeLoadBlocked 是「非官方客户端 / 异常流量」拦截。
	CodeLoadBlocked = 3012
	// CodeAdmissionLimit 是模型准入并发上限（账号仍可用）。
	CodeAdmissionLimit = 3010
	// CodePlanLacksModel 是套餐不含该模型。
	CodePlanLacksModel = 1311
)

// quotaFamilyCodes 是「该账号该模型额度已耗尽」的业务码集合。
//
// 依据 docs.z.ai 错误码表：
//
//	1113 欠费           1308 用量上限      1309 套餐过期
//	1310 周/月上限      1311 套餐不含该模型 1313 公平使用限制
//	1316-1321 5h/7d/子账号/企业级上限
//
// 与「普通 429」的区别：这些码意味着**等一会儿重试也没用**（要等周期重置），
// 所以正确处置是「按模型标死 + 立刻换号」，而不是给账号上冷却。
var quotaFamilyCodes = map[int]bool{
	1113: true, 1308: true, 1309: true, 1310: true, 1311: true, 1313: true,
	1316: true, 1317: true, 1318: true, 1319: true, 1320: true, 1321: true,
}

// BizCode 从响应体提取上游业务码（{code:N,...} 或 {error:{code:N}}）。
//
// 返回 0 表示提取不到：调用方必须把它当成「未知」而不是「成功」。
// 为什么不用正则：这是每次失败请求都会走的热路径，且 JSON 解析比正则更快更准。
func BizCode(body []byte) int {
	body = bytes.TrimSpace(body)
	if len(body) == 0 || body[0] != '{' {
		return 0
	}
	// code 可能是数字或字符串（上游两种都发过），用 json.Number 兼容。
	var probe struct {
		Code  json.Number `json:"code"`
		Error struct {
			Code json.Number `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &probe) != nil {
		return 0
	}
	for _, n := range []json.Number{probe.Code, probe.Error.Code} {
		if n == "" {
			continue
		}
		if v, err := n.Int64(); err == nil && v > 0 {
			return int(v)
		}
	}
	return 0
}

// IsQuotaExhausted 判定「该账号该模型额度已耗尽」。
//
// 两级判据（任一成立）：
//   - 业务码属于配额族（1308/1310/1316-1321 等），或等于 1005；
//   - HTTP 402（Payment Required）。
//
// 不看 HTTP 429 之外的错误体文案：上游文案会变（"quota"/"余额"/"insufficient"），
// 码表是稳定的。
func IsQuotaExhausted(status int, body []byte) bool {
	if status == http.StatusPaymentRequired {
		return true
	}
	code := BizCode(body)
	return code == CodeQuotaExhausted || quotaFamilyCodes[code]
}

// IsAdmissionLimit 判定「模型准入并发上限」（3010）。
//
// 语义与限流完全不同：账号健康、额度充足，只是该模型同时在跑的请求太多。
// 官方客户端的行为是「等 1s，再等 2s，同账号重试」，不换号也不换出口。
func IsAdmissionLimit(status int, body []byte) bool {
	return status == http.StatusTooManyRequests && BizCode(body) == CodeAdmissionLimit
}

// IsLoadBlocked 判定「客户端指纹/负载被拦」（3012 / unusual activity / 3009）。
//
// 这类失败**不是账号的问题**：换出口或换号都能过，账号不该被罚。
func IsLoadBlocked(status int, body []byte) bool {
	if status != http.StatusForbidden && status != http.StatusMethodNotAllowed &&
		status != http.StatusNotAcceptable && status != http.StatusTooManyRequests &&
		status != http.StatusServiceUnavailable {
		return false
	}
	if code := BizCode(body); code == CodeLoadBlocked || code == 3009 {
		return true
	}
	s := strings.ToLower(string(body))
	return strings.Contains(s, "unusual activity") || strings.Contains(s, "request has been blocked")
}

// IsPlanLacksModel 判定「套餐不含该模型」（1311 / not entitled）。
//
// 与配额耗尽的区别：换号可能有用（别的号套餐里可能有），但该号对该模型永久不可用，
// 不需要等周期重置——应记成长期避让而不是冷却。
func IsPlanLacksModel(status int, body []byte) bool {
	if code := BizCode(body); code == CodePlanLacksModel {
		return true
	}
	return IsModelNotEntitled(string(body))
}

// Verdict 是上游失败的统一分类结果。
type Verdict int

const (
	// VerdictUnknown 未分类（调用方按原有逻辑兜底）。
	VerdictUnknown Verdict = iota
	// VerdictQuotaExhausted 额度耗尽：按模型标死 + 换号，不冷却账号。
	VerdictQuotaExhausted
	// VerdictAdmissionLimit 准入并发上限：同账号短退避重试。
	VerdictAdmissionLimit
	// VerdictLoadBlocked 指纹/负载被拦：换号或换出口，不罚账号。
	VerdictLoadBlocked
	// VerdictCaptcha 人机验证失效：作废验证码缓存后同账号重试。
	VerdictCaptcha
	// VerdictUnauthorized 凭据被拒：真需要换凭证/重登。
	VerdictUnauthorized
	// VerdictRateLimited 普通限流：跟随 Retry-After 或软冷却。
	VerdictRateLimited
	// VerdictPlanLacksModel 套餐不含该模型：长期避让该 (账号, 模型)。
	VerdictPlanLacksModel
)

func (v Verdict) String() string {
	switch v {
	case VerdictQuotaExhausted:
		return "quota_exhausted"
	case VerdictAdmissionLimit:
		return "admission_limit"
	case VerdictLoadBlocked:
		return "load_blocked"
	case VerdictCaptcha:
		return "captcha"
	case VerdictUnauthorized:
		return "unauthorized"
	case VerdictRateLimited:
		return "rate_limited"
	case VerdictPlanLacksModel:
		return "plan_lacks_model"
	}
	return "unknown"
}

// Classify 把 (状态码, 响应体) 归到唯一的 Verdict。
//
// 判定顺序是刻意的，不能交换：
//
//  1. 402 最先：语义不依赖 body，且它一定意味着额度问题；
//  2. 验证码（3007）：可能伴随 403/429，但处置是「重取验证码」而不是罚号；
//  3. 指纹拦截（3012/3009）：同样可能伴随 401/403/405/429，账号无责，
//     且它是**可换端点绕过**的——必须排在「凭据被拒」之前，否则 3012+401
//     会被误判成「凭据坏了」而白白禁用账号；
//  4. 套餐不含模型（1311）：长期避让，比「配额耗尽」更精确（后者要等周期重置，
//     前者换号可能立刻有效），因此排在配额族之前；
//  5. 配额族：等重试没用，必须换号；
//  6. 准入上限（3010）：429 里唯一「同账号重试」有效的一类；
//  7. 401：凭据真被拒；
//  8. 其余 429：普通限流。
func Classify(status int, body []byte) Verdict {
	if status == http.StatusPaymentRequired {
		return VerdictQuotaExhausted
	}
	if code := BizCode(body); code == CodeCaptchaFail {
		return VerdictCaptcha
	}
	if IsLoadBlocked(status, body) {
		return VerdictLoadBlocked
	}
	if IsPlanLacksModel(status, body) {
		return VerdictPlanLacksModel
	}
	if IsQuotaExhausted(status, body) {
		return VerdictQuotaExhausted
	}
	if IsAdmissionLimit(status, body) {
		return VerdictAdmissionLimit
	}
	if status == http.StatusUnauthorized {
		return VerdictUnauthorized
	}
	if status == http.StatusTooManyRequests {
		return VerdictRateLimited
	}
	return VerdictUnknown
}
