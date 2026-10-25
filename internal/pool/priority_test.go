package pool

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jjjane233/workbuddy-3in1/internal/auth"
)

// TestPriorityZeroRegression 全池都不设优先级时，选号行为与引入前一致：
// 最高积分号仍占多数（说明 filterByPriority 原样返回，没有改变候选集）。
func TestPriorityZeroRegression(t *testing.T) {
	withNoPickGap(t)
	p := New("")
	p.Add(&auth.Auth{UID: "u1"})
	p.Add(&auth.Auth{UID: "u2"})
	p.SetCredits("u1", 100, 0)
	p.SetCredits("u2", 50000, 0)
	counts := map[string]int{}
	for i := 0; i < 2000; i++ {
		counts[p.Pick().UID]++
	}
	if counts["u2"] <= counts["u1"] {
		t.Errorf("未设优先级时应保持既有加权：u2(高积分) 应更多，got %v", counts)
	}
	if counts["u1"] == 0 {
		t.Errorf("低积分号不应被完全饿死（加权抽签而非硬过滤），got %v", counts)
	}
}

// TestPriorityDominatesWhenSet 设了优先级的号只要可用就独占流量。
func TestPriorityDominatesWhenSet(t *testing.T) {
	withNoPickGap(t)
	p := New("")
	p.Add(&auth.Auth{UID: "low"})
	p.Add(&auth.Auth{UID: "high"})
	// 故意让高优先号的积分远低于另一个：优先级必须压过积分因素。
	p.SetCredits("low", 100000, 0)
	p.SetCredits("high", 1, 0)
	if !p.SetPriority("high", 80) {
		t.Fatal("SetPriority 应返回 true")
	}
	for i := 0; i < 500; i++ {
		if got := p.Pick(); got == nil || got.UID != "high" {
			t.Fatalf("高优先级号可用时必须只选它，got %+v", got)
		}
	}
}

// TestPriorityOverflowToLowerTier 高优先级层全部不可用时，请求落到下一层
// （而不是返回 no_available_account）。
func TestPriorityOverflowToLowerTier(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "high"})
	p.Add(&auth.Auth{UID: "low"})
	p.SetPriority("high", 80)
	p.Cooldown("high", CoolSoft, time.Hour, "429")
	got := p.Pick()
	if got == nil || got.UID != "low" {
		t.Fatalf("高优先级层不可用时应溢出到低层，got %+v", got)
	}
}

// TestPriorityInFlightOverflow 高优先级号占满在途名额时同样溢出到低层，
// 不得因为「优先级高」而突破并发上限。
func TestPriorityInFlightOverflow(t *testing.T) {
	p := New("")
	p.SetMaxInFlight(1)
	p.Add(&auth.Auth{UID: "high"})
	p.Add(&auth.Auth{UID: "low"})
	p.SetPriority("high", 100)
	if !p.Acquire("high") {
		t.Fatal("acquire 应成功")
	}
	defer p.Release("high")
	got := p.Pick()
	if got == nil || got.UID != "low" {
		t.Fatalf("高优先级号占满时应溢出到低层，got %+v", got)
	}
}

// TestPriorityClearedRestoresDefault 设为 0 后立刻回到默认规则。
func TestPriorityClearedRestoresDefault(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1"})
	p.Add(&auth.Auth{UID: "u2"})
	p.SetPriority("u1", 90)
	if n, ok := p.Priority("u1"); !ok || n != 90 {
		t.Fatalf("Priority(u1)=%d,%v want 90,true", n, ok)
	}
	p.SetPriority("u1", 0)
	if n, _ := p.Priority("u1"); n != 0 {
		t.Fatalf("清除后 priority=%d want 0", n)
	}
	// 清除后两个号同层（都是 0），高积分的 u2 应能重新被选到。
	p.SetCredits("u2", 50000, 0)
	p.SetCredits("u1", 1, 0)
	withNoPickGap(t)
	seen := false
	for i := 0; i < 300; i++ {
		if got := p.Pick(); got != nil && got.UID == "u2" {
			seen = true
			break
		}
	}
	if !seen {
		t.Error("清除优先级后高积分号应重新可被选中")
	}
}

// TestPriorityClamp 越界输入按可纠正值处理，而不是报错或写坏状态。
func TestPriorityClamp(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1"})
	cases := []struct{ in, want int }{
		{-5, 0}, {0, 0}, {1, 1}, {100, 100}, {999, PriorityMax},
	}
	for _, c := range cases {
		p.SetPriority("u1", c.in)
		if got, _ := p.Priority("u1"); got != c.want {
			t.Errorf("SetPriority(%d) → %d want %d", c.in, got, c.want)
		}
	}
	if p.SetPriority("nope", 10) {
		t.Error("不存在的 uid 应返回 false")
	}
}

// TestPriorityFallbackRespectsTier 全冷却兜底也分层：优先捞高优先级层里最早到期的，
// 而不是跨层取全池最早到期者。
func TestPriorityFallbackRespectsTier(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "low"})
	p.Add(&auth.Auth{UID: "high"})
	p.SetPriority("high", 80)
	// low 更早到期（1 分钟），high 更晚（30 分钟）：按旧逻辑会选 low。
	p.Cooldown("low", CoolSoft, time.Minute, "429")
	p.Cooldown("high", CoolSoft, 30*time.Minute, "429")
	got := p.Pick()
	if got == nil || got.UID != "high" {
		t.Fatalf("兜底应优先高优先级层，got %+v", got)
	}
}

// TestPriorityPersist 优先级跨重启保留（落盘 → 重新加载）。
func TestPriorityPersist(t *testing.T) {
	fp := filepath.Join(t.TempDir(), "state.json")
	p := New(fp)
	p.Add(&auth.Auth{UID: "u1"})
	p.Add(&auth.Auth{UID: "u2"})
	p.SetPriority("u1", 75)
	p.Flush()

	// 重新加载：state.json 里应带回 priority。
	raw, err := os.ReadFile(fp)
	if err != nil {
		t.Fatal(err)
	}
	if !containsAll(string(raw), `"priority": 75`) {
		t.Fatalf("state.json 未落盘 priority:\n%s", raw)
	}
	p2 := New(fp)
	p2.Add(&auth.Auth{UID: "u1"})
	p2.Add(&auth.Auth{UID: "u2"})
	if n, ok := p2.Priority("u1"); !ok || n != 75 {
		t.Fatalf("重启后 Priority(u1)=%d,%v want 75,true", n, ok)
	}
	if n, _ := p2.Priority("u2"); n != 0 {
		t.Fatalf("未设优先级的号重启后应为 0，got %d", n)
	}
}

// TestPriorityPersistClampsDirtyValue 手改 state.json 写入越界值时恢复侧钳位，
// 不让一个坏数字把整池选号搞乱。
func TestPriorityPersistClampsDirtyValue(t *testing.T) {
	fp := filepath.Join(t.TempDir(), "state.json")
	raw := `{"accounts":{"u1":{"credits":10,"priority":9999,"cool_kind":0}}}`
	if err := os.WriteFile(fp, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	p := New(fp)
	p.Add(&auth.Auth{UID: "u1"})
	if n, _ := p.Priority("u1"); n != PriorityMax {
		t.Fatalf("越界优先级应钳到 %d，got %d", PriorityMax, n)
	}
}

// TestPriorityStickyAvailablePool 粘性分配池只含最高优先级层——
// 否则新会话会被哈希到低权号上，而绑定在 TTL 内不迁移。
func TestPriorityStickyAvailablePool(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "a"})
	p.Add(&auth.Auth{UID: "b"})
	p.Add(&auth.Auth{UID: "c"})
	p.SetPriority("b", 60)

	uids := p.AvailableUIDs()
	if len(uids) != 1 || uids[0] != "b" {
		t.Fatalf("粘性分配池应只含最高层 [b]，got %v", uids)
	}
	// 清除后回到全池。
	p.SetPriority("b", 0)
	if got := p.AvailableUIDs(); len(got) != 3 {
		t.Fatalf("清除优先级后应回到全池 3 个，got %v", got)
	}
}

// TestPriorityStickyWeightedPool 加权粘性池（带快过期虚拟实例）同样分层，
// 且分层在虚拟实例展开之前——否则快过期权重会把低权号重新拉进分配池。
func TestPriorityStickyWeightedPool(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "low"})
	p.Add(&auth.Auth{UID: "high"})
	// low 有快过期积分（会得到 3 个虚拟实例），high 没有。
	p.SetCreditsDetailed("low", 100, 100, 100, time.Now().Add(time.Hour), 100)
	p.SetPriority("high", 50)

	got := p.WeightedAvailableUIDsForModelRealm("", "cn")
	if len(got) != 1 || got[0] != "high" {
		t.Fatalf("分层应先于虚拟实例展开：want [high]，got %v", got)
	}
}

// TestPriorityStatusExposed Status 必须透出 priority（面板 chip 的唯一数据源）。
func TestPriorityStatusExposed(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1"})
	p.SetPriority("u1", 42)
	st, ok := p.Status("u1")
	if !ok {
		t.Fatal("Status 应存在")
	}
	if st.Priority != 42 {
		t.Fatalf("Status.Priority=%d want 42", st.Priority)
	}
	// 零值也要透出（前端据此判断「未设置」并渲染 +权重）。
	p.Add(&auth.Auth{UID: "u2"})
	st2, _ := p.Status("u2")
	if st2.Priority != 0 {
		t.Fatalf("Status.Priority=%d want 0", st2.Priority)
	}
}

// TestPriorityPerProviderIsolation 优先级只在同一 provider 内比较：
// 高优先级的 Trae 号不得抢走 CodeBuddy 的请求（串号防线优先于分层）。
func TestPriorityPerProviderIsolation(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "cb"})
	p.Add(&auth.Auth{UID: "tr", Provider: auth.ProviderTrae})
	p.SetPriority("tr", 100)
	got := p.Pick()
	if got == nil || got.UID != "cb" {
		t.Fatalf("Pick() 应锁 CodeBuddy，got %+v", got)
	}
}

// containsAll 简易包含判断（避免为一个断言引入 strings 之外的工具）。
func containsAll(s string, subs ...string) bool {
	for _, sub := range subs {
		found := false
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}
