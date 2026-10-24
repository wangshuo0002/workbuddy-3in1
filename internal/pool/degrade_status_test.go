package pool

import (
	"testing"
	"time"

	"github.com/jjjane233/workbuddy-3in1/internal/auth"
)

// TestStatusReportsDegradeAsCooling 连败降权必须出现在 Status 的 Cooling/CoolKind/
// CoolRemaining 上，否则面板会显示「可用」而请求根本选不到它。
// 现场：web 端账号卡显示「连败降权」但后端 Status.Cooling 漏了 degradeUntil，
// 导致 /status 与 healthy() 口径分裂（healthy 早已把降权当不可选）。
func TestStatusReportsDegradeAsCooling(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1"})
	p.SetDegrade(1, 10*time.Minute, time.Hour)
	p.NoteFailures("u1") // threshold=1 → 立即降权

	st, ok := p.Status("u1")
	if !ok {
		t.Fatal("no status")
	}
	if !st.Cooling {
		t.Fatalf("降权必须计入 cooling: %+v", st)
	}
	if st.CoolKind != "degraded" {
		t.Errorf("cool_kind=%q want degraded", st.CoolKind)
	}
	if st.CoolRemaining <= 0 || st.CoolRemaining > 600 {
		t.Errorf("cool_remaining_sec=%d want in (0,600]", st.CoolRemaining)
	}
	if st.Reason != degradeReason {
		t.Errorf("reason=%q want %q", st.Reason, degradeReason)
	}
	// 与 healthy() 口径一致：降权中的号不计入 healthy（CountsDetailed 的 cooling 计数）。
	if total, healthy, cooling, _, _ := p.CountsDetailed(); total != 1 || healthy != 0 || cooling != 1 {
		t.Errorf("counts total=%d healthy=%d cooling=%d want 1/0/1", total, healthy, cooling)
	}
	// 降权号不参与 normal 选号（无健康号时才会被「全冷却兜底」捞走，那是设计行为）。
	if uids := p.AvailableUIDs(); len(uids) != 0 {
		t.Errorf("AvailableUIDs=%v want 空（降权号不可选）", uids)
	}
}

// TestStatusDegradeNotReportedAfterExpiry 降权到期后 Status 不再报 cooling（不残留）。
func TestStatusDegradeNotReportedAfterExpiry(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1"})
	p.SetDegrade(1, time.Millisecond, time.Hour)
	p.NoteFailures("u1")
	time.Sleep(5 * time.Millisecond)

	st, _ := p.Status("u1")
	if st.Cooling || st.CoolKind == "degraded" || st.Reason != "" {
		t.Errorf("到期后不应残留降权状态: %+v", st)
	}
}

// TestReviveClearsDegrade 面板「解冻」必须清掉降权（惩罚状态一律归零），
// 否则复活后账号仍被降权截止挡在选号之外。
func TestReviveClearsDegrade(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1"})
	p.SetDegrade(1, 10*time.Minute, time.Hour)
	p.NoteFailures("u1")

	if !p.Revive("u1") {
		t.Fatal("Revive should return true")
	}
	st, _ := p.Status("u1")
	if st.Cooling || st.CoolKind == "degraded" {
		t.Errorf("Revive 后不应残留降权: %+v", st)
	}
	if st.ConsecutiveFails != 0 {
		t.Errorf("consecutive_fails=%d want 0", st.ConsecutiveFails)
	}
	if uids := p.AvailableUIDs(); len(uids) != 1 || uids[0] != "u1" {
		t.Errorf("Revive 后账号应可选, got %v", uids)
	}
}

// TestNoteSuccessClearsDegrade 成功是最强恢复证据：清连败计数与降权截止。
func TestNoteSuccessClearsDegrade(t *testing.T) {
	p := New("")
	p.Add(&auth.Auth{UID: "u1"})
	p.SetDegrade(1, 10*time.Minute, time.Hour)
	p.NoteFailures("u1")
	p.NoteSuccess("u1")

	st, _ := p.Status("u1")
	if st.Cooling || st.ConsecutiveFails != 0 {
		t.Errorf("成功后应清降权: %+v", st)
	}
}
