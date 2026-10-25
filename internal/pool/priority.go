// 账号优先级（人工加权）：显式设置后形成「优先级层」，选号优先在最高层内进行；
// 未设置（0）的账号完全沿用既有规则——全池都不设优先级时行为逐位不变。
//
// 为什么用分层而不是权重倍数：
//   - 「谁权重最高优先用谁」的语义是**优先**，不是"更多次"。倍数加权在候选多时
//     只会让高权号多拿几倍流量，低权号仍会分到请求——与"优先用"的预期不符。
//   - 分层天然给出**溢出**语义：最高层全部冷却/占满后自动落到下一层，不需要额外
//     调参；层内仍走既有加权抽签（成本分层 + 余额 + 空闲补偿），所以单层内多号
//     并存时不会把流量压到同一个号上（防惊群逻辑原样保留）。
package pool

// PriorityMax 优先级上限：取值 1..PriorityMax，数字越大越优先；0 = 未设置。
//
// 为什么是 1..100 的整数而不是浮点：它是**分层键**，不是概率权重——同层内的
// 概率分布仍由既有加权抽签决定，细粒度数值没有信息量。100 档足够表达
// 「主力 / 备选 / 兜底」这类运维语义，面板上也一眼可读。
const PriorityMax = 100

// clampPriority 把外部输入钳到合法区间。越界视为可纠正的输入而非错误：面板/接口
// 传 999 应得到 100（而不是整次请求失败），避免一次手误变成一次排障。
func clampPriority(n int) int {
	if n < 0 {
		return 0
	}
	if n > PriorityMax {
		return PriorityMax
	}
	return n
}

// SetPriority 设置账号优先级（0 = 清除，恢复默认规则）。返回 false 表示 uid 不存在。
//
// 立即生效：priority 只参与选号过滤，不进任何冷却/熔断状态机，改完下一个请求
// 就按新分层挑号。落盘由 dirty 驱动（与 credits/冷却共用 5s flusher），
// 因此面板上改完不必重启，也不会因为重启丢偏好。
func (p *Pool) SetPriority(uid string, n int) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	e, ok := p.byUID[uid]
	if !ok {
		return false
	}
	n = clampPriority(n)
	if e.priority == n {
		return true
	}
	e.priority = n
	p.dirty.Store(true)
	return true
}

// Priority 返回账号当前优先级（0 = 未设置）。uid 不存在返回 (0, false)。
func (p *Pool) Priority(uid string) (int, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	e, ok := p.byUID[uid]
	if !ok {
		return 0, false
	}
	return e.priority, true
}

// topPriorityTier 返回候选集中的最高优先级层（全为 0 = 都没设 → 返回 0）。
func topPriorityTier(cands []*entry) int {
	best := 0
	for _, e := range cands {
		if e.priority > best {
			best = e.priority
		}
	}
	return best
}

// filterByPriority 把候选集裁剪到最高优先级层。
//
// 零回归的关键：最高层为 0（即没有任何候选设过优先级）时**原样返回入参**——
// 不复制、不重排，调用方后续逻辑（成本分层 / Top5 / 加权抽签）看到的候选与
// 引入本函数前完全一致。
//
// 只保留最高层而不是"按层加权"：分层的语义就是「只要这一档还有人可用，就不该
// 往下用」。向下溢出由调用方的 healthyForModel / inFlightFull 过滤天然提供。
func filterByPriority(cands []*entry) []*entry {
	top := topPriorityTier(cands)
	if top <= 0 {
		return cands
	}
	out := make([]*entry, 0, len(cands))
	for _, e := range cands {
		if e.priority == top {
			out = append(out, e)
		}
	}
	return out
}

// filterUIDsByPriorityLocked 把 uid 列表裁剪到最高优先级层（调用方须已持 p.mu）。
//
// 供会话粘性的**分配池**使用：新会话按这份列表哈希分配账号。若不过滤，新会话会被
// 分到低优先级号上，而会话一旦绑定就 TTL 内不迁移（默认 30m 滚动续期），
// 「优先用高权号」在粘性客户端上会形同虚设。
//
// 与 filterByPriority 同样的零回归口径：全为 0 时原样返回，不分配新切片。
func (p *Pool) filterUIDsByPriorityLocked(uids []string) []string {
	top := 0
	for _, uid := range uids {
		if e := p.byUID[uid]; e != nil && e.priority > top {
			top = e.priority
		}
	}
	if top <= 0 {
		return uids
	}
	out := make([]string, 0, len(uids))
	for _, uid := range uids {
		if e := p.byUID[uid]; e != nil && e.priority == top {
			out = append(out, uid)
		}
	}
	return out
}
