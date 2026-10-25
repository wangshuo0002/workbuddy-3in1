// 分池选号域：按 realm（cn/global）与 provider（workbuddy/trae/zcode）过滤选号与可用集合。
// 两个维度的空值都退化为「不过滤」，但对外导出的存量入口一律显式锁 CodeBuddy ——
// Trae/ZCode 账号无 domain，Realm() 会回落成 "cn"，不排除就会被 CN 分池误选（串号）。
package pool

import (
	"sort"
	"time"

	"github.com/jjjane233/workbuddy-3in1/internal/auth"
)

// expiringVirtualSlots 快过期账号在新会话候选集中的虚拟实例权重。
// 3:1 是温和偏好，不是固定比例：账号组成变化会自然改变最终占比。
const expiringVirtualSlots = 3

// AvailableUIDsForRealm 同 AvailableUIDs，但仅返回 Realm()==realm 的 CodeBuddy 账号。
// realm=="" 退化为 AvailableUIDs 的 CodeBuddy 子集（现状语义不变）。
func (p *Pool) AvailableUIDsForRealm(realm string) []string {
	return p.availableUIDsForScope(realm, auth.ProviderWorkbuddy)
}

// AvailableUIDsForScope 同 AvailableUIDsForRealm，但 provider 也可指定（三合一网关）。
// provider 空值归一化为 CodeBuddy，漏传时不会把上游号混进来。
func (p *Pool) AvailableUIDsForScope(realm, provider string) []string {
	return p.availableUIDsForScope(realm, auth.NormalizeProvider(provider))
}

func (p *Pool) availableUIDsForScope(realm, provider string) []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	now := time.Now()
	uids := make([]string, 0, len(p.byUID))
	for uid, e := range p.byUID {
		if realm != "" && e.a.Realm() != realm {
			continue
		}
		if e.a.ProviderValue() != provider {
			continue
		}
		if !e.healthy(now) {
			continue
		}
		if p.inFlightFull(e) {
			continue
		}
		uids = append(uids, uid)
	}
	sort.Strings(uids)
	// 人工优先级分层：与 pick 同口径裁剪到最高层（全池未设时原样返回）。
	// 这里也过滤的理由：本方法供面板「按上游挑一个号去拉模型目录」等场景使用，
	// 挑到低优先级号会让人误以为优先级不生效。
	return p.filterUIDsByPriorityLocked(uids)
}

// WeightedAvailableUIDsForModelRealm 返回带虚拟实例权重的可用账号列表。
//
// 普通账号出现 1 次；有效快过期账号出现 expiringVirtualSlots 次。调用方继续按
// 原有序列表哈希，即可让新会话对快过期账号形成温和偏好。重复项按 UID 排序后
// 展开，保证同一账号拓扑下不同进程得到一致列表。
//
// 该方法是现有 AvailableUIDsForModelRealm 的增量入口，不改变旧方法语义，也不
// 修改配置、状态或 Redis schema。prefer_expiring=false 时退化为逐账号一次。
func (p *Pool) WeightedAvailableUIDsForModelRealm(model, realm string) []string {
	return p.weightedAvailableUIDsForScope(model, realm, auth.ProviderWorkbuddy)
}

// WeightedAvailableUIDsForScope 同上，但可指定 provider（三合一网关的粘性选号用）。
func (p *Pool) WeightedAvailableUIDsForScope(model, realm, provider string) []string {
	return p.weightedAvailableUIDsForScope(model, realm, auth.NormalizeProvider(provider))
}

func (p *Pool) weightedAvailableUIDsForScope(model, realm, provider string) []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	now := time.Now()
	uids := make([]string, 0, len(p.byUID))
	for uid, e := range p.byUID {
		if realm != "" && e.a.Realm() != realm {
			continue
		}
		if e.a.ProviderValue() != provider {
			continue
		}
		if !e.healthyForModel(now, model) {
			continue
		}
		if p.inFlightFull(e) {
			continue
		}
		uids = append(uids, uid)
	}
	sort.Strings(uids)
	// 人工优先级分层：新会话的哈希分配池先裁到最高层。
	//
	// 为什么粘性分配也必须过滤：会话一旦绑定就在 TTL 内不迁移（默认 30m 滚动续期），
	// 若把新会话分给低优先级号，「优先用高权号」对长期对话型客户端会彻底失效——
	// 高权号只在会话过期后才有机会被选到。
	//
	// 副作用（刻意接受）：本列表同时是粘性快路径的**命中校验集**（session.availableSet），
	// 因此调高某号的优先级后，原先绑定在低优先级号上的会话会在下一次请求时重新分配。
	// 这与「该号冷却/占满时解绑重分配」是同一套既有机制，代价是那一次请求失去上游
	// prompt 缓存命中；换来的是设置优先级立即生效，而不是等 30 分钟会话过期。
	// 调低/清除优先级不会反向迁移（低权号不再在候选集里，但已有绑定仍然合法，
	// 因为清除优先级后所有账号同层、全部回到候选集）。
	uids = p.filterUIDsByPriorityLocked(uids)

	out := make([]string, 0, len(uids))
	for _, uid := range uids {
		slots := 1
		if p.preferExpiring {
			if e := p.byUID[uid]; expiringNow(e, now) {
				slots = expiringVirtualSlots
			}
		}
		for i := 0; i < slots; i++ {
			out = append(out, uid)
		}
	}
	return out
}

// AvailableUIDsForModelRealm 同 AvailableUIDsForModel，但仅返回 Realm()==realm 的
// CodeBuddy 账号（6004 模型豁免照常生效）。realm=="" 退化为 CodeBuddy 子集。
func (p *Pool) AvailableUIDsForModelRealm(model, realm string) []string {
	return p.availableUIDsForModelScope(model, realm, auth.ProviderWorkbuddy)
}

// AvailableUIDsForModelScope 同上，但可指定 provider（三合一网关）。
func (p *Pool) AvailableUIDsForModelScope(model, realm, provider string) []string {
	return p.availableUIDsForModelScope(model, realm, auth.NormalizeProvider(provider))
}

func (p *Pool) availableUIDsForModelScope(model, realm, provider string) []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	now := time.Now()
	uids := make([]string, 0, len(p.byUID))
	for uid, e := range p.byUID {
		if realm != "" && e.a.Realm() != realm {
			continue
		}
		if e.a.ProviderValue() != provider {
			continue
		}
		if !e.healthyForModel(now, model) {
			continue
		}
		if p.inFlightFull(e) {
			continue
		}
		uids = append(uids, uid)
	}
	sort.Strings(uids)
	// 人工优先级分层：与 pick / 粘性分配池同口径（全池未设时原样返回）。
	return p.filterUIDsByPriorityLocked(uids)
}
