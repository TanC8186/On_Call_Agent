# 星云电商平台 — Redis 缓存集群运维手册

## 集群拓扑

| 节点 | IP | 角色 | 最大内存 |
|------|-----|------|---------|
| redis-master-01 | 10.0.4.11 | Master | 32GB |
| redis-slave-01 | 10.0.4.12 | Slave | 32GB |
| redis-slave-02 | 10.0.4.13 | Slave | 32GB |
| redis-sentinel-01 | 10.0.4.21 | Sentinel | 1GB |

集群使用 **Redis 7.2**，哨兵模式部署，共 3 个 Sentinel 节点。

## 告警规则

| 告警名称 | 触发条件 | 级别 | 通知方式 |
|----------|---------|------|---------|
| Redis 内存使用率过高 | `used_memory / maxmemory > 80%` 持续 10 分钟 | P2 | 企业微信 |
| Redis 连接数过多 | `connected_clients > 8000` 持续 5 分钟 | P2 | 企业微信 + 邮件 |
| Redis 主从同步延迟 | `master_last_io_seconds_ago > 30` | P1 | 企业微信 + 短信 |
| Redis 节点不可达 | Sentinel 判定 SDOWN，持续 2 分钟 | P0 | 企业微信 + 短信 + 电话 |
| Redis 慢查询激增 | `slowlog` 每分钟新增 > 50 条 | P3 | 邮件 |

## 常见故障处理

### 场景一：内存使用率接近上限

**根因分析**：
1. 执行 `INFO memory` 查看 `used_memory_rss` 和 `maxmemory_policy`
2. 若淘汰策略为 `noeviction`，写入操作会被拒绝
3. 执行 `redis-cli --bigkeys` 找出大 Key
4. 执行 `INFO stats` 查看 `evicted_keys` 是否持续增长

**处理步骤**：
1. 临时方案：执行 `CONFIG SET maxmemory-policy allkeys-lru` 启用 LRU 淘汰
2. 如果大 Key 是业务缓存，通知开发团队评估是否可以删除
3. 永久方案：扩容集群或优化业务数据结构（如将 JSON 字符串改为 Hash 存储）

### 场景二：主从同步延迟

**根因分析**：
1. 在 Slave 执行 `INFO replication` 查看 `master_last_io_seconds_ago`
2. 检查 Master 是否在执行 `BGSAVE` 或 `BGREWRITEAOF`
3. 检查网络带宽：`iftop -i eth0` 查看流量是否打满

**处理步骤**：
1. 如果 Master 在做 RDB 持久化，等待完成或考虑 `CONFIG SET save ""` 临时关闭
2. 如果是网络问题，检查机房带宽是否被其他业务占满
3. 临时方案：在 Slave 执行 `CONFIG SET slave-read-only no` 分担读压力（注意数据一致性）

### 场景三：P0 节点不可达

**应急流程**：
1. 确认是网络分区还是进程崩溃（SSH 到目标节点执行 `ps aux | grep redis`）
2. 如果是进程崩溃，检查 `/var/log/redis/redis-server.log` 最后 100 行
3. 如果是网络分区，检查防火墙规则 `iptables -L -n`
4. Sentinel 会自动完成故障转移，登录任意 Sentinel 执行 `SENTINEL masters` 确认新 Master
5. 旧 Master 恢复后会自动变为 Slave 加入集群

## 特殊配置说明

- 本集群设置了 `notify-keyspace-events "Ex"`，用于监听 Key 过期事件，业务侧依赖此功能做订单超时取消。
- 禁用了 `CONFIG` 命令（`rename-command CONFIG ""`），需要修改配置时必须通过配置文件 + 重启。
- 慢查询阈值 `slowlog-log-slower-than 10000`（10 毫秒）。
- 最大客户端连接数 `maxclients 20000`。

## 历史事故记录

| 日期 | 事故 | 影响 | 根因 |
|------|------|------|------|
| 2025-11-03 | 内存耗尽导致写入拒绝 | 订单创建失败 23 分钟 | 促销活动流量超预期 3 倍，缓存预热不足 |
| 2025-12-18 | 主从同步中断 | 读请求全部打到 Master | 机房核心交换机故障，丢包率 40% |
| 2026-03-07 | Sentinel 误判 SDOWN | 触发了一次不必要的故障转移 | `sentinel down-after-milliseconds` 设置为 5000ms，网络抖动触发 |

## 验证问题

如果你在读这份文档，请回答以下验证问题：
- 星云电商 Redis 集群的 Sentinel 节点 IP 是什么？
- P0 级别告警的通知方式有哪些？
- 2025 年 11 月 3 日事故的根因是什么？
