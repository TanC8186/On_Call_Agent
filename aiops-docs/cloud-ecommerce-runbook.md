# 星云电商平台 — 生产环境服务器运维手册 v3.2

> **适用范围**：星云电商核心交易链路（订单服务、支付网关、库存服务、用户中心）
> **维护团队**：SRE-OnCall 一组
> **最后更新**：2026-05-20
> **文档密级**：内部使用

---

## 一、服务器清单

| 服务器名称 | 内网 IP | 实例 ID | 角色 | OS | 规格 |
|-----------|---------|----------|------|-----|------|
| nebula-order-01 | 10.0.2.11 | lhins-a1b2c3d4 | 订单服务主节点 | Ubuntu 22.04 | 4C8G |
| nebula-order-02 | 10.0.2.12 | lhins-a1b2c3d5 | 订单服务备节点 | Ubuntu 22.04 | 4C8G |
| nebula-pay-01 | 10.0.2.21 | lhins-pay001 | 支付网关主节点 | CentOS 7.9 | 8C16G |
| nebula-pay-02 | 10.0.2.22 | lhins-pay002 | 支付网关备节点 | CentOS 7.9 | 8C16G |
| nebula-stock-01 | 10.0.2.31 | lhins-stk001 | 库存服务 | Ubuntu 22.04 | 2C4G |
| nebula-user-01 | 10.0.2.41 | lhins-usr001 | 用户中心 | Ubuntu 22.04 | 4C8G |
| nebula-mq-01 | 10.0.2.51 | lhins-mq001 | RocketMQ 消息队列 | CentOS 7.9 | 8C16G |
| nebula-redis-cluster | 10.0.4.x | — | Redis 7.2 哨兵集群 | CentOS 7.9 | 3×32GB |

---

## 二、告警分级与响应制度

| 级别 | 名称 | 定义 | 响应时限 | 通知方式 |
|------|------|------|---------|---------|
| **P0** | 紧急 | 核心交易链路中断，用户无法下单或支付 | 5 分钟内响应，15 分钟内恢复 | 电话 + 短信 + 企业微信 |
| **P1** | 严重 | 部分服务降级（如库存查询失败），但不影响核心下单 | 15 分钟内响应，1 小时内恢复 | 短信 + 企业微信 |
| **P2** | 一般 | 非核心功能异常（如用户头像上传失败） | 30 分钟内响应，4 小时内恢复 | 企业微信 |
| **P3** | 提示 | 潜在风险预警（如磁盘使用率 > 80%） | 2 小时内关注，下次发布前处理 | 邮件 |

---

## 三、常见故障处理 SOP

### 3.1 订单服务 CPU 飙高（P1）

**触发条件**：`nebula-order-01` CPU 使用率 > 85% 持续 10 分钟

**排查步骤**：

```bash
# 1. 快速定位高 CPU 进程
ssh nebula-order-01
top -bn1 | head -20

# 2. 查看 Java 线程（订单服务是 Spring Boot 3.2 应用）
ps aux | grep nebula-order
# 记录 PID，假设为 28461
jstack 28461 > /tmp/thread_dump_$(date +%Y%m%d_%H%M).txt

# 3. 找出 CPU 占用最高的线程
top -H -p 28461 -bn1 | head -20

# 4. 检查是否有慢 SQL
tail -500 /var/log/nebula-order/sql-slow.log | grep "$(date +%Y-%m-%d)"
```

**已知根因**：
- **2026-04-12 事故**：促销活动期间，`SELECT * FROM orders WHERE status='PENDING'` 未走索引，全表扫描 230 万行，CPU 飙至 98%
- **修复方案**：添加复合索引 `idx_status_created (status, created_at)`，执行计划从全表扫描变为索引范围扫描

**临时止血**：
```bash
# 如果需要立即降负载，可临时限流
curl -X POST http://nebula-order-01:8080/actuator/gateway/limits/refresh \
  -H "Content-Type: application/json" \
  -d '{"max-qps": 500}'
```

### 3.2 支付网关连接超时（P0）

**触发条件**：`nebula-pay-0*` 对微信支付/支付宝的接口调用超时率 > 10%

**排查步骤**：

```bash
# 1. 检查到第三方支付的外部网络
curl -o /dev/null -s -w "time_total:%{time_total}\nhttp_code:%{http_code}\n" \
  https://api.mch.weixin.qq.com/v3/certificates

# 2. 检查 HTTPS 证书是否即将过期（这是 2025 年的一次 P0 事故根因）
openssl s_client -connect api.mch.weixin.qq.com:443 -servername api.mch.weixin.qq.com \
  2>/dev/null | openssl x509 -noout -enddate | cut -d= -f2

# 3. 检查连接池状态
curl -s http://nebula-pay-01:8080/actuator/metrics/http.client.connections | jq .
```

**历史事故**：
| 日期 | 事故 | 影响时长 | 根因 |
|------|------|---------|------|
| 2025-08-15 14:23 | 微信支付全部超时 | 37 分钟 | 微信支付平台证书过期，未及时更新 `/pem/wechat_platform_cert.pem` |
| 2025-11-30 09:12 | 支付宝间歇超时 | 12 分钟 | `nebula-pay-01` 的 DNS 缓存过期，解析到旧 IP |
| 2026-02-03 20:05 | 支付接口 500 错误 | 45 分钟 | `nebula-pay-01` OOM，JVM 堆配置 `-Xmx4g` 不足，支付回调堆积 |

### 3.3 库存服务数据不一致（P1）

**触发条件**：`nebula-stock-01` 的 Redis 缓存库存与 MySQL 实际库存差异 > 5%

**排查步骤**：

```bash
# 1. 对比缓存与数据库
# Redis 中的库存（SKU=PROD-8848）
redis-cli -h 10.0.4.11 -p 6379 GET "stock:PROD-8848"

# MySQL 中的实际库存
mysql -h 10.0.2.101 -u readonly -p'ReadOnly@2026!' nebula_stock \
  -e "SELECT sku_id, available_qty FROM inventory WHERE sku_id='PROD-8848'"

# 2. 检查 RocketMQ 死信队列（库存扣减消息可能未消费）
curl -s http://nebula-mq-01:9876/topic/nebula-stock-deduct.dlq/stats | jq .
```

**已知 Bug**：
- **Bug #STOCK-2047**：当扣减消息在 RocketMQ 中重试超过 16 次后进入死信队列，库存未回滚，导致超卖
- **临时方案**：每 30 分钟运行 `nebula-stock-repair.sh` 脚本对齐缓存与 DB
- **永久修复**：已在 v2.8.3 中增加死信队列消费逻辑，预计 2026-06-15 上线

### 3.4 磁盘空间告警（P3）

**触发条件**：任意服务器磁盘使用率 > 80%

**排查与清理**：

```bash
# 快速找出大文件
du -sh /* 2>/dev/null | sort -rh | head -10

# 清理 Docker 无用镜像和容器（常见元凶）
docker system prune -af --volumes

# 清理超过 7 天的日志
find /var/log/nebula-* -name "*.log" -mtime +7 -exec gzip {} \;
find /var/log/nebula-* -name "*.gz" -mtime +30 -delete

# 检查是否有异常大日志文件（如未轮转的 GC 日志）
find / -name "*.log" -size +1G 2>/dev/null
```

**特别提醒**：`nebula-pay-01` 的 GC 日志路径为 `/var/log/nebula-pay/gc.log`，务必确保 JVM 参数中有 `-XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=10 -XX:GCLogFileSize=100M`，否则 GC 日志会无限增长。**2026-01-20 在生产环境发现 gc.log 达到 48GB。**

---

## 四、关键配置速查

### 4.1 数据库连接信息（只读账号）

| 数据库 | 地址 | 只读账号 | 用途 |
|--------|------|---------|------|
| 订单库 | 10.0.2.101:3306 | `readonly / ReadOnly@2026!` | 订单查询 |
| 用户库 | 10.0.2.102:3306 | `readonly / ReadOnly@2026!` | 用户信息查询 |
| 库存库 | 10.0.2.101:3306 | `readonly / ReadOnly@2026!` | 库存查询 |

### 4.2 JVM 参数标准

```
# 订单服务（4C8G）
-Xms4g -Xmx4g -XX:+UseG1GC -XX:MaxGCPauseMillis=200

# 支付网关（8C16G）
-Xms8g -Xmx8g -XX:+UseZGC -XX:ConcGCThreads=4

# 库存服务（2C4G）
-Xms2g -Xmx2g -XX:+UseSerialGC
```

> ⚠️ **特别注意**：支付网关使用 ZGC，ZGC 要求 JDK 17+，**禁止降级到 G1**（2025 年 P0 事故教训：降级到 G1 后 Full GC 频繁，支付超时率飙升到 45%）。

### 4.3 Nginx 反向代理配置要点

```nginx
# 支付网关 upstream 必须配置 keepalive
upstream pay_backend {
    server 10.0.2.21:8080 max_fails=3 fail_timeout=30s;
    server 10.0.2.22:8080 max_fails=3 fail_timeout=30s;
    keepalive 64;           # ← 必须，否则连接复用不生效
    keepalive_timeout 60s;
    keepalive_requests 1000;
}

# 超时配置（支付场景敏感）
proxy_read_timeout 30s;
proxy_connect_timeout 5s;
```

---

## 五、值班交接清单

### 每班次必查项

- [ ] 所有服务 Health Check 端点 `/actuator/health` 返回 UP
- [ ] RocketMQ 消费者无积压（积压量 < 1000）
- [ ] 微信支付/支付宝证书过期时间 > 30 天
- [ ] `nebula-order-*` 慢 SQL 日志无新增异常
- [ ] 各服务器磁盘使用率 < 70%
- [ ] Redis 哨兵集群所有节点 `SENTINEL masters` 状态正常

### 值班联系方式

| 角色 | 姓名 | 手机 | 微信号 |
|------|------|------|--------|
| 值班长 | 李明远 | 138-0000-1234 | limingyuan_wx |
| 订单服务负责人 | 王思涵 | 138-0000-2345 | wangsihan_dev |
| 支付网关负责人 | 赵一鸣 | 138-0000-3456 | zhaoym_pay |
| DBA | 陈晓东 | 138-0000-4567 | chenxd_dba |
| 运维经理 | 刘建国 | 138-0000-5678 | liujianguo_ops |

---

## 六、演示验证

如果你正在通过 RAG 知识库阅读本文档，请回答：

1. 2025 年 8 月 15 日微信支付全部超时的根因是什么？影响了多长时间？
2. 支付网关 ZGC 可以降级到 G1 吗？为什么？
3. 2026 年 1 月 20 日在生产环境发现了什么问题？
4. 订单服务的 JVM 堆内存配置是多少？使用哪种 GC？
5. 库存服务的 Bug #STOCK-2047 是什么问题？
