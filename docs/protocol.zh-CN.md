# 通信协议与重定向输入

[English](protocol.md) | 简体中文

## 真机消息

PUB 消息为单个 UTF-8 帧：`<topic> <JSON object>`。默认端点为本机回环地址。

| 发布端 | 端点 | Topic | 字段 |
| --- | --- | --- | --- |
| EX16 | `tcp://127.0.0.1:5567` | `ex16/state` | `name`、`sequence`、`timestamp`、`urdf_deg` |
| GX16 | `tcp://127.0.0.1:5557` | `gx16/state` | `timestamp`、实测 `motor_deg`、实测 `urdf_deg` |

角度为 16 个有限数值，单位度，按 joint1..joint16 排列：拇指、食指、中指、
无名指各四个关节。GX16 URDF 角度等于电机相对角度乘配置中的方向符号。
硬件驱动另加 90 度偏置。`gex-view` 直接订阅这两路数据，已删除旧的
重定向状态 topic 和控制统计消息。

GX16 REP 绑定 `tcp://127.0.0.1:5556`，请求格式为 `{ "cmd": "...", ... }`。
响应格式为 `{ "ok": boolean, "result": object|null, "error": string|null }`。

| 命令 | 参数 / 返回值 |
| --- | --- |
| `ping`、`status` | 返回 `connected`、`port` |
| `getjs` | 返回 `positions`、`units` |
| `setjs` | `positions`：16 个有限数值 |
| `setj` | `joint`：1..16 的整数；`position`：有限数值 |
| `home` | 将所有电机移动到相对零位 |
| `torque_on`、`torque_off` | 开启/关闭电机扭矩 |
| `shutdown` | 退出服务并尝试关闭扭矩 |

`getjs/setjs/setj` 接受 `units="urdf_deg"`（默认）或 `"motor_deg"`。
成功的写命令返回空结果对象。直接运动命令不执行重定向限位或碰撞检查。
已删除 `teleop_start/stop`、gate 模式、dry-run 和独立客户端 CLI；
使用 `GX16Client.request()` 或驱动 API 控制。服务在单个循环中处理串口与 ZMQ，
较慢的硬件读取也会延迟命令处理；`--state-hz` 控制读取频率。

## GEX Retarget 输入

- `--checkpoint-dir`：默认使用仓库内的 `src/resources/retarget`（安装后为包内
  `resources/retarget`）。其中 `last.pth` IK 权重与配套 `config.json` 包含
  `joint_order`、`fingertip_link` 映射和 `joint.lower/upper` 限位。
- `--calibration`：默认 `src/resources/retarget/calibration.npz`，包含 `calibration_qpos_deg`、`calibration_scale`、
  `calibration_rotation`，以及含 `reference_frame` 的 `metadata_json`。
- `--reference`：默认 `src/resources/retarget/reference.npy`，匹配的有限数值 `[T,21,3]` NPY，单位米。
  EX16 不测量小指，因此小指关键点为合成数据。

模型、示例标定及参考数据均随源码和 wheel 分发。
标定中的数值数组保持不变，仅将 `metadata_json` 中的旧格式名称和机器绝对路径
替换为中性格式名称和相对资源路径。仓库不提供录制、标定采集、回放或训练命令。
数组加载禁用 pickle；权重须匹配保留的 IK 结构，来源可信且有授权。
真机遥操作默认使用 CUDA，离线滑动条查看器也支持 CPU；
碰撞检测需要 SAPIEN 2.x 和包内独立的 GX16 碰撞 URDF。

## 查看器 Base 位姿

`viewer.py` 和 `retarget_viewer.py` 共用带版本号的 JSON 布局文件，
由 `--base-poses` 指定，默认 `~/.local/share/gex16/base_poses.json`。
只有点击保存才原子写入文件，重置布局不会覆盖已保存内容。
文件格式错误时显示错误并使用默认布局，直到显式保存或重新加载。

```json
{"version": 1, "bases": {
  "EX16": {"position": [-0.23, 0, 0], "wxyz": [1, 0, 0, 0]},
  "GX16": {"position": [0.23, 0, 0], "wxyz": [1, 0, 0, 0]}
}}
```

位置单位为米，四元数顺序为 w/x/y/z，加载时归一化。
Base 位姿仅变换场景根节点，不参与重定向的腕部局部坐标投影，也不改变真机坐标。
离线测试查看器不连接串口或 ZMQ 硬件节点，不发送控制消息。

## EX16 初始关节姿态

离线查看器通过 **Save initial joints** 单独保存启动关节角度。
`--initial-pose` 指定文件，默认 `~/.local/share/gex16/ex16_initial_pose.json`。
文件包含 `version: 1`、`joint_names: ["joint1", ..., "joint16"]`，以及 `urdf_deg`
中的 16 个有限角度值；角度必须在 EX16 滑动条限位内。
加载时校验关节顺序和限位，重新加载失败不会改变当前姿态。
启动时若无有效的已保存姿态则使用标定姿态。保存为显式原子操作，
滑动条归零或恢复标定姿态不会覆盖已保存的初始姿态。
