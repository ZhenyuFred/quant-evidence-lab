# 审查你自己的月度研究

`review-file` 用于审查你**明确选择**的一份本地月度回测 CSV。程序先在本机验证结构和算术，形成证据摘要，再让模型针对你提出的具体问题进行审查。

**数据边界：**构建证据摘要本身不联网、不调用 API，也不读取 `.env`。实际运行 `review-file` 的模型审查会将所选数据的**摘要指标、完整性检查、哈希，以及你提供的来源/参数选择/信号时点声明**发送给官方 OpenAI API；不会发送全部原始月度 CSV，也不会扫描其他研究文件。你可先在本地构建、查看摘要，再决定是否调用模型。元数据中也不要放入密钥、账号或不希望外发的私人信息。

## CSV 契约

必须且仅包含以下七列，可以调整列顺序：

```csv
month,gross_return,net_return,excess_return,turnover,cost,wealth
2024-01,0.02,0.019,0.017,1.0,0.001,101.9
2024-02,-0.01,-0.0105,-0.0125,0.5,0.0005,100.83005
```

上例仅用于解释格式，不是市场数据。初始净值为 100，且 `101.9 × (1 − 0.0105) = 100.83005`。

- `month` 为 `YYYY-MM`；日期必须唯一、递增、连续。缺失月份不会自动补齐。
- 收益为小数：`0.01` 表示 1%，不能写成 `1` 或 `1%` 来表达 1%。`cost` 是按组合规模计的收益扣减；`turnover` 是买卖名义金额与组合规模之比。
- `net_return = gross_return − cost`。成本与换手不能为负，净收益必须高于 −100%。
- `wealth` 必须为正，并与净收益复利一致；允许任何共同的正初始净值，程序由第一行反推初始净值，不假定其单位是美元。
- `excess_return` 是你提供的超额收益。程序会用它计算夏普，但不会自动确认其中的无风险利率是真实、适用且时间对齐的。

用 `evidence_metadata.json` 明确声明 `"return_units": "decimal_returns"`。如果省略元数据，则按现有月度 CSV 契约解释为小数收益，并在摘要中标明这是默认约定、不是独立验证。未明确声明单位而出现绝对值大于 1 的收益时，程序会拒绝并要求明确单位；不会猜测或自动除以 100。若写 `"units": "percent"` 或其他单位，也会拒绝。

元数据模板中的文本只是占位说明，必须替换成实际情况；不了解的字段直接删除。`source`、`selection`、`signal_timing` 的缺失值标为 `unknown`，提供的值标为 `unverified_user_statement`。你写了一个网址、截止日期或“无未来函数”，都不意味着程序已经核实了它。

## 先在本地查看证据包

可直接使用离线命令，保存位置会显示在终端：

```powershell
python run.py review-file --csv "examples/synthetic_monthly.csv" --dry-run
```

在项目目录运行以下 Python；把路径替换成你实际选择的文件：

```python
from pathlib import Path
import json
from quant_evidence.evidence_pack import build_evidence_pack

pack = build_evidence_pack(
    Path("examples/synthetic_monthly.csv"),
    None,  # 可改成你填写好的元数据 JSON 路径
)
print(json.dumps(pack, ensure_ascii=False, indent=2))
```

这个步骤没有任何 API 费用。返回四组证据：`imported_performance`、`imported_integrity`、`imported_provenance`、`imported_limits`，不包含原始月度行或本机绝对路径。

## 调用真实模型审查

完成安全 API 配置并确认准备外发的摘要内容后：

```powershell
python run.py review-file --help
python run.py review-file --csv "我的研究\monthly.csv" --metadata "我的研究\metadata.json" --question "这份结果是否足以支持我声称在未触碰测试期取得了无未来数据泄漏的超额收益？"
python run.py budget
```

`--metadata` 可省略；缺失声明不会由模型补造。`--question` 应是具体可检查的研究判断，例如费用口径是否自洽、某个收益数值是否能从证据推出、或是否缺少证明样本外设计的信息。这个命令会产生真实 API 用量，并受项目已有预算账本约束。

## 它能确认什么，不能确认什么

能确认：文件结构、月份连续性、有限数值、成本非负、净收益减费等式、净值复利关系，以及在上述口径下的确定性统计量。CSV 哈希用于识别这次审查对应的确切字节。

不能仅凭这份收益序列确认：数据源真实、数据当时可得、研究没有未来信息泄漏、参数未按测试期表现选择、手续费/滑点全部建模、策略可以实际交易或未来继续盈利。**哈希不等于真实来源证明，算术正确不等于研究设计正确。**需要核实这些问题时，应再提供相应的取数记录、信号代码、逐期持仓、时间戳或实验选择记录，而不是让 AI 猜测。
