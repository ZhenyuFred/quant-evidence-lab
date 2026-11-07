# Quant Evidence Lab

用 Python 核对月度回测的收益、费用与净值，也可以用同一项目复现一个行业动量实验。自有 CSV 的本地核对只需 Python 3.11+、NumPy 和 pandas。

## 快速开始

在下载的仓库根目录执行以下命令。建议先创建并激活虚拟环境。

```sh
python -m pip install -e .
python run.py review-file --csv examples/synthetic_monthly.csv --dry-run
```

终端会给出 `outputs/imports/<哈希>/evidence.json` 的位置。样例使用三个月的**合成数据**，初始净值 100，最终净值 102.2416707。可在结果的 `imported_performance.metrics.total_return` 核对累计收益约 2.24167%。将 `--csv` 换成自己的路径即可重复检查。

CSV 需要七列，`month,gross_return,net_return,excess_return,turnover,cost,wealth`。月份须连续，收益以小数表示，例如 1% 填 0.01。工具检查每月的净收益与扣费是否一致、净值是否按净收益复利，并汇总收益、波动和回撤。数据来源、信号生成时间和样本外选择需要另外提供证据。详情见 [CSV 输入约定](examples/README.md)。

## 复现行业动量实验

```sh
python run.py research
python run.py verify
python run.py report
```

首次运行 `research` 会从 [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) 下载美国 12 行业月收益和三因子数据，在本地记录来源和哈希。后续运行复用并核验这份数据。该实验在 2000–2014 年的验证期选择动量参数，使用 2015 年开始的留出区间比较同一组行业的等权和逆波动率加权。结束日期与结果由实际下载的数据版本决定。

用浏览器打开 `outputs/Quant-Evidence-Lab.html` 查看结果。逐月收益和权重在 `outputs/research/`；离线浏览报告和复算 CSV 均无需 API Key。

## 可选的 AI 审查

```sh
python -m pip install -e ".[ai]"
python run.py review-file --help
```

在本机环境中设置自己的 `OPENAI_API_KEY`。先用 `--dry-run` 检查上面的证据摘要，再执行下面的付费命令。模型收到你的问题、指标摘要、完整性检查和可选来源声明。新建本地账本的默认预算上限为 38 美元，仅用于本项目请求的费用预留，实际账单见自己的 API 平台。

```sh
python run.py review-file --csv examples/synthetic_monthly.csv --question "净收益与费用核对后，可以支持哪些结论？"
python run.py budget
```

支持的模型和价格假设在 `quant_evidence/budget.py` 中；调用前应核对自己的模型权限和适用价格。`review` 可审查本机生成的行业研究，`evaluate` 可运行有标签的审查案例，两者也会产生 API 费用。

## 本地测试

```sh
python -m unittest tests.test_evidence_pack tests.test_backtest tests.test_data
```

安装 AI 可选依赖后，可运行 `python -m unittest discover -s tests` 执行完整测试。测试使用合成输入和模拟的网络响应。生成的数据缓存、报告、账本及私人 CSV 审查结果均在本机工作目录中。
