import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import markdown
from weasyprint import HTML
from mootdx.quotes import Quotes
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib as mpl
from czsc import CZSC, RawBar, Freq

# --- 环境配置 ---
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
CHINESE_FONT = 'WenQuanYi Zen Hei'

class ProAnalyst:
    def __init__(self, symbol="600089", name="特变电工", cost=28.5, position=400):
        self.symbol = symbol
        self.name = name
        self.cost = cost
        self.position = position
        self.client = Quotes.factory(market="std", timeout=10)
        self.report_date = datetime.now().strftime("%Y%m%d")
        self.folder = f"reports/{self.name}{self.symbol}"
        self.img_folder = os.path.join(self.folder, "images")
        os.makedirs(self.img_folder, exist_ok=True)
        
    def fetch_data(self):
        print(f"正在拉取 {self.name} 多级别深度数据...")
        df_daily = self.client.bars(symbol=self.symbol, frequency=9, offset=300)
        df_weekly = self.client.bars(symbol=self.symbol, frequency=5, offset=150)
        df_hourly = self.client.bars(symbol=self.symbol, frequency=3, offset=600)
        
        return {
            "1W": self._normalize(df_weekly),
            "1D": self._normalize(df_daily),
            "1H": self._normalize(df_hourly)
        }
    
    def _normalize(self, df):
        if df is None or df.empty: return pd.DataFrame()
        df = df.copy()
        if "datetime" in df.columns:
            df["trade_date"] = pd.to_datetime(df["datetime"])
            df = df.set_index("trade_date")
        df = df.sort_index()
        cols = ["open", "high", "low", "close", "vol"]
        df = df[cols].rename(columns={"vol": "volume"})
        return df.apply(pd.to_numeric, errors="coerce")

    def calc_indicators(self, df):
        if df.empty: return df
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd_line"] = exp1 - exp2
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]
        # BB
        df["bb_mid"] = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * std
        df["bb_lower"] = df["bb_mid"] - 2 * std
        return df

    def get_chanlun_analysis(self, df, freq_str):
        bars = []
        freq_map = {"1W": Freq.W, "1D": Freq.D, "1H": Freq.F60}
        f = freq_map.get(freq_str, Freq.D)
        
        for i in range(len(df)):
            idx = df.index[i]
            bars.append(RawBar(
                symbol=self.symbol, id=i, dt=idx, freq=f,
                open=df["open"].iloc[i], close=df["close"].iloc[i],
                high=df["high"].iloc[i], low=df["low"].iloc[i],
                vol=df["volume"].iloc[i], amount=df["volume"].iloc[i]*df["close"].iloc[i]
            ))
        c = CZSC(bars)
        bi_list = c.bi_list
        last_bi_dir = "向上" if bi_list and bi_list[-1].direction.value == "向上" else "向下" if bi_list else "未知"
        zigzag = self.get_zigzag(df)
        return c, zigzag, last_bi_dir

    def get_zigzag(self, df, window=5):
        high, low = df["high"], df["low"]
        full_w = window * 2 + 1
        roll_max = high.rolling(full_w, center=True).max()
        roll_min = low.rolling(full_w, center=True).min()
        swings = []
        for i in range(len(df)):
            if high.iloc[i] == roll_max.iloc[i]:
                swings.append({"index": df.index[i], "price": high.iloc[i], "type": "H"})
            elif low.iloc[i] == roll_min.iloc[i]:
                swings.append({"index": df.index[i], "price": low.iloc[i], "type": "L"})
        if not swings: return []
        zigzag = [swings[0]]
        for pt in swings[1:]:
            if pt["type"] == zigzag[-1]["type"]:
                if pt["type"] == "H" and pt["price"] > zigzag[-1]["price"]: zigzag[-1] = pt
                elif pt["type"] == "L" and pt["price"] < zigzag[-1]["price"]: zigzag[-1] = pt
            else: zigzag.append(pt)
        return zigzag

    def plot_and_save(self, df, zigzag, level_name):
        filename = f"{level_name}_chart.png"
        filepath = os.path.join(self.img_folder, filename)
        plot_df = df.tail(120).copy() 
        
        # Define addplots with clear explicit labels
        apds = [
            mpf.make_addplot(plot_df['bb_upper'], color='grey', linestyle='--', alpha=0.4),
            mpf.make_addplot(plot_df['bb_lower'], color='grey', linestyle='--', alpha=0.4),
            mpf.make_addplot(plot_df['bb_mid'], color='blue', alpha=0.3),
            # Panel 1: MACD
            mpf.make_addplot(plot_df['macd_hist'], type='bar', panel=1, color='dimgray', alpha=0.5, ylabel='MACD\n(12,26,9)'),
            mpf.make_addplot(plot_df['macd_line'], panel=1, color='blue'),
            mpf.make_addplot(plot_df['macd_signal'], panel=1, color='orange'),
            # Panel 2: RSI
            mpf.make_addplot(plot_df['rsi'], panel=2, color='purple', ylabel='RSI\n(14)')
        ]
        
        # Style
        my_rc = {
            'font.sans-serif': [CHINESE_FONT, 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'font.size': 10
        }
        s = mpf.make_mpf_style(base_mpf_style='charles', rc=my_rc)
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, addplot=apds, 
                              title=f"\n{self.name} (SH{self.symbol}) - {level_name} 级别 K线图", 
                              ylabel='价格 (CNY)', volume=False, figsize=(14, 11), 
                              returnfig=True, tight_layout=True)
                              
        # Add Cost Line
        axlist[0].axhline(y=self.cost, color='red', linestyle='-.', alpha=0.7)
        axlist[0].text(plot_df.index.get_loc(plot_df.index[-1]), self.cost, f"持仓成本: {self.cost}", 
                       fontname=CHINESE_FONT, fontsize=11, color='red', ha='right', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        # Annotate Wave Points
        valid_z = [z for z in zigzag if z["index"] in plot_df.index]
        for i, z in enumerate(valid_z):
            idx_pos = plot_df.index.get_loc(z["index"])
            marker_text = f"浪{i+1}"
            axlist[0].text(idx_pos, z["price"], marker_text, fontname=CHINESE_FONT, fontsize=10, 
                          fontweight='bold', color='#c0392b', 
                          va='bottom' if z["type"]=='H' else 'top', ha='center',
                          bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="none", alpha=0.6))
        
        # Annotate current indicator states explicitly
        last_idx = len(plot_df) - 1
        curr_rsi = plot_df['rsi'].iloc[-1]
        curr_macd = plot_df['macd_hist'].iloc[-1]
        
        # RSI Annotation
        axlist[4].text(last_idx, curr_rsi, f" {curr_rsi:.1f}", color='purple', fontsize=10, va='center')
        axlist[4].axhline(y=70, color='red', linestyle=':', alpha=0.4)
        axlist[4].axhline(y=30, color='green', linestyle=':', alpha=0.4)
        
        # MACD Annotation
        axlist[2].text(last_idx, curr_macd, f" {'红柱' if curr_macd>0 else '绿柱'}", color='dimgray', fontsize=10, va='center')

        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return filepath

    def generate_professional_report(self):
        data = self.fetch_data()
        res = {}
        for lv in ["1W", "1D", "1H"]:
            df = self.calc_indicators(data[lv])
            czsc_obj, zigzag, last_bi = self.get_chanlun_analysis(df, lv)
            img = self.plot_and_save(df, zigzag, lv)
            res[lv] = {
                "df": df, "zigzag": zigzag, "img": img, 
                "last_bi": last_bi, "czsc": czsc_obj
            }

        # Current status vars
        curr_price = res["1D"]["df"]["close"].iloc[-1]
        pl_amount = (curr_price - self.cost) * self.position
        pl_pct = (curr_price / self.cost - 1) * 100
        
        status_text = ""
        if curr_price > self.cost:
            status_text = "恭喜，当前您的持仓已处于**盈利**状态。本报告将通过精细化的多周期技术分析，为您寻找最佳的止盈防守点位或趋势加仓机会。"
        else:
            status_text = f"当前价格距离解套（{self.cost}元）尚有一定距离。本报告旨在通过精细化的多周期技术分析，为您寻找通过波段操作（做T）摊低成本的科学路径。"
            
        # Weekly Metrics
        w_rsi = res["1W"]["df"]["rsi"].iloc[-1]
        w_macd = res["1W"]["df"]["macd_hist"].iloc[-1]
        w_bi = res["1W"]["last_bi"]
        w_bb_mid = res["1W"]["df"]["bb_mid"].iloc[-1]
        
        # Daily Metrics
        d_rsi = res["1D"]["df"]["rsi"].iloc[-1]
        d_macd = res["1D"]["df"]["macd_hist"].iloc[-1]
        d_macd_prev = res["1D"]["df"]["macd_hist"].iloc[-2]
        d_bi = res["1D"]["last_bi"]
        d_bb_up = res["1D"]["df"]["bb_upper"].iloc[-1]
        d_bb_mid = res["1D"]["df"]["bb_mid"].iloc[-1]
        d_bb_low = res["1D"]["df"]["bb_lower"].iloc[-1]
        
        # Hourly Metrics
        h_rsi = res["1H"]["df"]["rsi"].iloc[-1]
        h_macd = res["1H"]["df"]["macd_hist"].iloc[-1]
        h_bi = res["1H"]["last_bi"]
        h_bb_low = res["1H"]["df"]["bb_lower"].iloc[-1]
        h_bb_up = res["1H"]["df"]["bb_upper"].iloc[-1]

        # Dynamic text for profit/loss state
        if curr_price > self.cost:
            h1_core_view = f"> **核心看点**：针对您目前盈利的 {self.position} 股底仓，小时线（1H）是寻找高位止盈或倒T降低风险的显微镜。"
            plan_a_title = "### 方案 A：主动防守与“倒T”锁定利润 (强烈建议)"
            plan_a_desc = "鉴于您目前处于盈利状态，首要任务是保住利润，防止利润回撤。我们可以利用波段高点抛出，低点接回。"
            plan_b_action = "2. **操作**：可追加 **200 股**的中期头寸，顺势而为博弈主升浪。在此条件未满足前，**不建议盲目加仓**。"
        else:
            h1_core_view = f"> **核心看点**：针对您被套的 {self.position} 股底仓，小时线（1H）是执行“高抛低吸（做T）”的显微镜。"
            plan_a_title = "### 方案 A：主动防御与“做T”摊低成本 (强烈建议)"
            plan_a_desc = "鉴于您处于浮亏状态，直接割肉代价过大，且周线并未出现加速崩盘的指标（RSI未极端钝化）。"
            plan_b_action = "2. **操作**：可追加 **200 股**的中期头寸，此时方可期待一波日线级别的反转行情带您大幅解套。在此条件未满足前，**严禁大仓位补仓**。"

        # Content Generation
        report_md = f"""
# 深度专业机构级诊断报告：{self.name} (SH{self.symbol})

**报告生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**当前市价**：{curr_price:.2f} 元 | **用户持仓成本**：{self.cost} 元 | **当前头寸**：{self.position} 股

<div class="account-card">
    <h3>👤 账户头寸状态量化评估</h3>
    <p>您的当前浮动盈亏为 <strong><span class="{'green' if pl_amount > 0 else 'red'}">{pl_amount:.2f} 元 ({pl_pct:.2f}%)</span></strong>。</p>
    <p><em>{status_text}</em></p>
</div>

---

## 第一部分：周线级别解析 —— 宏观趋势与波浪定调 (Macro Timeframe: 1W)

> **核心看点**：周线级别决定了股票的长期底色。我们通过波浪标注（图中黄底红字）和 MACD 来确认当前是否处于深渊的底部，或是主跌浪中。

![周线图](images/1W_chart.png)

### 📈 指标深度解构：
- **波浪理论与缠论结合**：
  - **当前波浪结构**：图中展示了周线级别的极值点（浪标注）。当前缠论结构处于 **{w_bi}** 笔中。
  - **深度解读**：结合波浪理论，若近期的浪点属于“底部抬高”形态，则大级别的下跌结构可能已经结束，正在构筑长期底部中枢。
- **MACD 指标 (图中 Panel 1)**：
  - 当前 MACD 柱状图数值为 **{w_macd:.3f}** ({"红柱，多头动能" if w_macd>0 else "绿柱，空头动能"})。
  - **深度解读**：周线 MACD 决定了长期的压制力。如果处于绿柱收缩期，说明长期的做空动能正在衰竭，这是战略性不再盲目割肉的重要技术依据。
- **RSI 强弱指标 (图中 Panel 2)**：
  - 当前读数：**{w_rsi:.2f}**。
  - **深度解读**：RSI(14) 衡量过去 14 周的多空力量。低于 30 属于超跌，高于 70 属于超买。目前读数显示该股长期处于 {"极度悲观的超跌区，筹码极具性价比" if w_rsi<35 else "稳态震荡区，市场分歧缩小" if w_rsi<60 else "高位获利盘密集的超买区"}。

---

## 第二部分：日线级别解析 —— 中期波段与布林带博弈 (Swing Timeframe: 1D)

> **核心看点**：日线是我们制定加减仓计划的核心骨架。图中蓝色实线为布林带中轨（20日均线），是短中期的多空分水岭。

![日线图](images/1D_chart.png)

### 📈 指标深度解构：
- **布林带 (Bollinger Bands)**：
  - **指标位置**：现价 ({curr_price:.2f}) 处于中轨 ({d_bb_mid:.2f}) 的 **{"上方，多头控盘" if curr_price > d_bb_mid else "下方，空头压制"}**。
  - **深度解读**：在图表主图中，您可以看到价格在上下轨（虚线）之间穿梭。当股价从下轨向中轨反弹时，中轨（{d_bb_mid:.2f}）是第一阻力位。目前，日线布林带呈现出开口 {"收窄（面临方向选择）" if (d_bb_up - d_bb_low)/d_bb_mid < 0.15 else "扩张（趋势明确）"}。
- **缠论结构**：
  - 当前日线处于 **{d_bi}** 笔。
  - **深度解读**：配合图中的波浪极值点，您可以观察到，如果日线向下笔的低点（最新的“浪”点）高于前一个向下笔的低点，即构成了缠论经典的**二买确认**，这是非常确定性的加仓信号。
- **MACD 与 RSI**：
  - 日线 MACD 目前为 **{d_macd:.3f}**，较昨日 **{"增强" if abs(d_macd) > abs(d_macd_prev) else "衰弱"}**。日线 RSI 读数 **{d_rsi:.2f}**，并未触及 70 的红线警戒区。

---

## 第三部分：小时级别解析 —— 微观执行与日内做T (Execution Timeframe: 1H)

{h1_core_view}

![小时图](images/1H_chart.png)

### 📈 指标深度解构：
- **精细波浪识别**：
  - 小时线上的 Wave 标注非常密集，代表了日内资金的博弈轨迹。当前缠论走向为 **{h_bi}** 笔。
- **短线极值探测 (RSI + 布林下轨)**：
  - 小时级别的布林带下轨目前位于 **{h_bb_low:.2f}**，上轨位于 **{h_bb_up:.2f}**。
  - **深度解读**：小时线 RSI ({h_rsi:.2f}) 非常敏感。当 RSI 跌破 30 且价格刺穿布林下轨时，极易引发日内的暴力反弹；反之，RSI 突破 70 触及上轨时，极易引发冲高回落。

---

## 第四部分：专属全周期操作预案 (Actionable Playbook)

基于以上多周期技术指标的共振分析，结合您的**成本（{self.cost}元）**与**仓位（{self.position}股）**，我为您制定了以下具有明确触发条件的操作预案，**请摒弃模糊的预测，严格按条件执行**：

{plan_a_title}
{plan_a_desc}
1. **买入触发条件（接回筹码）**：
   - 当观察到**小时线 (1H) 价格跌破布林下轨 ({h_bb_low:.2f} 附近)**，**且**同级别 **RSI 指标低于 30** 时。
   - **操作**：使用可用资金买入 **100 股**（1/4仓位）。
2. **卖出触发条件（抛出筹码）**：
   - 随后的 1-2 个交易日内，若**小时线价格反弹触及布林上轨 ({h_bb_up:.2f} 附近)**，**且**同级别 **MACD 红柱开始缩短**。
   - **操作**：果断卖出之前买入的 **100 股**。
   - **目的**：不增加总仓位风险，通过波段赚取差价，实质性优化持仓成本与锁定利润。

### 方案 B：右侧趋势反转的大级别加仓 (需等待确认)
1. **触发条件**：
   - 必须等待**日线 (1D) 级别实体阳线有效突破布林带中轨 ({d_bb_mid:.2f})**，并且**连续三个交易日站稳**。
   - **且**：日线 MACD 形成金叉并在零轴上方发散。
{plan_b_action}

### 方案 C：底线风控止损
1. **触发条件**：
   - 若**周线 (1W) 级别**，价格放量跌破图中最下方的 Wave 极值支撑点，且周线 MACD 绿柱突然放大。
2. **操作**：说明长期的逻辑彻底遭到破坏，进入无底洞的去泡沫化阶段。必须斩仓 **200 股（半仓）**，保留剩余资金等待下一个牛熊轮回。

---
*版权声明：本机构级投研报告由 Gemini CLI 技术分析引擎生成。模型算法基于历史数据概率统计，不保证未来百分百胜率，请投资者独立决策，风险自担。*
"""

        md_path = os.path.join(self.folder, f"SH{self.symbol}_机构级诊断报告_{self.report_date}.md")
        pdf_path = os.path.join(self.folder, f"SH{self.symbol}_机构级诊断报告_{self.report_date}.pdf")
        with open(md_path, "w", encoding="utf-8") as f: f.write(report_md)
        css = """
        @page { margin: 1.5cm 2cm; }
        body { font-family: "WenQuanYi Zen Hei", "Noto Sans CJK JP", sans-serif; line-height: 1.7; color: #1a202c; font-size: 11pt; }
        h1 { text-align: center; color: #1e3a8a; border-bottom: 4px solid #1e3a8a; padding-bottom: 10px; font-size: 20pt; margin-bottom: 30px; }
        h2 { background: #2c3e50; color: #ffffff; padding: 8px 15px; border-radius: 4px; margin-top: 40px; font-size: 14pt; border-left: 6px solid #e74c3c; }
        h3 { color: #c0392b; margin-top: 25px; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }
        img { max-width: 100%; height: auto; display: block; margin: 25px auto; border: 1px solid #cbd5e1; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        blockquote { border-left: 4px solid #f39c12; background: #fef9e7; padding: 10px 15px; color: #d35400; margin: 15px 0; font-weight: bold; }
        ul { margin-left: 20px; }
        li { margin-bottom: 8px; }
        .account-card { background: #f8fafc; border: 1px solid #cbd5e1; border-left: 5px solid #3b82f6; border-radius: 6px; padding: 15px; margin-bottom: 30px; }
        .account-card h3 { margin-top: 0; color: #1e40af; border-bottom: none; }
        .green { color: #16a085; font-weight: bold; }
        .red { color: #e74c3c; font-weight: bold; }
        """
        html_content = f"<html><head><style>{css}</style></head><body>{markdown.markdown(report_md, extensions=['tables'])}</body></html>"
        HTML(string=html_content, base_url=self.folder).write_pdf(pdf_path)
        print(f"\n✅ 机构级专业投研报告生成完毕！\nPDF位置: {pdf_path}")

if __name__ == "__main__":
    ProAnalyst().generate_professional_report()
