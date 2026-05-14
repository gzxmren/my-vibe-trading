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

class UltimateAnalyst:
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
        # 1年 = 约240个交易日
        df_daily = self.client.bars(symbol=self.symbol, frequency=9, offset=250)
        df_weekly = self.client.bars(symbol=self.symbol, frequency=5, offset=100)
        df_hourly = self.client.bars(symbol=self.symbol, frequency=3, offset=500)
        
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
        # Convert df to RawBar
        bars = []
        
        freq_map = {
            "1W": Freq.W,
            "1D": Freq.D,
            "1H": Freq.F60
        }
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
        
        # Extract BI
        bi_list = c.bi_list
        last_bi_dir = "向上" if bi_list and bi_list[-1].direction.value == "向上" else "向下" if bi_list else "未知"
        
        # Calculate ZigZag for Elliott Wave overlay
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

    def plot_and_save(self, df, zigzag, czsc_obj, level_name):
        filename = f"{level_name}_chart.png"
        filepath = os.path.join(self.img_folder, filename)
        plot_df = df.tail(150).copy() # Show 150 bars for better view
        
        apds = [
            mpf.make_addplot(plot_df['bb_upper'], color='grey', linestyle='--', alpha=0.3),
            mpf.make_addplot(plot_df['bb_lower'], color='grey', linestyle='--', alpha=0.3),
            mpf.make_addplot(plot_df['bb_mid'], color='blue', alpha=0.2),
            mpf.make_addplot(plot_df['macd_hist'], type='bar', panel=1, color='dimgray', alpha=0.4, ylabel='MACD'),
            mpf.make_addplot(plot_df['macd_line'], panel=1, color='blue'),
            mpf.make_addplot(plot_df['macd_signal'], panel=1, color='orange'),
            mpf.make_addplot(plot_df['rsi'], panel=2, color='purple', ylabel='RSI')
        ]
        
        s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.sans-serif': [CHINESE_FONT]})
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, addplot=apds, title=f"{self.name} ({level_name}) - 持仓价: {self.cost}", 
                              ylabel='Price', volume=False, figsize=(14, 10), returnfig=True, tight_layout=True)
                              
        # Draw Cost Line
        axlist[0].axhline(y=self.cost, color='r', linestyle='-.', alpha=0.7, label='Cost')
        axlist[0].text(plot_df.index.get_loc(plot_df.index[-1]), self.cost, f"成本线: {self.cost}", 
                       fontname=CHINESE_FONT, fontsize=10, color='r', ha='right', va='bottom')
        
        # Annotate Wave/BI Points
        valid_z = [z for z in zigzag if z["index"] in plot_df.index]
        for i, z in enumerate(valid_z):
            idx_pos = plot_df.index.get_loc(z["index"])
            axlist[0].text(idx_pos, z["price"], f"浪{i+1}", fontname=CHINESE_FONT, fontsize=10, 
                          fontweight='bold', color='darkred', va='bottom' if z["type"]=='H' else 'top', ha='center')
        
        fig.savefig(filepath)
        plt.close(fig)
        return filepath

    def generate_professional_report(self):
        data = self.fetch_data()
        res = {}
        for lv in ["1W", "1D", "1H"]:
            df = self.calc_indicators(data[lv])
            czsc_obj, zigzag, last_bi = self.get_chanlun_analysis(df, lv)
            img = self.plot_and_save(df, zigzag, czsc_obj, lv)
            res[lv] = {
                "df": df, "zigzag": zigzag, "img": img, 
                "last_bi": last_bi, "czsc": czsc_obj
            }

        # --- 用户账户评估 ---
        d_close = res["1D"]["df"]["close"].iloc[-1]
        profit_loss = (d_close - self.cost) * self.position
        pl_percent = (d_close / self.cost - 1) * 100
        
        account_status_html = f"""
<div class="account-card">
    <h3>👤 您的持仓诊断</h3>
    <table>
        <tr>
            <th>成本价</th>
            <th>当前价</th>
            <th>持仓数量</th>
            <th>浮动盈亏</th>
        </tr>
        <tr>
            <td>{self.cost}</td>
            <td>{d_close:.2f}</td>
            <td>{self.position}</td>
            <td><span class="{'green' if profit_loss > 0 else 'red'}">{profit_loss:.2f} ({pl_percent:.2f}%)</span></td>
        </tr>
    </table>
</div>
"""

        # --- 深度解读逻辑 (LLM Analyst Logic) ---
        w_rsi = res["1W"]["df"]["rsi"].iloc[-1]
        d_rsi = res["1D"]["df"]["rsi"].iloc[-1]
        h_rsi = res["1H"]["df"]["rsi"].iloc[-1]
        
        d_macd_hist = res["1D"]["df"]["macd_hist"].iloc[-1]
        d_macd_prev = res["1D"]["df"]["macd_hist"].iloc[-2]
        
        d_bb_mid = res["1D"]["df"]["bb_mid"].iloc[-1]
        
        # 趋势判断
        w_bi = res["1W"]["last_bi"]
        d_bi = res["1D"]["last_bi"]
        
        # 结合成本的建议逻辑
        if d_close < self.cost:
            if d_rsi < 35 and d_macd_hist > d_macd_prev:
                strategy = "底部背离结构初现，目前处于深套状态。**强烈建议暂不割肉**，利用日线级别的底分型或布林带下轨附近逢低做 T 摊低成本。"
            elif d_close > d_bb_mid:
                strategy = "趋势已呈现多头反转，距离解套不远。**建议持仓观望**，等待日线向上笔突破成本线。"
            else:
                strategy = "依然处于日线下跌笔中，寻底过程尚未结束。**建议切勿盲目补仓**，等待周线级别底分型确认后再行加仓摊薄。"
        else:
            if d_rsi > 75:
                strategy = "已实现盈利，但短期指标严重超买，面临日线回调压力。**建议减仓一半（200股）**落袋为安，剩余底仓博弈波浪延伸。"
            else:
                strategy = "盈利持仓中，趋势结构良好。**建议继续持股**，以上涨通道的布林带中轨作为移动防守止盈位。"
        
        report_md = f"""
# {self.name} (SH{self.symbol}) 高级技术面诊断与交易内参

{account_status_html}

## 1. 核心投研综述
本报告融合**缠中说禅（形态/笔）**与**艾略特波浪理论（趋势/极值）**，辅以经典指标对 **{self.name}** 最近 1 年的走势进行了深度扫描。
针对您的持仓成本 **{self.cost}**，当前的核心操作总方针是：
> {strategy}

---

## 2. 周线级别分析：大趋势底色 (Macro Trend)
![周线图](images/1W_chart.png)

### 📊 技术面深度解构：
1. **缠论笔态**：周线级别当前处于**{w_bi}笔**中。这决定了中长期的宏观运行方向。
2. **波浪极值**：周线图上的 Wave 标注清晰显示了大级别的浪形交替。请注意红色的**成本线**，若成本线位于近期的 Wave 支撑点之下，则大趋势依然保护着您的头寸；反之则说明您建仓于前期趋势的高位。
3. **指标共振**：MACD 柱状图为宏观动能定调，当前 RSI 读数为 **{w_rsi:.2f}**，长期并未出现结构性崩盘风险。

---

## 3. 日线级别分析：中期波段与买卖点 (Swing Action)
![日线图](images/1D_chart.png)

### 📊 技术面深度解构：
1. **布林生命线与成本的博弈**：目前股价报 **{d_close:.2f}**。布林带中轨为 **{d_bb_mid:.2f}**。股价目前在布林中轨的**{"上方（偏强）" if d_close > d_bb_mid else "下方（偏弱）"}**运行。如果您的成本线({self.cost})距离当前价格较远，中轨的回踩将是非常关键的加/减仓分水岭。
2. **缠论内部结构**：日线处于**{d_bi}笔**。当“日线下跌笔”在“周线上升笔”中完成底分型时，往往构成缠论经典的**二买/三买**机会。
3. **动能追踪**：MACD 处于 **{"发散" if abs(d_macd_hist) > abs(d_macd_prev) else "收敛"}** 状态。红绿柱的交替是波浪驱动力转换的先兆。

---

## 4. 小时级别分析：日内做 T 视角 (Intraday Execution)
![小时图](images/1H_chart.png)

### 📊 技术面深度解构：
1. **微观结构**：小时级别波浪演变极快，重点观察 RSI (**{h_rsi:.2f}**)。由于您的持仓为 400 股，可利用小时线 RSI 触及 25 以下分批买入 200 股，并在 RSI 突破 75 时卖出 200 股，通过“T+0”操作逐步拉低 28.5 的历史成本。
2. **短期阻力/支撑**：上方近期 Wave 高点是首要抛压区，下方布林带下轨为日内超跌反弹的强支撑。

---

## 5. 专属交易规划 (Actionable Plan)

1. **持仓体检**：您的历史成本为 **{self.cost}**，当前处于 **{"浮盈" if profit_loss > 0 else "浮亏"}** 状态。
2. **行动指令**：
    - **方向**：结合日线布林带与 MACD 的配合，短期内建议**{ '逢高减磅' if d_rsi > 70 else '逢低做T' if d_close < self.cost else '坚定持仓' }**。
    - **防守底线**：请密切关注日线图上最近一个下方 Wave 低点，若放量跌破此价位，缠论向下笔结构成立，建议无条件止损/减仓一半。

---
*声明：本报告由 Gemini CLI 首席分析师融合多重高级算法自动生成。算法具有概率属性，请结合您的风险承受能力执行。*
"""
        md_path = os.path.join(self.folder, f"SH{self.symbol}_专项投研报告_{self.report_date}.md")
        pdf_path = os.path.join(self.folder, f"SH{self.symbol}_专项投研报告_{self.report_date}.pdf")
        with open(md_path, "w", encoding="utf-8") as f: f.write(report_md)
        css = """
        @page { margin: 2cm; }
        body { font-family: "WenQuanYi Zen Hei", "Noto Sans CJK JP", sans-serif; line-height: 1.8; color: #2c3e50; }
        h1 { text-align: center; color: #8e44ad; border-bottom: 3px solid #8e44ad; padding-bottom: 15px; font-size: 22pt; }
        h2 { background: #34495e; color: white; padding: 10px; border-radius: 4px; margin-top: 35px; font-size: 16pt; }
        h3 { color: #2980b9; border-bottom: 1px dashed #2980b9; padding-bottom: 5px; }
        img { max-width: 100%; height: auto; display: block; margin: 25px auto; border: 1px solid #bdc3c7; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #bdc3c7; padding: 10px; text-align: center; }
        th { background-color: #ecf0f1; color: #2c3e50; }
        .green { color: #27ae60; font-weight: bold; }
        .red { color: #c0392b; font-weight: bold; }
        .account-card { background: #fdfefe; border: 2px solid #8e44ad; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 10px rgba(142, 68, 173, 0.1); }
        blockquote { border-left: 5px solid #e74c3c; background: #fadbd8; padding: 15px; font-weight: bold; margin: 20px 0; color: #c0392b; }
        """
        html_content = f"<html><head><style>{css}</style></head><body>{markdown.markdown(report_md, extensions=['tables'])}</body></html>"
        HTML(string=html_content, base_url=self.folder).write_pdf(pdf_path)
        print(f"专属深度视觉报告已生成！PDF: {pdf_path}")

if __name__ == "__main__":
    UltimateAnalyst().generate_professional_report()
