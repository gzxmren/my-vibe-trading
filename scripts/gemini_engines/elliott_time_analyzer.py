#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import markdown
from weasyprint import HTML
from mootdx.quotes import Quotes
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib as mpl

# --- 环境配置 ---
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
CHINESE_FONT = 'WenQuanYi Zen Hei'

class ElliottTimeAnalyst:
    """艾略特波浪时空矩阵分析引擎"""
    def __init__(self, symbol, name):
        self.symbol = symbol.replace(".SH", "").replace(".SZ", "")
        self.name = name
        self.client = Quotes.factory(market="std", timeout=10)
        self.report_date = datetime.now().strftime("%Y%m%d_%H%M")
        self.folder = f"reports/{self.name}{self.symbol}/时空波浪"
        self.img_folder = os.path.join(self.folder, "images")
        os.makedirs(self.img_folder, exist_ok=True)
        
        # 频率映射 (mootdx: 0=5m, 1=15m, 2=30m, 3=1H, 9=1D)
        # A股每天交易4小时，所以 4H K线等价于 1D K线
        self.tf_config = {
            "5分钟": {"freq": 0, "offset": 300, "zz_w": 5},
            "15分钟": {"freq": 1, "offset": 300, "zz_w": 5},
            "30分钟": {"freq": 2, "offset": 200, "zz_w": 4},
            "60分钟 (1H)": {"freq": 3, "offset": 150, "zz_w": 4},
            "1日 (等效4H)": {"freq": 9, "offset": 150, "zz_w": 3}
        }
        
    def fetch_all_timeframes(self):
        print(f"正在拉取 {self.name} ({self.symbol}) 多维度时空数据...")
        data_map = {}
        for tf_name, cfg in self.tf_config.items():
            df = self.client.bars(symbol=self.symbol, frequency=cfg["freq"], offset=cfg["offset"])
            if df is not None and not df.empty:
                df = self._normalize(df)
                data_map[tf_name] = df
                print(f"  - {tf_name} 数据获取成功: {len(df)} 条")
        return data_map
        
    def _normalize(self, df):
        df = df.copy()
        if "datetime" in df.columns:
            df["trade_date"] = pd.to_datetime(df["datetime"])
            df = df.set_index("trade_date")
        df = df.sort_index()
        cols = ["open", "high", "low", "close", "vol"]
        df = df[cols].rename(columns={"vol": "volume"})
        return df.apply(pd.to_numeric, errors="coerce")

    def get_zigzag_with_time(self, df, window=5):
        """获取带有时间跨度（K线根数）的波浪极值"""
        high, low = df["high"], df["low"]
        full_w = window * 2 + 1
        roll_max = high.rolling(full_w, center=True).max()
        roll_min = low.rolling(full_w, center=True).min()
        
        swings = []
        for i in range(len(df)):
            idx = df.index[i]
            if high.iloc[i] == roll_max.iloc[i]:
                swings.append({"index": idx, "pos": i, "price": high.iloc[i], "type": "H"})
            elif low.iloc[i] == roll_min.iloc[i]:
                swings.append({"index": idx, "pos": i, "price": low.iloc[i], "type": "L"})
                
        if not swings: return []
        
        zigzag = [swings[0]]
        for pt in swings[1:]:
            if pt["type"] == zigzag[-1]["type"]:
                if pt["type"] == "H" and pt["price"] > zigzag[-1]["price"]: zigzag[-1] = pt
                elif pt["type"] == "L" and pt["price"] < zigzag[-1]["price"]: zigzag[-1] = pt
            else: zigzag.append(pt)
            
        # 计算波浪的时间跨度 (Duration in bars) 和 价格振幅
        for i in range(1, len(zigzag)):
            zigzag[i]["duration"] = zigzag[i]["pos"] - zigzag[i-1]["pos"]
            zigzag[i]["amplitude"] = abs(zigzag[i]["price"] - zigzag[i-1]["price"]) / zigzag[i-1]["price"]
            
        zigzag[0]["duration"] = 0
        zigzag[0]["amplitude"] = 0.0
        return zigzag

    def plot_wave_chart(self, df, zigzag, tf_name):
        filename = f"wave_{tf_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        filepath = os.path.join(self.img_folder, filename)
        
        plot_df = df.copy()
        
        # 提取有效范围内的 zigzag 点
        valid_z = [z for z in zigzag if z["index"] in plot_df.index]
        
        # 构建 mplfinance 的 alines (连续线段)
        seq = [(z["index"], z["price"]) for z in valid_z]
        alines = dict(alines=seq, colors='#2980b9', linewidths=2, alpha=0.8)
        
        my_rc = {
            'font.sans-serif': [CHINESE_FONT, 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'font.size': 10
        }
        s = mpf.make_mpf_style(base_mpf_style='charles', rc=my_rc)
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, alines=alines,
                              title=f"\\n{self.name} ({self.symbol}) - {tf_name} 波浪与时空结构", 
                              ylabel='价格 (CNY)', volume=False, figsize=(14, 8), 
                              returnfig=True, tight_layout=True)
        
        # 标注时间跨度和波段性质
        for i, z in enumerate(valid_z):
            idx_pos = plot_df.index.get_loc(z["index"])
            
            # 标注极值点名称
            axlist[0].text(idx_pos, z["price"], f"{z['type']}{i}", 
                          fontname=CHINESE_FONT, fontsize=11, fontweight='bold', color='#c0392b',
                          va='bottom' if z["type"]=='H' else 'top', ha='center',
                          bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="none", alpha=0.7))
            
            # 标注这波运行的时间（K线根数）和空间（幅度）
            if i > 0 and z["duration"] > 0:
                mid_pos = (plot_df.index.get_loc(valid_z[i-1]["index"]) + idx_pos) / 2
                mid_price = (valid_z[i-1]["price"] + z["price"]) / 2
                
                info_text = f"{z['duration']}T\\n{z['amplitude']*100:.1f}%"
                axlist[0].text(mid_pos, mid_price, info_text, 
                              fontname=CHINESE_FONT, fontsize=9, color='#2c3e50',
                              ha='center', va='center',
                              bbox=dict(boxstyle="round,pad=0.1", fc="#ecf0f1", ec="none", alpha=0.8))

        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return filepath

    def analyze_logic(self, zigzag):
        """基于时间与空间跨度的波浪逻辑推演"""
        if len(zigzag) < 4:
            return "波段数据不足，尚处于单一结构中演化。"
            
        last_wave = zigzag[-1]
        prev_wave = zigzag[-2]
        
        # 判断当前运动方向
        curr_dir = "向上" if last_wave["type"] == "H" else "向下"
        
        # 分析时间对称性
        dur_last = last_wave["duration"]
        dur_prev = prev_wave["duration"]
        time_ratio = dur_last / dur_prev if dur_prev > 0 else 0
        
        # 分析空间力度
        amp_last = last_wave["amplitude"]
        amp_prev = prev_wave["amplitude"]
        space_ratio = amp_last / amp_prev if amp_prev > 0 else 0
        
        logic = f"**当前处于{curr_dir}波段中**。\\n"
        logic += f"- **时间跨度分析**：本波段已运行 {dur_last} 个时间单位（T）。前一逆向波段运行了 {dur_prev}T。时间比例为 {time_ratio:.2f}。"
        if time_ratio > 1.618:
            logic += " 本波段运行时间已显著超过前一波段，处于**时间延伸状态**，随时可能面临时间窗口变盘。"
        elif time_ratio < 0.618:
            logic += " 本波段运行时间较短，结构**尚未走完**，原趋势大概率将继续延续。"
        else:
            logic += " 本波段与前一波段达到**时间对称**，属于经典的调整或等距推动周期。"
            
        logic += f"\\n- **空间力度分析**：本波段振幅 {amp_last*100:.1f}%，前一波段振幅 {amp_prev*100:.1f}%。空间比例 {space_ratio:.2f}。"
        if space_ratio > 1:
            logic += f" 当前{curr_dir}力度**强于**前期，属于主导结构（推动浪）。"
        else:
            logic += f" 当前{curr_dir}力度**弱于**前期，暂定性为次级结构（调整浪）。"
            
        return logic

    def generate_report(self):
        data_map = self.fetch_all_timeframes()
        if not data_map:
            print("未能获取到数据。")
            return
            
        results = {}
        for tf_name, cfg in self.tf_config.items():
            if tf_name not in data_map: continue
            df = data_map[tf_name]
            # 计算 ZigZag 和时空跨度
            zigzag = self.get_zigzag_with_time(df, window=cfg["zz_w"])
            img_path = self.plot_wave_chart(df, zigzag, tf_name)
            logic_desc = self.分析逻辑 = self.analyze_logic(zigzag)
            
            results[tf_name] = {
                "zigzag": zigzag,
                "img": img_path,
                "logic": logic_desc,
                "last_price": df["close"].iloc[-1]
            }

        last_p = list(results.values())[-1]["last_price"]

        report_md = f"""
# 艾略特波浪时空矩阵诊断报告：{self.name} ({self.symbol})

**报告生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**最新市价参考**：{last_p:.2f} 元  
**分析方法**：多周期艾略特波浪理论 + 斐波那契时空跨度推演 (Time-Span & Amplitude Analysis)

<div class="intro-card">
    <h3>🌊 什么是“时空矩阵”分析？</h3>
    <p>传统的指标往往只关注价格（空间），而忽略了时间周期。本引擎在多周期K线图上绘制 ZigZag 波浪结构，并自动计算每一浪的<strong>运行K线数（T）</strong>和<strong>涨跌幅（%）</strong>。</p>
    <p><em>注：根据A股交易规则（每日交易4小时），4小时(4H)级别在结构上与日线(1D)完全等效，本报告合并分析。</em></p>
</div>

---
"""
        # 动态组装各级别的分析内容
        for tf_name in self.tf_config.keys():
            if tf_name not in results: continue
            res = results[tf_name]
            
            report_md += f"## {tf_name} 级别 —— 时空波浪解构\n\n"
            report_md += f"![{tf_name}图表]({os.path.basename(res['img'])})\n\n"
            
            report_md += "### ⏱️ 时空周期判定：\n"
            report_md += f"{res['logic']}\n\n"
            
            # 列出最近3段波浪的数据
            if len(res['zigzag']) >= 3:
                last3 = res['zigzag'][-3:]
                report_md += "| 浪形 | 类型 | 终点价 | 时间跨度 (T) | 空间振幅 |\n"
                report_md += "| :--- | :--- | :--- | :--- | :--- |\n"
                for i, z in enumerate(last3):
                    report_md += f"| {z['type']}{len(res['zigzag'])-3+i} | {'向上' if z['type']=='H' else '向下'} | {z['price']:.2f} | {z['duration']} 根 | {z['amplitude']*100:.1f}% |\n"
            report_md += "\n---\n"

        report_md += f"""
## 综合时空推演与执行预案

基于 5分钟至日线（4H） 的全息时空切片，得出以下综合演化路径：

1. **时空共振点寻找**：
   观察以上各周期表格中的“时间跨度(T)”。当小级别（如15分钟）运行的时间周期（如运行了13T或21T），恰好在大级别（如60分钟）的关键支撑/阻力位重合时，极易发生**时空共振变盘**。
2. **操作指导原则**：
   - 蓝色实线勾勒出了完整的波浪生命周期。图中框注了每一浪的运行时间和空间。
   - **大周期（1日/60分）**决定主仓位的买卖方向。若大周期处于时间延伸的向上波段，持仓即可。
   - **小周期（15分/5分）**用于寻找波段拐点。当小周期的下跌波段时间跨度接近前一上涨波段的 0.618 或 1.0 时，准备右侧入场。

*声明：本时空矩阵报告由专业量化算法自动生成。市场存在变数，请结合盘口流动性进行实战。*
"""

        md_path = os.path.join(self.folder, f"SH{self.symbol}_时空波浪报告_{self.report_date}.md")
        pdf_path = os.path.join(self.folder, f"SH{self.symbol}_时空波浪报告_{self.report_date}.pdf")
        with open(md_path, "w", encoding="utf-8") as f: f.write(report_md)
        
        css = """
        @page { margin: 1.5cm 2cm; }
        body { font-family: "WenQuanYi Zen Hei", "Noto Sans CJK JP", sans-serif; line-height: 1.7; color: #2c3e50; font-size: 11pt; }
        h1 { text-align: center; color: #8e44ad; border-bottom: 3px solid #8e44ad; padding-bottom: 10px; font-size: 20pt; margin-bottom: 20px; }
        h2 { background: #2980b9; color: #ffffff; padding: 6px 12px; border-radius: 4px; margin-top: 30px; font-size: 14pt; }
        h3 { color: #d35400; margin-top: 15px; margin-bottom: 5px; }
        img { max-width: 100%; height: auto; display: block; margin: 15px auto; border: 1px solid #bdc3c7; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ecf0f1; padding: 8px; text-align: center; }
        th { background-color: #f8f9fa; color: #2c3e50; }
        .intro-card { background: #f4f6f7; border-left: 5px solid #8e44ad; padding: 15px; margin-bottom: 20px; border-radius: 4px; }
        .intro-card h3 { margin-top: 0; color: #8e44ad; }
        """
        
        html_content = f"<html><head><style>{css}</style></head><body>{markdown.markdown(report_md, extensions=['tables'])}</body></html>"
        HTML(string=html_content, base_url=self.img_folder).write_pdf(pdf_path)
        print(f"\n✅ 艾略特波浪时空矩阵报告生成完毕！\nPDF位置: {pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成个股的波浪时空分析报告")
    parser.add_argument("--symbol", type=str, required=True, help="股票代码")
    parser.add_argument("--name", type=str, required=True, help="股票名称")
    
    args = parser.parse_args()
    analyst = ElliottTimeAnalyst(symbol=args.symbol, name=args.name)
    analyst.generate_report()
