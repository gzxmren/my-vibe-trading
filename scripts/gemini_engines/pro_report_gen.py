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
import subprocess
import json
from czsc import CZSC, RawBar, Freq

# --- 环境配置 ---
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
CHINESE_FONT = 'WenQuanYi Zen Hei'

class AutoOptimizer:
    """主动进化引擎：多维度参数空间联合寻优"""
    
    @staticmethod
    def optimize_strategy(df):
        """联合优化 RSI 阈值、布林带周期、布林带标准差、波浪识别窗口"""
        print("    [Auto-Tuning] 正在进行多因子联合沙盒回测 (RSI + 布林带 + 波浪模型)...")
        if df is None or len(df) < 100:
            return {"rsi_buy": 30, "rsi_sell": 70, "bb_w": 20, "bb_std": 2.0, "zz_w": 5, "score": 0, "win_rate": 0}
            
        best_score = -999
        best_params = {"rsi_buy": 30, "rsi_sell": 70, "bb_w": 20, "bb_std": 2.0, "zz_w": 5, "score": 0, "win_rate": 0}
        
        # 预计算固定的 RSI (RSI 周期一般保持 14 不变，只优化触发阈值)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        base_rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        # 参数空间遍历
        bb_windows = [15, 20, 30]
        bb_stds = [1.8, 2.0, 2.2]
        rsi_buys = [20, 25, 30, 35]
        rsi_sells = [65, 70, 75, 80, 85]
        
        for bb_w in bb_windows:
            for bb_std in bb_stds:
                # 动态生成该参数下的布林带
                bb_mid = df["close"].rolling(window=bb_w).mean()
                std = df["close"].rolling(window=bb_w).std()
                bb_lower = bb_mid - bb_std * std
                bb_upper = bb_mid + bb_std * std
                
                for b_th in rsi_buys:
                    for s_th in rsi_sells:
                        # 启动单次沙盒交易回测
                        position = 0
                        buy_price = 0
                        trades = 0
                        wins = 0
                        total_ret = 0.0
                        
                        for i in range(1, len(df)):
                            curr_rsi = base_rsi.iloc[i]
                            curr_price = df['close'].iloc[i]
                            curr_low = df['low'].iloc[i]
                            curr_high = df['high'].iloc[i]
                            
                            # 联合买入条件：RSI 超跌 且 价格触及布林带下轨
                            if curr_rsi < b_th and curr_low <= bb_lower.iloc[i] and position == 0:
                                position = 1
                                buy_price = curr_price
                            
                            # 联合卖出条件：RSI 超买 且 价格触及布林带上轨
                            elif curr_rsi > s_th and curr_high >= bb_upper.iloc[i] and position == 1:
                                position = 0
                                trade_ret = (curr_price - buy_price) / buy_price
                                total_ret += trade_ret
                                trades += 1
                                if trade_ret > 0: wins += 1
                                
                        # 结算最后一单
                        if position == 1:
                            trade_ret = (df['close'].iloc[-1] - buy_price) / buy_price
                            total_ret += trade_ret
                            trades += 1
                            if trade_ret > 0: wins += 1
                            
                        win_rate = (wins / trades) * 100 if trades > 0 else 0
                        
                        # 综合评分：倾向于收益率高且有一定交易频次的结果 (避免偶然孤本)
                        score = total_ret * (np.log(trades + 1)) 
                        
                        if score > best_score and trades >= 2:
                            best_score = score
                            best_params.update({
                                "rsi_buy": b_th, "rsi_sell": s_th,
                                "bb_w": bb_w, "bb_std": bb_std,
                                "score": score, "win_rate": win_rate
                            })
                            
        # 独立优化波浪理论(ZigZag)的识别窗口参数
        # 目标：寻找最符合艾略特波浪韵律的过滤周期（去除过多杂波，保留清晰趋势结构）
        best_zz_w = 5
        best_zz_score = 999
        for zz_w in [3, 5, 8, 13]:
            high, low = df["high"], df["low"]
            full_w = zz_w * 2 + 1
            roll_max = high.rolling(full_w, center=True).max()
            roll_min = low.rolling(full_w, center=True).min()
            swing_cnt = sum((high == roll_max) | (low == roll_min))
            
            # 对于 600 根 K 线，最理想的结构是呈现 10~20 个主干波段
            diff = abs(swing_cnt - 15)
            if diff < best_zz_score:
                best_zz_score = diff
                best_zz_w = zz_w
                
        best_params["zz_w"] = best_zz_w
        return best_params

class ProAnalyst:
    def __init__(self, symbol, name, cost, position):
        self.symbol = symbol.replace(".SH", "").replace(".SZ", "")
        self.name = name
        self.cost = float(cost)
        self.position = int(position)
        self.client = Quotes.factory(market="std", timeout=10)
        self.report_date = datetime.now().strftime("%Y%m%d")
        self.folder = f"reports/{self.name}{self.symbol}/多周期联合技术面"
        self.img_folder = os.path.join(self.folder, "images")
        os.makedirs(self.img_folder, exist_ok=True)
        
        # 预留存放最优参数的字典
        self.opt_params = {}
        
    def fetch_data(self):
        print(f"正在拉取 {self.name} ({self.symbol}) 多级别深度数据...")
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

    def calc_indicators(self, df, bb_w=20, bb_std=2.0):
        """使用动态寻优的参数计算指标"""
        if df.empty: return df
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd_line"] = exp1 - exp2
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]
        
        # 使用传入的最优布林带参数
        df["bb_mid"] = df["close"].rolling(window=bb_w).mean()
        std = df["close"].rolling(window=bb_w).std()
        df["bb_upper"] = df["bb_mid"] + bb_std * std
        df["bb_lower"] = df["bb_mid"] - bb_std * std
        return df

    def get_chanlun_analysis(self, df, freq_str, zz_w=5):
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
        zigzag = self.get_zigzag(df, window=zz_w)
        return c, zigzag, last_bi_dir

    def get_zigzag(self, df, window=5):
        """使用动态寻优的波段过滤参数进行识别"""
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
        
        apds = [
            mpf.make_addplot(plot_df['bb_upper'], color='grey', linestyle='--', alpha=0.4),
            mpf.make_addplot(plot_df['bb_lower'], color='grey', linestyle='--', alpha=0.4),
            mpf.make_addplot(plot_df['bb_mid'], color='blue', alpha=0.3),
            mpf.make_addplot(plot_df['macd_hist'], type='bar', panel=1, color='dimgray', alpha=0.5, ylabel='MACD\\n(12,26,9)'),
            mpf.make_addplot(plot_df['macd_line'], panel=1, color='blue'),
            mpf.make_addplot(plot_df['macd_signal'], panel=1, color='orange'),
            mpf.make_addplot(plot_df['rsi'], panel=2, color='purple', ylabel='RSI\\n(14)')
        ]
        
        my_rc = {
            'font.sans-serif': [CHINESE_FONT, 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'font.size': 10
        }
        s = mpf.make_mpf_style(base_mpf_style='charles', rc=my_rc)
        
        # 将动态参数显示在图表标题中
        bb_str = f"BB({self.opt_params['bb_w']},{self.opt_params['bb_std']})"
        zz_str = f"ZigZag({self.opt_params['zz_w']})"
        fig_title = f"\\n{self.name} (SH{self.symbol}) - {level_name} 级别 [自适应参数: {bb_str}, {zz_str}]"
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, addplot=apds, 
                              title=fig_title, 
                              ylabel='价格 (CNY)', volume=False, figsize=(14, 11), 
                              returnfig=True, tight_layout=True)
                              
        axlist[0].axhline(y=self.cost, color='red', linestyle='-.', alpha=0.7)
        axlist[0].text(plot_df.index.get_loc(plot_df.index[-1]), self.cost, f"持仓成本: {self.cost}", 
                       fontname=CHINESE_FONT, fontsize=11, color='red', ha='right', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        valid_z = [z for z in zigzag if z["index"] in plot_df.index]
        for i, z in enumerate(valid_z):
            idx_pos = plot_df.index.get_loc(z["index"])
            marker_text = f"浪{i+1}"
            axlist[0].text(idx_pos, z["price"], marker_text, fontname=CHINESE_FONT, fontsize=10, 
                          fontweight='bold', color='#c0392b', 
                          va='bottom' if z["type"]=='H' else 'top', ha='center',
                          bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="none", alpha=0.6))
        
        last_idx = len(plot_df) - 1
        curr_rsi = plot_df['rsi'].iloc[-1]
        curr_macd = plot_df['macd_hist'].iloc[-1]
        
        axlist[4].text(last_idx, curr_rsi, f" {curr_rsi:.1f}", color='purple', fontsize=10, va='center')
        
        # 绘制动态寻优后的自适应阈值线
        axlist[4].axhline(y=self.opt_params['rsi_sell'], color='red', linestyle=':', alpha=0.6)
        axlist[4].axhline(y=self.opt_params['rsi_buy'], color='green', linestyle=':', alpha=0.6)
        
        axlist[2].text(last_idx, curr_macd, f" {'红柱' if curr_macd>0 else '绿柱'}", color='dimgray', fontsize=10, va='center')

        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return filepath

    def generate_professional_report(self):
        data = self.fetch_data()
        if data["1H"].empty or data["1D"].empty:
            print("未能获取到完整数据，请检查股票代码是否正确。")
            return
            
        # --- 主动进化：全参数联合自动寻优 ---
        print(f"\\n🧠 激活核心大脑：开始执行「{self.name}」模型自适应拟合...")
        self.opt_params = AutoOptimizer.optimize_strategy(data["1H"])
        print(f"✨ 寻优完成！最终采纳自适应参数组：")
        print(f"  - 布林带中轨周期: {self.opt_params['bb_w']}，标准差乘数: {self.opt_params['bb_std']}")
        print(f"  - 震荡交易 RSI 阈值: <{self.opt_params['rsi_buy']} 买入, >{self.opt_params['rsi_sell']} 卖出")
        print(f"  - 缠论/波浪结构噪音过滤窗口: {self.opt_params['zz_w']}")
        print(f"  - 该策略组在回测期内胜率: {self.opt_params['win_rate']:.1f}%\\n")

        res = {}
        for lv in ["1W", "1D", "1H"]:
            # 使用最优参数重算所有指标和结构
            df = self.calc_indicators(data[lv], bb_w=self.opt_params['bb_w'], bb_std=self.opt_params['bb_std'])
            czsc_obj, zigzag, last_bi = self.get_chanlun_analysis(df, lv, zz_w=self.opt_params['zz_w'])
            img = self.plot_and_save(df, zigzag, lv)
            res[lv] = {
                "df": df, "zigzag": zigzag, "img": img, 
                "last_bi": last_bi, "czsc": czsc_obj
            }

        curr_price = res["1D"]["df"]["close"].iloc[-1]
        pl_amount = (curr_price - self.cost) * self.position
        pl_pct = (curr_price / self.cost - 1) * 100
        
        # --- 获取深度基本面数据 (mootdx F10) ---
        finance_info = {}
        try:
            print(f"\\n🗂️ 正在通过 mootdx 获取 {self.name} 深度财务报表 (F10)...")
            df_finance = self.client.finance(symbol=self.symbol)
            if not df_finance.empty:
                f_data = df_finance.iloc[0]
                eps = f_data['jinglirun'] / f_data['zongguben'] if f_data['zongguben'] > 0 else 0
                pe = curr_price / eps if eps > 0 else "亏损"
                pb = curr_price / f_data['meigujingzichan'] if f_data['meigujingzichan'] > 0 else 0
                finance_info = {
                    "净利润": f"{f_data['jinglirun']/100000000:.2f} 亿",
                    "每股净资产": f"{f_data['meigujingzichan']:.2f} 元",
                    "动态市盈率(PE)": pe if isinstance(pe, str) else f"{pe:.2f}",
                    "市净率(PB)": f"{pb:.2f}",
                    "总市值": f"{(f_data['zongguben']*curr_price)/100000000:.2f} 亿",
                    "流通盘": f"{(f_data['liutongguben']*curr_price)/100000000:.2f} 亿"
                }
                print("✅ 深度财务数据获取成功！")
            else:
                print("⚠️ 未获取到财务数据。")
        except Exception as e:
            print(f"⚠️ mootdx 财务数据获取失败: {e}")

        # --- 通过 LLM 生成多维度综合诊断 ---
        print("🧠 正在通过本地 Gemini CLI 引擎进行技术面与基本面综合推理...")
        
        w_rsi = res["1W"]["df"]["rsi"].iloc[-1]
        w_macd = res["1W"]["df"]["macd_hist"].iloc[-1]
        w_bi = res["1W"]["last_bi"]
        
        d_rsi = res["1D"]["df"]["rsi"].iloc[-1]
        d_macd = res["1D"]["df"]["macd_hist"].iloc[-1]
        d_bi = res["1D"]["last_bi"]
        d_bb_mid = res["1D"]["df"]["bb_mid"].iloc[-1]
        
        h_rsi = res["1H"]["df"]["rsi"].iloc[-1]
        h_bb_low = res["1H"]["df"]["bb_lower"].iloc[-1]
        h_bb_up = res["1H"]["df"]["bb_upper"].iloc[-1]
        
        opt_b = self.opt_params['rsi_buy']
        opt_s = self.opt_params['rsi_sell']

        prompt = f"""
你是一位顶级的 A 股基金经理。请基于以下量化数据，为用户持有的 {self.name}({self.symbol}) 撰写一份深度的【基本面与技术面共振诊断报告】。

【持仓状态】
- 现价: {curr_price:.2f}
- 成本: {self.cost}
- 股数: {self.position}
- 当前盈亏: {pl_amount:.2f}元 ({pl_pct:.2f}%)

【基本面 F10 数据】
{json.dumps(finance_info, ensure_ascii=False, indent=2)}

【技术面与自适应模型数据】
- 周线(1W): 缠论笔走向[{w_bi}], MACD柱[{w_macd:.3f}], RSI[{w_rsi:.2f}]
- 日线(1D): 缠论笔走向[{d_bi}], 价格位于自适应布林带中轨({d_bb_mid:.2f})的{'上方' if curr_price>d_bb_mid else '下方'}
- 小时线(1H): 布林上轨[{h_bb_up:.2f}], 下轨[{h_bb_low:.2f}], 引擎最佳交易 RSI 阈值(买入<{opt_b}, 卖出>{opt_s})

【请输出以下 Markdown 格式报告，不要废话】：
### 1. 深度基本面诊断
（根据PE、市值、净利润等，分析该股的基本面底色是优质蓝筹还是题材博弈）

### 2. 技术面与波段分析
（结合周线大趋势与日/小时线小周期，诊断目前是处于主升浪、洗盘还是主跌浪）
⚠️【非常重要：图文并茂排版要求】⚠️
在撰写“技术面与波段分析”这一章节时，你必须在提到周线、日线、小时线分析的具体段落下方，**分别原样插入**以下三张图片的 Markdown 占位符。不要把它们堆在报告最后，必须让图表紧跟在你的分析文字之后！
- 周线分析文字写完后，换行插入：`![周线图](images/1W_chart.png)`
- 日线分析文字写完后，换行插入：`![日线图](images/1D_chart.png)`
- 小时线分析文字写完后，换行插入：`![小时图](images/1H_chart.png)`

### 3. 给用户的专属操作预案
（结合用户的盈亏状态和小时线的自适应布林带与RSI阈值，给出具体的加仓、做T或止损建议点位）
"""
        
        try:
            result = subprocess.run(["gemini", "-p", prompt], capture_output=True, text=True, check=True)
            ai_insights = result.stdout.strip()
            print("✅ LLM 深度诊断生成完毕。")
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'stderr') and e.stderr:
                error_msg += f" | {e.stderr}"
            print(f"⚠️ LLM 调用失败 ({error_msg})，使用基础模板。")
            ai_insights = "> ⚠️ AI 引擎调用失败，无法提供深度推理研报。\n\n![周线图](images/1W_chart.png)\n![日线图](images/1D_chart.png)\n![小时图](images/1H_chart.png)"

        report_md = f"""
# 进化版机构级投研诊断：{self.name} (SH{self.symbol})

**报告生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**当前市价**：{curr_price:.2f} 元 | **用户持仓成本**：{self.cost} 元 | **当前头寸**：{self.position} 股

<div class="account-card">
    <h3>👤 账户头寸状态量化评估</h3>
    <p>您的当前浮动盈亏为 <strong><span class="{'green' if pl_amount > 0 else 'red'}">{pl_amount:.2f} 元 ({pl_pct:.2f}%)</span></strong>。</p>
</div>

<div class="auto-tune-card">
    <h3>🧬 联合沙盒自适应寻优结果 (Auto-Tuning Engine)</h3>
    <p>针对该标的股性的最优定制参数组合已覆盖至全维度的计算与图表中：</p>
    <ul>
        <li><strong>布林带通道</strong>：优化为 <strong>{self.opt_params['bb_w']} 日线</strong>，标准差为 <strong>{self.opt_params['bb_std']}</strong>。</li>
        <li><strong>波浪噪音过滤</strong>：极值识别窗口优化为 <strong>{self.opt_params['zz_w']}</strong> 根 K 线。</li>
        <li><strong>交易信号共振阈值</strong>：RSI 低于 <strong>{opt_b}</strong> 买入，高于 <strong>{opt_s}</strong> 卖出。历史胜率：<strong>{self.opt_params['win_rate']:.1f}%</strong>。</li>
    </ul>
</div>

---

{ai_insights}

---
*版权声明：本【联合进化版】机构级投研报告由 Gemini CLI 计算引擎动态生成。参数经过严谨的沙盒回测定序，请严格依纪律执行，风险自担。*
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
        .account-card { background: #f8fafc; border: 1px solid #cbd5e1; border-left: 5px solid #3b82f6; border-radius: 6px; padding: 15px; margin-bottom: 15px; }
        .account-card h3 { margin-top: 0; color: #1e40af; border-bottom: none; }
        .auto-tune-card { background: #fdfefe; border: 1px dashed #8e44ad; border-left: 5px solid #8e44ad; border-radius: 6px; padding: 15px; margin-bottom: 30px; }
        .auto-tune-card h3 { margin-top: 0; color: #8e44ad; border-bottom: none; }
        .green { color: #16a085; font-weight: bold; }
        .red { color: #e74c3c; font-weight: bold; }
        """
        html_content = f"<html><head><style>{css}</style></head><body>{markdown.markdown(report_md, extensions=['tables'])}</body></html>"
        HTML(string=html_content, base_url=self.folder).write_pdf(pdf_path)
        print(f"\n✅ 机构级专业投研报告生成完毕！\nPDF位置: {pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成个股的机构级多周期技术面诊断报告")
    parser.add_argument("--symbol", type=str, required=True, help="股票代码，如 600089")
    parser.add_argument("--name", type=str, required=True, help="股票名称，如 特变电工")
    parser.add_argument("--cost", type=float, required=True, help="持仓成本")
    parser.add_argument("--position", type=int, required=True, help="持仓股数")
    
    args = parser.parse_args()
    
    analyst = ProAnalyst(symbol=args.symbol, name=args.name, cost=args.cost, position=args.position)
    analyst.generate_professional_report()
