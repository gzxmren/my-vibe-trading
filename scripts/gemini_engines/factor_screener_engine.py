#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import markdown
from weasyprint import HTML
import akshare as ak
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import pi

# 引入项目根目录以加载 agent 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from agent.src.providers.chat import ChatLLM

# --- 环境配置 ---
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
CHINESE_FONT = 'WenQuanYi Zen Hei'

class FactorScreenerPro:
    def __init__(self, top_n=20):
        self.top_n = top_n
        self.report_date = datetime.now().strftime("%Y%m%d_%H%M")
        self.folder = "reports/多因子智能选股雷达"
        self.img_folder = os.path.join(self.folder, "images")
        os.makedirs(self.img_folder, exist_ok=True)
        self.report_type = "technical"
        self.industry_map = {}
        
    def _fetch_industry_mapping(self):
        """尝试获取行业映射关系 (东方财富行业板块)"""
        print("尝试拉取全市场行业映射数据...")
        try:
            # 获取行业板块列表
            boards = ak.stock_board_industry_name_em()
            # 由于全量遍历太慢，如果只需展示，我们可以跳过实时拉取全量映射，或者只对 TopN 单独拉取。
            # 这里为了不阻塞，我们在后续处理中直接从股票代码进行简单的分类推演，或者如果在东方财富接口成功时自带板块。
            # 为了演示，此处我们先 pass，在 TopN 生成时再尝试单股获取。
            pass
        except Exception as e:
            print(f"行业数据拉取受限: {e}")

    def get_stock_industry(self, symbol):
        """对单一股票获取行业，增加容错"""
        try:
            info = ak.stock_individual_info_em(symbol=symbol)
            industry = info[info['item'] == '行业']['value'].values[0]
            return industry
        except:
            clean_symbol = str(symbol).lower().replace('sh', '').replace('sz', '')
            if clean_symbol.startswith('60'): return "沪市主板"
            if clean_symbol.startswith('00'): return "深市主板"
            if clean_symbol.startswith('30'): return "创业板"
            if clean_symbol.startswith('68'): return "科创板"
            if clean_symbol.startswith('8') or clean_symbol.startswith('4') or clean_symbol.startswith('9'): return "北交所/新三板"
            return "未知板块"

    def fetch_market_data(self):
        print("正在尝试获取全市场 A 股横截面数据...")
        
        # 优先从本地数据库加载
        db_path = 'data/fundamentals.db'
        if os.path.exists(db_path):
            print(f"📦 发现本地基本面数据库 ({db_path})，正在极速加载数据...")
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                df = pd.read_sql('SELECT * FROM a_share_fundamentals', conn)
                conn.close()
                print(f"✅ 成功从本地加载 {len(df)} 只股票的财务与量价数据！")
                df['data_type'] = 'fundamental_local'
                self.report_type = 'fundamental'
                return df
            except Exception as e:
                print(f"⚠️ 读取本地数据库失败: {e}，将尝试在线拉取...")

        print("📡 本地无缓存，正在从东方财富接口拉取全市场数据...")
        max_retries = 1
        for attempt in range(max_retries):
            try:
                df = ak.stock_zh_a_spot_em()
                print(f"✅ 成功获取 {len(df)} 只股票数据 (含基本面)。")
                self.report_type = 'fundamental'
                df['data_type'] = 'fundamental'
                return df
            except Exception as e:
                print(f"⚠️ 主接口拉取失败: {e}")
                time.sleep(1)
                
        print("🔄 尝试使用新浪财经备用接口 (仅量价因子)...")
        try:
            df = ak.stock_zh_a_spot()
            print(f"✅ 成功获取 {len(df)} 只股票数据 (仅量价)。")
            self.report_type = 'technical'
            df['data_type'] = 'technical_sina'
            return df
        except Exception as e:
            print(f"❌ 备用接口拉取失败: {e}")
            
        print("❌ 全市场数据拉取失败，请检查网络状态或运行 update_fundamentals.py。")
        return pd.DataFrame()

    def process_data(self, df):
        if df.empty: return df
            
        print("开始清洗数据与量化因子计算...")
        
        if self.report_type == 'fundamental':
            cols_map = {
                "代码": "symbol", "名称": "name", "最新价": "close", "涨跌幅": "momentum",
                "换手率": "turnover", "市盈率-动态": "pe", "市净率": "pb", "总市值": "mkt_cap"
            }
            df = df.rename(columns=cols_map)
            df = df[~df['name'].str.contains('ST|退', na=False)]
            df = df[~df['symbol'].str.startswith(('8', '4'), na=False)]
            for col in ["close", "momentum", "turnover", "pe", "pb", "mkt_cap"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["pe", "pb", "mkt_cap", "momentum"])
            df = df[(df["pe"] > 0) & (df["pb"] > 0)]
            
            df["factor_value"] = 1.0 / df["pe"]
            df["factor_quality"] = 1.0 / df["pb"]
            df["factor_momentum"] = df["momentum"]
            df["factor_size"] = -np.log(df["mkt_cap"])
            self.factor_cols = ["factor_value", "factor_quality", "factor_momentum", "factor_size"]
            self.factor_names = ["价值估值", "资产质量", "短期动能", "小盘效应"]
            weights = {"factor_value": 0.4, "factor_quality": 0.2, "factor_size": 0.2, "factor_momentum": 0.2}
            
        else:
            cols_map = {
                "代码": "symbol", "名称": "name", "最新价": "close", "涨跌幅": "momentum",
                "最高": "high", "最低": "low", "成交额": "amount"
            }
            df = df.rename(columns=cols_map)
            df = df[~df['name'].str.contains('ST|退', na=False)]
            df = df[~df['symbol'].str.startswith(('8', '4'), na=False)]
            for col in ["close", "momentum", "high", "low", "amount"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close", "momentum", "high", "low", "amount"])
            df = df[df["close"] > 0]
            
            df["factor_momentum"] = df["momentum"] 
            df["factor_stability"] = -((df["high"] - df["low"]) / df["close"]) 
            df["factor_liquidity"] = np.log(df["amount"] + 1) 
            df["factor_price"] = -np.log(df["close"]) 
            self.factor_cols = ["factor_momentum", "factor_stability", "factor_liquidity", "factor_price"]
            self.factor_names = ["多头动能", "日内稳定", "资金流爆度", "低价弹性"]
            weights = {"factor_momentum": 0.4, "factor_stability": 0.2, "factor_liquidity": 0.2, "factor_price": 0.2}
        
        # Z-Score 标准化
        for col in self.factor_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[col] = df[col].clip(lower=mean_val - 3*std_val, upper=mean_val + 3*std_val)
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
            
        df["score"] = 0
        for col, w in weights.items():
            df["score"] += df[col] * w
            
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        return df

    def draw_radar_chart(self, top_stocks):
        """生成风格画像卡片（雷达图）"""
        print("正在生成风格雷达图...")
        fig_path = os.path.join(self.img_folder, "style_radar.png")
        
        # 计算 TopN 的因子均值
        mean_scores = top_stocks[self.factor_cols].mean().values.tolist()
        # 为了雷达图好看，将 Z-score 映射到 0-100 区间 (假设 0 是均值，3是极高)
        mapped_scores = [max(0, min(100, (s + 2) / 4 * 100)) for s in mean_scores]
        
        categories = self.factor_names
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        mapped_scores += mapped_scores[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], categories, fontname=CHINESE_FONT, size=12)
        
        ax.plot(angles, mapped_scores, linewidth=2, linestyle='solid', color='#8e44ad')
        ax.fill(angles, mapped_scores, color='#9b59b6', alpha=0.4)
        
        plt.title("入选股票池的整体风格画像", size=15, color='#2c3e50', y=1.1, fontname=CHINESE_FONT)
        plt.savefig(fig_path, dpi=150, transparent=True)
        plt.close()
        return fig_path

    def generate_dynamic_insights(self, top_stocks):
        """动态分析逻辑：通过大模型进行深度推理"""
        print("🧠 正在调用 LLM 进行深度穿透分析...")
        mean_scores = top_stocks[self.factor_cols].mean()
        dominant_factor_idx = np.argmax(mean_scores.values)
        dominant_factor_name = self.factor_names[dominant_factor_idx]
        
        avg_momentum = top_stocks['momentum'].mean()
        
        # 获取行业分布
        industries = []
        for sym in top_stocks['symbol'].tolist():
            industries.append(self.get_stock_industry(sym))
        top_stocks['industry'] = industries
        
        # 组装传给大模型的数据上下文
        stocks_info = []
        for i, row in top_stocks.iterrows():
            stocks_info.append(f"{i+1}. {row['name']} ({row['symbol']}) - 行业: {row['industry']}, 涨幅: {row['momentum']}%, 综合得分: {row['score']:.2f}")
        stocks_text = "\n".join(stocks_info)
        
        prompt = f"""
你是一位顶级的 A 股量化基金经理。请根据以下多因子选股引擎筛选出的今日 Top {self.top_n} 股票数据，撰写一段专业的动态深度点评报告。

【数据上下文】
- 主导因子：{dominant_factor_name} (说明当前市场 Alpha 偏向于该维度)
- 平均日涨幅：{avg_momentum:.2f}%
- 入选标的列表（按得分排序）：
{stocks_text}

【你的任务】
请直接输出 Markdown 格式的点评，切忌废话、不要套话，要求言之有物：
1. **市场风格与主线研判**：通过这 {self.top_n} 只股票的行业分布和涨幅，分析当前市场是在炒作什么主线？资金在抱团还是在游资炒妖？
2. **前三名金股专属短评**：针对前3名的标的，各给出50字左右的一句话核心逻辑点评。
3. **操作风险提示**：基于上述风格特征（如平均涨幅过高、板块过度集中等），给出具体的建仓或避险建议。

输出要求：结构清晰，使用标题和列表，直接开始输出报告内容即可。
"""
        
        try:
            print("正在通过本地 Gemini CLI 获取大模型深度分析...")
            import subprocess
            # 使用本地 gemini cli 代理执行大模型调用，利用现有的 Google One AI Pro 套餐，无需额外 API Key
            result = subprocess.run(["gemini", "-p", prompt], capture_output=True, text=True, check=True)
            response_text = result.stdout.strip()
            insight_md = "### 🧠 核心大脑动态洞察 (AI Engine Insights)\n\n" + response_text
            print("✅ LLM 深度分析生成完毕。")
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'stderr') and e.stderr:
                error_msg += f" | {e.stderr}"
            print(f"⚠️ LLM 调用失败 ({error_msg})，降级为基础报告。")
            insight_md = f"### ⚠️ 核心大脑动态洞察\n\nAI 引擎调用失败，当前为主导因子 [{dominant_factor_name}]，平均涨幅 {avg_momentum:.2f}%。"

        return insight_md, top_stocks

    def generate_report(self):
        raw_df = self.fetch_market_data()
        if raw_df.empty: return
            
        df = self.process_data(raw_df)
        top_stocks = df.head(self.top_n)
        
        # 1. 绘制雷达图
        self.draw_radar_chart(top_stocks)
        
        # 2. 绘制因子贡献柱状图
        fig_path = os.path.join(self.img_folder, "factor_contribution.png")
        plt.figure(figsize=(12, 8))
        plot_df = top_stocks.head(10).copy()
        plot_df = plot_df.set_index("name")
        plot_df[self.factor_cols].plot(kind='bar', stacked=True, colormap='plasma', figsize=(12, 6))
        plt.legend(self.factor_names, prop={'family': CHINESE_FONT})
        plt.title("Top 10 金股：多因子得分贡献拆解", fontname=CHINESE_FONT, fontsize=16)
        plt.xlabel("股票名称", fontname=CHINESE_FONT)
        plt.ylabel("标准化因子得分 (Z-Score)", fontname=CHINESE_FONT)
        plt.xticks(rotation=45, fontname=CHINESE_FONT)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        # 3. 动态洞察
        insights_md, top_stocks = self.generate_dynamic_insights(top_stocks)

        # 4. 生成 Markdown 报告
        if self.report_type == "fundamental":
            strategy_desc = "基于市盈率、市净率和量价动量，捕捉戴维斯双击的绝对价值标的。"
            table_headers = "| 排名 | 股票代码 | 股票名称 | 所属板块 | 综合得分 | 动态PE | 市净率 | 总市值(亿) | 日涨幅 |\n| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
            table_rows = ""
            for i, row in top_stocks.iterrows():
                mkt_cap_y = row['mkt_cap'] / 1e8 if pd.notna(row['mkt_cap']) else 0
                table_rows += f"| {i+1} | {row['symbol']} | **{row['name']}** | {row['industry']} | {row['score']:.2f} | {row['pe']:.2f} | {row['pb']:.2f} | {mkt_cap_y:.1f} | <span class='{'red' if row['momentum']>0 else 'green'}'>{row['momentum']}%</span> |\n"
        else:
            strategy_desc = "在缺乏基础财务数据的容错模式下，引擎自动降级为【纯量价动能侦测模式】，捕捉极度活跃、流动性充沛的强势资金标的。"
            table_headers = "| 排名 | 股票代码 | 股票名称 | 所属板块 | 综合得分 | 最新价 | 日涨幅 | 成交额(亿) | 振幅(%) |\n| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
            table_rows = ""
            for i, row in top_stocks.iterrows():
                amount_y = row['amount'] / 1e8 if pd.notna(row['amount']) else 0
                range_p = ((row['high'] - row['low']) / row['close']) * 100 if row['close'] > 0 else 0
                table_rows += f"| {i+1} | {row['symbol']} | **{row['name']}** | {row['industry']} | {row['score']:.2f} | {row['close']:.2f} | <span class='{'red' if row['momentum']>0 else 'green'}'>{row['momentum']}%</span> | {amount_y:.1f} | {range_p:.1f}% |\n"
                
        report_md = f"""
# A股全市场多因子智能选股雷达 (Alpha Screener Pro 2.0)

**报告生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**扫描标的池**：A 股全市场 (剔除 ST、亏损股及流动性枯竭标的)  
**入选名单**：综合评分 Top {self.top_n}

<div class="intro-card">
    <h3>🧬 选股模型工作状态</h3>
    <p>当前运行在：<strong>{'基本面+量价综合模式' if self.report_type == 'fundamental' else '纯技术面动能监控模式'}</strong></p>
    <p><em>量化理念：{strategy_desc}</em></p>
</div>

---

## 🧭 第一部分：市场风格与行业暴露度诊断

![风格雷达图](images/style_radar.png)

{insights_md}

---

## 🏆 第二部分：Alpha 核心标的池 (Top {self.top_n})

{table_headers}{table_rows}

---

## 📊 第三部分：因子得分穿透解析 (Top 10)

为了防止某单一因子畸高导致的“假性高分”，以下图表拆解了前 10 名金股的得分结构，您可以直观看出它们是偏向于防守（价值/稳定）还是偏向于进攻（动量/弹性）：

![因子贡献图](images/factor_contribution.png)

### 💡 实战组合构建建议：
1. 量化选股的精髓在于**分散配置**。切忌单吊排名第一的个股，这会暴露在单点个股的黑天鹅风险下。
2. 建议将资金分为 5 份或 10 份，在剔除过度拥挤行业的标的后，等权买入。
3. 请在两周后重新生成此报告，对于跌出前 50 名的持仓股予以果断剔除。

---
*版权声明：本【智能选股引擎】由 Gemini CLI 量化终端驱动。因子模型存在失效周期，股市有风险，投资需谨慎。*
"""

        md_path = os.path.join(self.folder, f"多因子选股雷达_{self.report_date}.md")
        pdf_path = os.path.join(self.folder, f"多因子选股雷达_{self.report_date}.pdf")
        with open(md_path, "w", encoding="utf-8") as f: f.write(report_md)
        
        css = """
        @page { margin: 1.5cm 2cm; }
        body { font-family: "WenQuanYi Zen Hei", "Noto Sans CJK JP", sans-serif; line-height: 1.7; color: #1e293b; font-size: 10.5pt; }
        h1 { text-align: center; color: #b91c1c; border-bottom: 4px double #b91c1c; padding-bottom: 10px; font-size: 22pt; margin-bottom: 25px; }
        h2 { background: #334155; color: #ffffff; padding: 8px 15px; border-radius: 6px; margin-top: 35px; font-size: 15pt; border-left: 6px solid #ef4444; }
        h3 { color: #d97706; margin-top: 20px; margin-bottom: 10px; font-size: 13pt; }
        img { max-width: 90%; height: auto; display: block; margin: 20px auto; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 10pt; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        th, td { border: 1px solid #cbd5e1; padding: 10px; text-align: center; }
        th { background-color: #f8fafc; color: #0f172a; font-weight: bold; }
        tr:nth-child(even) { background-color: #f1f5f9; }
        .intro-card { background: #fdfefe; border: 1px dashed #3b82f6; border-left: 5px solid #3b82f6; border-radius: 8px; padding: 15px; margin-bottom: 25px; }
        .intro-card h3 { margin-top: 0; color: #2563eb; border-bottom: none; }
        ul { margin-top: 10px; margin-bottom: 10px; padding-left: 25px; }
        li { margin-bottom: 8px; }
        .red { color: #dc2626; font-weight: bold; }
        .green { color: #16a085; font-weight: bold; }
        """
        html_content = f"<html><head><style>{css}</style></head><body>{markdown.markdown(report_md, extensions=['tables'])}</body></html>"
        HTML(string=html_content, base_url=self.folder).write_pdf(pdf_path)
        print(f"\n✅ 升级版智能选股雷达报告生成完毕！\nPDF位置: {pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成A股多因子选股报告 (Pro 2.0)")
    parser.add_argument("--top", type=int, default=15, help="输出排名前N的股票")
    args = parser.parse_args()
    
    screener = FactorScreenerPro(top_n=args.top)
    screener.generate_report()
