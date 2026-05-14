import os
import sys
import pandas as pd
import numpy as np
from smartmoneyconcepts import smc
import yfinance as yf
from datetime import datetime, timedelta
import markdown
from weasyprint import HTML

# Ensure utf-8 for console and files
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def analyze_pltr():
    # 1. Setup
    symbol = "PLTR"
    name = "Palantir"
    end_date = datetime(2026, 5, 12)
    start_date = end_date - timedelta(days=180) # More data for stability
    
    print(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    # 2. Fetch Data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    if df.empty:
        print("Error: No data found.")
        return

    # Standardize OHLCV
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    
    original_dates = df.index
    # Temporarily use RangeIndex for SMC computation as the library expects it
    ohlc = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    ohlc.columns = ["open", "high", "low", "close", "volume"]
    ohlc = ohlc.reset_index(drop=True)
    
    # 3. SMC Analysis
    print("Performing SMC analysis...")
    # Swing Points
    swing_hl = smc.swing_highs_lows(ohlc, swing_length=5)
    
    # BOS / ChoCH
    bos_choch = smc.bos_choch(ohlc, swing_highs_lows=swing_hl)
    
    # FVG
    fvg = smc.fvg(ohlc)
    
    # Order Blocks (Supply/Demand Zones)
    ob = smc.ob(ohlc, swing_highs_lows=swing_hl)
    
    # Re-align with original dates
    swing_hl.index = original_dates
    bos_choch.index = original_dates
    fvg.index = original_dates
    ob.index = original_dates
    
    # 4. Identify Latest Zones
    latest_demand = ob[ob['OB'] == 1].tail(3)
    latest_supply = ob[ob['OB'] == -1].tail(3)
    
    # Filter only the last quarter for display
    display_start = end_date - timedelta(days=90)
    
    # 5. Generate Markdown Report
    report_md = f"""
# Smart Money Concepts (SMC) 投研报告 - {name} ({symbol})

## 1. 基础信息
- **分析标的**: {name} ({symbol}.US)
- **分析周期**: {display_start.date()} 至 {end_date.date()} (最近一个季度)
- **数据源**: Yahoo Finance
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 2. 市场结构分析 (Market Structure)

机构资金流动的核心在于识别趋势的延续 (BOS) 与 反转 (ChoCH)。

### 结构突破 (BOS) 与 性质转变 (ChoCH)
"""
    # Summary of BOS/ChoCH
    recent_struc = bos_choch.loc[display_start:]
    recent_struc = recent_struc[(recent_struc['BOS'].notna() & (recent_struc['BOS'] != 0)) | 
                                (recent_struc['CHOCH'].notna() & (recent_struc['CHOCH'] != 0))].tail(5)
    
    if not recent_struc.empty:
        report_md += "| 日期 | 类型 | 方向 | 说明 |\n| :--- | :--- | :--- | :--- |\n"
        for idx, row in recent_struc.iterrows():
            if pd.notna(row['BOS']) and row['BOS'] != 0:
                stype = "BOS"
                val = row['BOS']
            else:
                stype = "ChoCH"
                val = row['CHOCH']
            direction = "🟢 看多 (Bullish)" if val == 1 else "🔴 看空 (Bearish)"
            report_md += f"| {idx.date()} | {stype} | {direction} | 趋势确认/反转信号 |\n"
    else:
        report_md += "本季度未检测到显著的结构突破信号。\n"

    report_md += """
## 3. 供需区间分析 (Supply & Demand Zones)

订单块 (Order Blocks) 是主力资金留下“脚印”的地方，通常作为未来的支撑或阻力。

### 🟢 需求区间 (Demand Zones / Bullish OB)
"""
    # Show demand zones formed recently
    demand_view = ob[(ob['OB'] == 1) & (ob.index >= display_start)].tail(3)
    if not demand_view.empty:
        report_md += "| 形成日期 | 区间上限 | 区间下限 | 说明 |\n| :--- | :--- | :--- | :--- |\n"
        for idx, row in demand_view.iterrows():
            report_md += f"| {idx.date()} | {row['Top']:.2f} | {row['Bottom']:.2f} | 潜在支撑区域 |\n"
    else:
        report_md += "本季度未检测到显著的需求区间。\n"

    report_md += """
### 🔴 供应区间 (Supply Zones / Bearish OB)
"""
    # Show supply zones formed recently
    supply_view = ob[(ob['OB'] == -1) & (ob.index >= display_start)].tail(3)
    if not supply_view.empty:
        report_md += "| 形成日期 | 区间上限 | 区间下限 | 说明 |\n| :--- | :--- | :--- | :--- |\n"
        for idx, row in supply_view.iterrows():
            report_md += f"| {idx.date()} | {row['Top']:.2f} | {row['Bottom']:.2f} | 潜在阻力区域 |\n"
    else:
        report_md += "本季度未检测到显著的供应区间。\n"

    report_md += """
## 4. 失衡区分析 (Fair Value Gaps)
价格倾向于在未来填补这些流动性失衡区。

### 最近的公允价值缺口 (FVG)
"""
    latest_fvg = fvg[(fvg['FVG'] != 0) & (fvg.index >= display_start)].tail(5)
    if not latest_fvg.empty:
        report_md += "| 日期 | 方向 | 顶端 | 底端 | 状态 |\n| :--- | :--- | :--- | :--- | :--- |\n"
        for idx, row in latest_fvg.iterrows():
            direction = "🟢 看多 (Bullish)" if row['FVG'] == 1 else "🔴 看空 (Bearish)"
            report_md += f"| {idx.date()} | {direction} | {row['Top']:.2f} | {row['Bottom']:.2f} | 待回补 |\n"
    else:
        report_md += "本季度目前盘面较为均衡，未发现明显的 FVG 缺口。\n"

    report_md += f"""
## 5. 综合结论
基于 Smart Money Concepts (SMC) 逻辑，**{symbol}** 的季度研判如下：
1. **当前趋势**: 观察最近的 BOS/ChoCH 信号方向。
2. **关键位置**: 
    - 关注下方的需求区间 **{latest_demand['Bottom'].iloc[-1] if not latest_demand.empty else 'N/A'} - {latest_demand['Top'].iloc[-1] if not latest_demand.empty else 'N/A'}**。
    - 关注上方的供应区间 **{latest_supply['Bottom'].iloc[-1] if not latest_supply.empty else 'N/A'} - {latest_supply['Top'].iloc[-1] if not latest_supply.empty else 'N/A'}**。
3. **交易策略**: 机构通常在价格回踩需求区间或回补 FVG 后再次入场。建议在上述区间寻找价格行为确认。

---
*风险提示：本报告由 Gemini CLI 首席分析师调用 SMC 算法自动生成，仅供研究参考，不构成任何买卖建议。*
"""

    # 6. Save Markdown and Convert to PDF
    report_date = "20260512"
    folder = "reports/PalantirPLTR"
    filename_base = f"USPLTR_SMC分析_{report_date}"
    
    md_path = os.path.join(folder, f"{filename_base}.md")
    pdf_path = os.path.join(folder, f"{filename_base}.pdf")
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    # Simple CSS for PDF
    css = """
    @page { margin: 2cm; }
    body { font-family: sans-serif; line-height: 1.6; color: #333; }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; text-align: center; }
    h2 { color: #2980b9; margin-top: 40px; border-left: 6px solid #2980b9; padding-left: 15px; background: #f8f9fa; }
    h3 { color: #e67e22; border-bottom: 1px dashed #e67e22; padding-bottom: 5px; }
    table { width: 100%; border-collapse: collapse; margin: 25px 0; }
    th, td { border: 1px solid #dfe6e9; padding: 12px; text-align: left; font-size: 11pt; }
    th { background-color: #34495e; color: white; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    """
    
    html_content = f"<html><head><style>{css}</style></head><body>{markdown.markdown(report_md, extensions=['tables'])}</body></html>"
    HTML(string=html_content).write_pdf(pdf_path)
    
    print(f"PDF Report successfully generated: {pdf_path}")

if __name__ == "__main__":
    analyze_pltr()
