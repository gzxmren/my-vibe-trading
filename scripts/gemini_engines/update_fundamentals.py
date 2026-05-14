#!/usr/bin/env python3
import os
import sys
import sqlite3
import pandas as pd
import akshare as ak
import tushare as ts
import time
import random
import argparse
from datetime import datetime
from dotenv import load_dotenv

# --- 环境配置 ---
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 加载 .env 获取 Token
load_dotenv('agent/.env')

class FundamentalsUpdater:
    def __init__(self, use_deep=False):
        self.use_deep = use_deep
        self.db_path = 'data/fundamentals.db'
        os.makedirs('data', exist_ok=True)
        self.tushare_token = os.environ.get("TUSHARE_TOKEN", "")
        if self.tushare_token and self.tushare_token != "your-tushare-token":
            ts.set_token(self.tushare_token)
            self.pro = ts.pro_api()
        else:
            self.pro = None

    def fetch_industry_mapping(self):
        print("\n🗂️ 正在构建行业板块映射表...")
        df_industry = pd.DataFrame()
        
        # 1. 尝试 Tushare (最稳健，如果有Token)
        if self.pro:
            try:
                print("   -> 尝试通过 Tushare 获取行业分类...")
                df_industry = self.pro.stock_basic(exchange='', list_status='L', fields='symbol,industry')
                # Tushare 的 symbol 是 000001.SZ，转换为 000001
                df_industry['symbol'] = df_industry['symbol'].str.split('.').str[0]
                print(f"   ✅ Tushare 获取成功，共 {len(df_industry)} 条行业记录。")
                return df_industry
            except Exception as e:
                print(f"   ⚠️ Tushare 行业数据获取失败: {e}")

        # 2. 如果没有 Tushare，尝试使用 akshare (新浪行业)
        try:
            print("   -> 尝试通过 Sina 获取行业分类...")
            # 注意：全市场遍历行业较慢，这里使用一个折中方法：在选股引擎中通过代码前缀做 fallback，
            # 暂时返回空 DataFrame，避免长时间爬取被封 IP。
            print("   ⚠️ 为避免触发防爬虫，暂时跳过大规模在线板块遍历。选股引擎将自动使用代码前缀推断板块。")
        except Exception as e:
            pass
            
        return pd.DataFrame(columns=['symbol', 'industry'])

    def fetch_market_snapshot(self):
        print("\n📡 正在拉取全市场量价与基础财务快照...")
        df = pd.DataFrame()
        
        # 1. 尝试东方财富接口
        try:
            print("   -> 尝试连接东方财富接口 (ak.stock_zh_a_spot_em)...")
            df = ak.stock_zh_a_spot_em()
            print(f"   ✅ 东财接口获取成功，共 {len(df)} 只股票。")
            
            cols_map = {
                "代码": "symbol", "名称": "name", "最新价": "close", "涨跌幅": "momentum",
                "换手率": "turnover", "市盈率-动态": "pe", "市净率": "pb", "总市值": "mkt_cap",
                "流通市值": "float_cap"
            }
            available_cols = [c for c in cols_map.keys() if c in df.columns]
            df = df[available_cols].rename(columns={k: v for k, v in cols_map.items() if k in available_cols})
            return df
        except Exception as e:
            print(f"   ❌ 东财接口拉取失败: {e}")
            print("   💡 原因：东方财富单次拉取全市场往往不受数量限制，而是受【请求频率】限制。您当前的 IP 可能暂时被封禁。")
            
        # 2. 降级：尝试新浪接口
        try:
            print("   -> 自动降级：尝试连接新浪备用接口 (ak.stock_zh_a_spot)...")
            df = ak.stock_zh_a_spot()
            print(f"   ✅ 新浪接口获取成功，共 {len(df)} 只股票 (仅量价数据)。")
            
            cols_map = {
                "代码": "symbol", "名称": "name", "最新价": "close", "涨跌幅": "momentum",
                "成交量": "volume", "成交额": "amount"
            }
            available_cols = [c for c in cols_map.keys() if c in df.columns]
            df = df[available_cols].rename(columns={k: v for k, v in cols_map.items() if k in available_cols})
            
            # 清理 symbol 中的 sh/sz 前缀
            df['symbol'] = df['symbol'].str.replace(r'^[A-Za-z]+', '', regex=True)
            return df
        except Exception as e:
            print(f"   ❌ 新浪接口也失败了: {e}")

        return pd.DataFrame()

    def fetch_deep_fundamentals(self, base_df):
        if not self.use_deep or not self.pro or base_df.empty:
            return base_df
            
        print("\n🤿 开启深度模式 (Deep Mode)：正在从 Tushare 获取深度财务因子 (ROE 等)...")
        try:
            # 获取最近一期的财务指标
            # 注意：Tushare pro_api 限制频率，一次获取全市场通常会报错，可以通过 ts.pro_bar 或者分批。
            # 为了简单健壮，我们获取 daily_basic (包含 PE, PB, 换手率等)
            trade_date = datetime.now().strftime("%Y%m%d")
            print("   -> 尝试拉取每日指标 (daily_basic)...")
            df_basic = self.pro.daily_basic(ts_code='', trade_date='', fields='ts_code,turnover_rate,pe_ttm,pb,total_mv')
            if not df_basic.empty:
                df_basic['symbol'] = df_basic['ts_code'].str.split('.').str[0]
                df_basic = df_basic.rename(columns={'pe_ttm': 'pe_tushare', 'pb': 'pb_tushare', 'total_mv': 'mkt_cap_tushare'})
                
                # 合并数据
                base_df = pd.merge(base_df, df_basic[['symbol', 'pe_tushare', 'pb_tushare', 'mkt_cap_tushare']], on='symbol', how='left')
                # 如果东方财富缺失数据，用 tushare 的填补
                if 'pe' not in base_df.columns: base_df['pe'] = base_df['pe_tushare']
                if 'pb' not in base_df.columns: base_df['pb'] = base_df['pb_tushare']
                if 'mkt_cap' not in base_df.columns: base_df['mkt_cap'] = base_df['mkt_cap_tushare'] * 10000 # 万转绝对值
                
                print("   ✅ 成功合并 Tushare 深度指标。")
        except Exception as e:
            print(f"   ⚠️ Tushare 深度数据拉取失败 (可能是积分权限不足): {e}")
            
        return base_df

    def update_database(self):
        print("==================================================")
        print("  🏗️  A 股本地基本面数据库构建中心 (Data Forge 2.0)")
        print("==================================================\n")
        
        # 1. 抓取量价快照
        df_market = self.fetch_market_snapshot()
        if df_market.empty:
            print("\n❌ 数据库更新中止：无法获取市场快照。")
            return

        # 2. 抓取行业映射
        df_industry = self.fetch_industry_mapping()
        if not df_industry.empty:
            df_market = pd.merge(df_market, df_industry, on='symbol', how='left')
            
        # 3. 深度财务模式
        df_market = self.fetch_deep_fundamentals(df_market)
        
        # 4. 数据清洗与入库
        print("\n🧹 正在清洗与标准化数据...")
        num_cols = ["close", "momentum", "turnover", "pe", "pb", "mkt_cap", "float_cap"]
        for col in num_cols:
            if col in df_market.columns:
                df_market[col] = pd.to_numeric(df_market[col], errors="coerce")
                
        df_market['update_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("💾 正在将数据序列化至 SQLite 本地数据库...")
        conn = sqlite3.connect(self.db_path)
        df_market.to_sql('a_share_fundamentals', conn, if_exists='replace', index=False)
        
        cursor = conn.cursor()
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON a_share_fundamentals (symbol)')
        conn.commit()
        conn.close()
        
        print("\n🎉 数据库构建与更新完成！")
        print(f"   📂 存储路径: {os.path.abspath(self.db_path)}")
        print("   💡 现在，选股引擎可以极速、稳定地从本地读取横截面数据。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="更新 A 股本地基本面数据库")
    parser.add_argument("--deep", action="store_true", help="开启深度模式，使用 Tushare 补充深度财务因子 (需配置 Token)")
    args = parser.parse_args()
    
    updater = FundamentalsUpdater(use_deep=args.deep)
    updater.update_database()
