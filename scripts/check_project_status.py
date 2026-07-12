import os
import json
import subprocess
import pandas as pd
from pathlib import Path

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_git_status():
    print("="*50)
    print("🔍 [1/4] Git 版本與提交紀錄檢查")
    print("="*50)
    print("📌 當前分支與狀態:")
    print(run_command("git status -s"))
    print("\n📌 最近 3 次 Commit 紀錄:")
    print(run_command("git log -n 3 --oneline"))
    print()

def check_data_files():
    print("="*50)
    print("📂 [2/4] 核心數據檔案體檢 (data/)")
    print("="*50)
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ 找不到 data/ 資料夾！")
        return
        
    target_files = [
        "current_positions.json",
        "intraday_hko_10min.parquet",
        "lookup_upside.parquet",
        "lookup_downside.parquet",
        "hko_tmax_historical.parquet",
        "live_forecast_history.parquet",
        "forward_test_log.parquet"
    ]
    
    for fname in target_files:
        fpath = data_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            if fname.endswith(".json"):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"✅ {fname} ({size_kb:.1f} KB) | 內容摘要: {list(data.keys()) if isinstance(data, dict) else 'List'}")
                    if fname == "current_positions.json":
                        print(f"   👉 完整 JSON 內容:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
                except Exception as e:
                    print(f"⚠️ {fname} 讀取失敗: {e}")
            elif fname.endswith(".parquet"):
                try:
                    df = pd.read_parquet(fpath)
                    print(f"✅ {fname} ({size_kb:.1f} KB) | 行數: {len(df):,} | 欄位: {list(df.columns)}")
                except Exception as e:
                    print(f"❌ {fname} 損壞或讀取失敗: {e}")
        else:
            print(f"❌ 缺失: {fname}")
    print()

def check_code_fingerprints():
    print("="*50)
    print("🧬 [3/4] 核心程式碼指紋檢查 (驗證最新修復是否落地)")
    print("="*50)
    
    checks = {
        "execution/rebalancer.py": [
            ("NO PnL 修復", "current_market_price = 1.0 - price_yes"),
            ("粉塵過濾修復", "0 < target_qty < PM_MIN_QTY"),
            ("再平衡記帳修復", "delta = target_qty - current_qty")
        ],
        "execution/clob_slippage.py": [
            ("NO 基準價格修復", "base_price = yes_price if is_buy_yes else (1.0 - yes_price)"),
            ("VWAP 計算修復", "total_cost_actual += max_cost_here")
        ],
        "dashboard.py": [
            ("PnL 隔離修復", "calculate_pnl({slug: current_market_positions}"),
            ("頂部搜尋欄", "st.text_input(\"搜尋關鍵字\"")
        ]
    }
    
    for filepath, fingerprints in checks.items():
        fpath = Path(filepath)
        if not fpath.exists():
            print(f"❌ 找不到 {filepath}")
            continue
            
        content = fpath.read_text(encoding='utf-8')
        print(f"📄 {filepath}:")
        for name, keyword in fingerprints:
            status = "✅ 已落地" if keyword in content else "❌ 未發現 (可能為舊版程式碼)"
            print(f"   - {name}: {status}")
    print()

def check_project_tree():
    print("="*50)
    print("🌳 [4/4] 專案目錄結構 (排除 .git 與 __pycache__)")
    print("="*50)
    ignore_dirs = {'.git', '__pycache__', '.venv', 'venv', '.streamlit'}
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        level = root.replace(".", "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            if f.endswith(('.py', '.json', '.parquet', '.yml', '.yaml', '.md', '.txt')):
                print(f"{subindent}📄 {f}")
    print()

if __name__ == "__main__":
    print("🚀 開始執行 Weather Bot 專案狀態深度診斷...\n")
    check_git_status()
    check_data_files()
    check_code_fingerprints()
    check_project_tree()
    print("✅ 診斷完成！請將以上所有輸出結果複製並發送給 AI。")