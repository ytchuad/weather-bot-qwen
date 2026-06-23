"""
策略執行診斷腳本
用途：手動觸發策略執行，觀察 Edge 計算與風控閘門是否攔截交易。
執行方式：python scripts/diagnose_strategy.py
"""
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# 將專案根目錄加入路徑
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 設定詳細日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_diagnosis(strategy_id="enhanced_v2_paper"):
    logger.info(f"========== 開始診斷策略: {strategy_id} ==========")
    
    try:
        # 1. 載入策略帳戶與設定
        from execution.strategy_account import StrategyAccountStore
        from execution.market_templates import resolve_slug
        
        store = StrategyAccountStore()
        account = store.load(strategy_id)
        
        if not account:
            logger.error(f"❌ 找不到策略 ID: {strategy_id}")
            return
            
        logger.info(f"✅ 策略已載入: {account.label} (Status: {account.status}, Capital: ${account.capital})")
        
        # 2. 載入策略定義
        registry_path = Path("config/paper_strategies.json")
        if not registry_path.exists():
            logger.error("❌ 找不到 config/paper_strategies.json")
            return
            
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
            
        sdef = registry.get("strategies", {}).get(strategy_id)
        if sdef is None:
            logger.error(f"❌ 策略 '{strategy_id}' 不在 paper_strategies.json 中")
            return

        # 3. 準備執行參數
        model = account.model or "baseline"
        capital = account.capital
        params = account.params or {}
        template = account.market_template or "hk-tmax"
        event_slug = resolve_slug(template)
        is_min_temp = template == "hk-tmin"
        
        logger.info(f"目標 Slug: {event_slug} | Model: {model} | Capital: ${capital}")
        
        # 4. 抓取資料
        logger.info("---------- 抓取即時資料 ----------")
        from app.services.weather_service import fetch_hko_data, get_intraday_state, hkt_now, compute_rain_kwargs
        from app.services.market_service import fetch_today_event, fetch_event_markets
        from app.services.model_service import run_all_models
        
        target_date = hkt_now().date()
        target_date_str = target_date.strftime("%Y-%m-%d")
        _sd = target_date_str.replace("-", "")
        
        today_event = fetch_today_event(target_date_str)
        slug = today_event.get("slug") if today_event else event_slug
        markets = fetch_event_markets(slug, is_min_temp=is_min_temp) if slug else []
        
        if not markets:
            logger.error("❌ 找不到 Polymarket 市場數據，中止診斷。")
            return
            
        hko = fetch_hko_data(target_date_str)
        state = get_intraday_state(_sd)
        rain_kwargs = compute_rain_kwargs(_sd, hkt_now())
        forecast_key = "forecast_min" if is_min_temp else "forecast_max"
        forecast_aws = hko.get(forecast_key) if hko else None
        
        results = run_all_models(
            target_date=target_date,
            target_date_str=target_date_str,
            is_min_temp=is_min_temp,
            bias=params.get("bias", 0.0),
            std_mult=params.get("std_mult", 1.0),
            state=state,
            rain_kwargs=rain_kwargs,
            markets=markets,
            forecast_aws_val=forecast_aws,
            is_today=True
        )
        
        # 5. 準備策略引擎所需的 context
        target_probs = results.get(model, {}).get("probs", {})
        if not target_probs:
            logger.error(f"❌ 模型 {model} 沒有產出預測機率 (probs)。")
            return
            
        prices_dict = {m["bucket"]: m.get("yes_price", 0.5) for m in markets}
        token_ids_dict = {m["bucket"]: m.get("token_id", "") for m in markets}
        
        logger.info(f"Capital: ${capital}")
        logger.info(f"Target Probs: {target_probs}")
        logger.info(f"Prices Dict: {prices_dict}")
        
        # --- 關鍵診斷：手動測試 Kelly 計算 ---
        logger.info("---------- 測試凱利計算 ----------")
        from execution.kelly_betting import compute_multi_kelly_bets
        from execution.clob_slippage import apply_slippage_to_bets
        
        kelly_frac = params.get("kelly_fraction", 0.25)
        
        # 嘗試呼叫 compute_multi_kelly_bets
        bets = compute_multi_kelly_bets(
            target_probs, prices_dict, capital,
            max_per_bucket=0.15, total_max=0.50
        )
        logger.info(f"Raw Kelly Bets: {bets}")
        
        adjusted_bets = apply_slippage_to_bets(
            bets, token_ids_dict, prices_dict=prices_dict, mock_mode=True
        )
        logger.info(f"Adjusted Bets (after slippage): {adjusted_bets}")
        
        # --- 手動修正：如果滑點模擬失敗，直接使用原始 Kelly Bets ---
        if not adjusted_bets:
            logger.warning("⚠️ 滑點模擬返回空字典，手動構造 adjusted_bets 進行測試...")
            adjusted_bets = {}
            for bucket, bet in bets.items():
                # 假設滑點為 0，直接使用原始 amount 作為 quantity
                price = prices_dict.get(bucket, 0.5)
                if price > 0:
                    adjusted_bets[bucket] = {
                        "action": bet.get("action"),
                        "adjusted_quantity": bet.get("amount", 0) / price,  # 將金額換算為數量
                        "avg_fill_price": price,
                        "slippage_pct": 0.0,
                        "filled": True
                    }
            logger.info(f"手動構造的 Adjusted Bets: {adjusted_bets}")
        
        # 6. 準備 context 並觸發引擎
        context = dict(
            capital=capital,
            model_key=model,
            mock_slippage=True,
            bias=params.get("bias", 0.0),
            std_mult=params.get("std_mult", 1.0),
            kelly_fraction=kelly_frac,
            slug=slug,
            target_probs=target_probs,
            prices_dict=prices_dict,
            token_ids_dict=token_ids_dict,
            temp_now=state.get("temp_now") if state else None,
            max_so_far=state.get("max_so_far") if state else None,
            rain_regime=rain_kwargs.get("rain_regime", "no_rain"),
            model_std=1.5,
            recent_price_volatility=0.0,
            hours_to_settlement=24.0,
            nowcast_stale=False,
            data_missing=False,
            drawdown_pct=0.0,
            post_mean=results.get(model, {}).get("mean", 30.0)
        )
        
        logger.info("---------- 嘗試觸發執行引擎 ----------")
        try:
            from execution.strategy_runner import run_single_strategy_cycle
            
            logger.info("呼叫 run_single_strategy_cycle...")
            result = run_single_strategy_cycle(
                strategy_key=strategy_id,
                strategy_config=sdef,
                portfolio_id=strategy_id,
                event_slug=slug,
                **context
            )
            
            # 7. 分析執行結果
            status = result.get("status", "unknown")
            decisions = result.get("decisions", [])
            
            logger.info(f"✅ 引擎執行完畢。狀態: {status}")
            logger.info(f"總共產生 {len(decisions)} 個決策:")
            
            entry_count = 0
            exit_count = 0
            blocked_count = 0
            no_trade_count = 0
            hold_count = 0
            
            for d in decisions:
                action = d.get("action", "UNKNOWN")
                reason = d.get("reason", "")
                detail = d.get("detail", "")
                bucket = d.get("bucket", "")
                
                if action == "ENTRY":
                    entry_count += 1
                    logger.info(f"  🟢 [ENTRY] {bucket}: {detail}")
                elif action in ("EXIT", "REDUCE"):
                    exit_count += 1
                    logger.info(f"  🔴 [{action}] {bucket}: {detail}")
                elif action == "BLOCKED":
                    blocked_count += 1
                    logger.warning(f"  🟠 [BLOCKED] {bucket}: Reason={reason} | {detail}")
                elif action == "NO_TRADE":
                    no_trade_count += 1
                    logger.info(f"  ⚪ [NO_TRADE] {bucket}: {detail}")
                elif action == "HOLD":
                    hold_count += 1
                    
            logger.info(f"---------- 決策摘要 ----------")
            logger.info(f"Entry: {entry_count} | Exit: {exit_count} | Blocked: {blocked_count} | NoTrade: {no_trade_count} | Hold: {hold_count}")
                
        except Exception as e:
            logger.error(f"❌ 觸發引擎時發生錯誤: {e}", exc_info=True)
            
    except Exception as e:
        logger.error(f"❌ 診斷過程中發生未預期錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    TARGET_STRATEGY = "enhanced_v2_paper"
    run_diagnosis(TARGET_STRATEGY)