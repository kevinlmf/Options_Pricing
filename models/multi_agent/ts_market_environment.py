"""
Time Series Driven Multi-Agent Market Environment

Market environment where agents interact to form option prices through:
1. Order submission
2. Order matching
3. Price discovery
4. Trade execution

The emergent market prices are then compared with traditional pricing models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.multi_agent.timeseries_driven_agents import (
    TimeSeriesDrivenAgent, TSMarketMaker, TSInformedTrader, TSArbitrageur,
    Order, Trade
)
from models.black_scholes import BlackScholesModel, BSParameters


@dataclass
class OrderBook:
    """Order book for a single option"""
    bids: List[Tuple[float, float, str]] = field(default_factory=list)  # [(price, qty, agent_id)]
    asks: List[Tuple[float, float, str]] = field(default_factory=list)  # [(price, qty, agent_id)]

    def add_order(self, order: Order) -> None:
        """Add order to the book."""
        entry = (order.limit_price, order.quantity, order.agent_id)

        if order.side == 'buy':
            self.bids.append(entry)
            # Sort bids: highest price first
            self.bids.sort(key=lambda x: -x[0])
        else:  # sell
            self.asks.append(entry)
            # Sort asks: lowest price first
            self.asks.sort(key=lambda x: x[0])

    def get_best_bid(self) -> Optional[Tuple[float, float, str]]:
        """Get best bid."""
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> Optional[Tuple[float, float, str]]:
        """Get best ask."""
        return self.asks[0] if self.asks else None

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2
        elif best_bid:
            return best_bid[0]
        elif best_ask:
            return best_ask[0]
        return None

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None

    def clear(self) -> None:
        """Clear all orders."""
        self.bids.clear()
        self.asks.clear()


class TSMarketEnvironment:
    """
    Market environment for time series driven agents.

    Manages:
    - Spot price evolution
    - Option contracts
    - Order books
    - Trade matching and execution
    - Price discovery
    """

    def __init__(self,
                 initial_spot: float = 100.0,
                 spot_drift: float = 0.0001,
                 spot_vol: float = 0.02,
                 risk_free_rate: float = 0.05):
        """
        Initialize market environment.

        Parameters:
        -----------
        initial_spot : float
            Initial underlying price
        spot_drift : float
            Drift for spot price (daily)
        spot_vol : float
            Volatility for spot price (daily)
        risk_free_rate : float
            Risk-free rate
        """
        self.initial_spot = initial_spot
        self.spot_price = initial_spot
        self.spot_drift = spot_drift
        self.spot_vol = spot_vol
        self.risk_free_rate = risk_free_rate

        # Spot price history
        self.spot_history: List[float] = [initial_spot]

        # Agents
        self.agents: Dict[str, TimeSeriesDrivenAgent] = {}

        # Options
        self.available_options: List[Tuple[float, float]] = []  # [(strike, expiry)]

        # Order books
        self.order_books: Dict[Tuple[float, float], OrderBook] = {}

        # Trade history
        self.trades: List[Trade] = []

        # Market prices (emerging from agent interactions)
        self.market_prices: Dict[Tuple[float, float], List[float]] = defaultdict(list)

        # Traditional model prices (for comparison)
        self.bs_prices: Dict[Tuple[float, float], List[float]] = defaultdict(list)

        # Time
        self.current_time = 0

    def add_agent(self, agent: TimeSeriesDrivenAgent) -> None:
        """Add an agent to the market."""
        self.agents[agent.agent_id] = agent
        print(f"Added agent: {agent.agent_id} (Role: {agent.role.value})")

    def add_option(self, strike: float, expiry: float) -> None:
        """Add an option contract to the market."""
        option_key = (strike, expiry)
        if option_key not in self.available_options:
            self.available_options.append(option_key)
            self.order_books[option_key] = OrderBook()
            print(f"Added option: Strike={strike}, Expiry={expiry}")

    def simulate_spot_price_movement(self) -> None:
        """Simulate one step of spot price evolution."""
        # Geometric Brownian Motion
        dt = 1.0  # daily
        dW = np.random.normal(0, 1)

        self.spot_price = self.spot_price * np.exp(
            (self.spot_drift - 0.5 * self.spot_vol**2) * dt +
            self.spot_vol * np.sqrt(dt) * dW
        )

        self.spot_history.append(self.spot_price)

        # Update all agents with new spot price
        for agent in self.agents.values():
            agent.observe_market(self.spot_price)

    def collect_orders(self) -> List[Order]:
        """Collect orders from all agents."""
        all_orders = []

        market_state = {
            'available_options': self.available_options,
            'order_book': {
                key: {
                    'bids': book.bids[:3],  # Top 3 levels
                    'asks': book.asks[:3]
                }
                for key, book in self.order_books.items()
            },
            'spot_price': self.spot_price,
            'risk_free_rate': self.risk_free_rate,
            'timestamp': self.current_time
        }

        for agent in self.agents.values():
            try:
                orders = agent.make_decision(market_state)
                all_orders.extend(orders)
            except Exception as e:
                print(f"Agent {agent.agent_id} error: {e}")

        return all_orders

    def match_orders(self) -> List[Trade]:
        """Match orders and generate trades."""
        trades = []

        for option_key in self.available_options:
            book = self.order_books[option_key]

            while True:
                best_bid = book.get_best_bid()
                best_ask = book.get_best_ask()

                # Check if orders can match
                if not best_bid or not best_ask:
                    break

                bid_price, bid_qty, buyer_id = best_bid
                ask_price, ask_qty, seller_id = best_ask

                if bid_price < ask_price:
                    break  # No match possible

                # Match at midpoint
                trade_price = (bid_price + ask_price) / 2
                trade_qty = min(bid_qty, ask_qty)

                # Create trade
                trade = Trade(
                    buyer_id=buyer_id,
                    seller_id=seller_id,
                    option_key=option_key,
                    quantity=trade_qty,
                    price=trade_price,
                    timestamp=self.current_time
                )
                trades.append(trade)

                # Update positions
                buyer = self.agents[buyer_id]
                seller = self.agents[seller_id]

                # Update buyer
                buyer.positions[option_key] = buyer.positions.get(option_key, 0) + trade_qty
                buyer.cash -= trade_qty * trade_price

                # Update seller
                seller.positions[option_key] = seller.positions.get(option_key, 0) - trade_qty
                seller.cash += trade_qty * trade_price

                # Remove or reduce orders
                if trade_qty >= bid_qty:
                    book.bids.pop(0)
                else:
                    book.bids[0] = (bid_price, bid_qty - trade_qty, buyer_id)

                if trade_qty >= ask_qty:
                    book.asks.pop(0)
                else:
                    book.asks[0] = (ask_price, ask_qty - trade_qty, seller_id)

                # Record market price
                self.market_prices[option_key].append(trade_price)

        return trades

    def calculate_bs_prices(self) -> Dict[Tuple[float, float], float]:
        """Calculate Black-Scholes prices for comparison."""
        bs_prices = {}

        # Estimate volatility from recent spot price history
        if len(self.spot_history) > 20:
            returns = np.log(pd.Series(self.spot_history[-20:]) /
                            pd.Series(self.spot_history[-21:-1]))
            historical_vol = returns.std() * np.sqrt(252)
        else:
            historical_vol = self.spot_vol * np.sqrt(252)

        for strike, expiry in self.available_options:
            params = BSParameters(
                S0=self.spot_price,
                K=strike,
                T=max(expiry - self.current_time/252, 0.01),  # Time remaining
                r=self.risk_free_rate,
                sigma=historical_vol,
                q=0.0
            )

            model = BlackScholesModel(params)
            bs_price = model.call_price()
            bs_prices[(strike, expiry)] = bs_price

            # Record for history
            self.bs_prices[(strike, expiry)].append(bs_price)

        return bs_prices

    def run_trading_session(self) -> Dict[str, Any]:
        """
        Run one trading session.

        Returns:
        --------
        Dict: Session statistics
        """
        # Clear order books
        for book in self.order_books.values():
            book.clear()

        # Collect orders from agents
        orders = self.collect_orders()

        # Add orders to order books
        for order in orders:
            if order.option_key in self.order_books:
                self.order_books[order.option_key].add_order(order)

        # Match orders
        trades = self.match_orders()
        self.trades.extend(trades)

        # Calculate BS prices for comparison
        bs_prices = self.calculate_bs_prices()

        # Get current market prices
        market_prices = {}
        for option_key, book in self.order_books.items():
            mid = book.get_mid_price()
            if mid:
                market_prices[option_key] = mid

        # Calculate agent P&Ls
        agent_pnls = {}
        for agent_id, agent in self.agents.items():
            pnl = agent.calculate_pnl(market_prices)
            agent_pnls[agent_id] = pnl

        return {
            'timestamp': self.current_time,
            'spot_price': self.spot_price,
            'num_orders': len(orders),
            'num_trades': len(trades),
            'market_prices': market_prices,
            'bs_prices': bs_prices,
            'agent_pnls': agent_pnls
        }

    def simulate(self, num_periods: int = 100, verbose: bool = True) -> pd.DataFrame:
        """
        Run full simulation.

        Parameters:
        -----------
        num_periods : int
            Number of trading periods
        verbose : bool
            Print progress

        Returns:
        --------
        pd.DataFrame: Simulation results
        """
        results = []

        print(f"\n{'='*80}")
        print(f"Starting Multi-Agent Time Series Driven Option Pricing Simulation")
        print(f"{'='*80}")
        print(f"Agents: {len(self.agents)}")
        print(f"Options: {len(self.available_options)}")
        print(f"Periods: {num_periods}")
        print(f"{'='*80}\n")

        for t in range(num_periods):
            self.current_time = t

            # Simulate spot price movement
            self.simulate_spot_price_movement()

            # Run trading session
            session_stats = self.run_trading_session()
            results.append(session_stats)

            if verbose and (t+1) % 20 == 0:
                print(f"Period {t+1}/{num_periods} - Spot: ${self.spot_price:.2f} - "
                      f"Trades: {session_stats['num_trades']}")

        print(f"\n{'='*80}")
        print(f"Simulation Complete")
        print(f"{'='*80}\n")

        return self._process_results(results)

    def _process_results(self, results: List[Dict]) -> pd.DataFrame:
        """Process simulation results into DataFrame."""
        processed = []

        for result in results:
            for option_key in self.available_options:
                strike, expiry = option_key

                market_price = result['market_prices'].get(option_key, np.nan)
                bs_price = result['bs_prices'].get(option_key, np.nan)

                processed.append({
                    'timestamp': result['timestamp'],
                    'spot_price': result['spot_price'],
                    'strike': strike,
                    'expiry': expiry,
                    'market_price': market_price,
                    'bs_price': bs_price,
                    'price_diff': market_price - bs_price if not np.isnan(market_price) else np.nan,
                    'price_diff_pct': ((market_price - bs_price) / bs_price * 100)
                                      if not np.isnan(market_price) and bs_price > 0 else np.nan,
                    'num_trades': result['num_trades']
                })

        return pd.DataFrame(processed)

    def get_summary_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        stats = {}

        # Overall statistics
        stats['total_periods'] = len(results_df['timestamp'].unique())
        stats['total_trades'] = results_df['num_trades'].sum()
        stats['avg_trades_per_period'] = results_df.groupby('timestamp')['num_trades'].first().mean()

        # Price comparison statistics
        valid_diffs = results_df['price_diff'].dropna()
        if len(valid_diffs) > 0:
            stats['mean_price_diff'] = valid_diffs.mean()
            stats['std_price_diff'] = valid_diffs.std()
            stats['mae'] = valid_diffs.abs().mean()
            stats['rmse'] = np.sqrt((valid_diffs ** 2).mean())

        valid_diffs_pct = results_df['price_diff_pct'].dropna()
        if len(valid_diffs_pct) > 0:
            stats['mean_price_diff_pct'] = valid_diffs_pct.mean()
            stats['std_price_diff_pct'] = valid_diffs_pct.std()

        # Agent statistics
        stats['agent_performance'] = {}
        for agent_id, agent in self.agents.items():
            final_pnl = agent.pnl_history[-1] if agent.pnl_history else 0
            stats['agent_performance'][agent_id] = {
                'role': agent.role.value,
                'final_pnl': final_pnl,
                'num_positions': len([p for p in agent.positions.values() if abs(p) > 0.01]),
                'cash': agent.cash
            }

        return stats
