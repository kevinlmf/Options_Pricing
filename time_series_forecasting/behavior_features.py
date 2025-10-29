"""
Behavior Analysis Feature Extraction

Extract market microstructure and agent behavior features to enhance
time series forecasting.

Features include:
1. Order flow imbalance
2. Bid-ask spread dynamics
3. Market depth
4. Trading volume patterns
5. Agent position changes
6. Market sentiment indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BehaviorFeatures:
    """Container for behavior analysis features"""
    # Order flow features
    order_flow_imbalance: float  # (buy_volume - sell_volume) / total_volume
    order_flow_momentum: float   # Change in order flow

    # Spread features
    bid_ask_spread: float
    spread_change: float
    relative_spread: float  # spread / mid_price

    # Depth features
    bid_depth: float  # Total quantity on bid side
    ask_depth: float  # Total quantity on ask side
    depth_imbalance: float  # (bid_depth - ask_depth) / total_depth

    # Volume features
    trading_volume: float
    volume_change: float
    volume_volatility: float

    # Agent behavior features
    market_maker_inventory: float  # Total MM inventory
    informed_trader_activity: float  # Informed trader order size
    arbitrage_pressure: float  # Arbitrageur activity level

    # Sentiment features
    bullish_sentiment: float  # Ratio of buy orders
    sentiment_change: float

    # Price impact features
    price_impact: float  # Recent price change per unit volume

    # Timestamp
    timestamp: int


class BehaviorFeatureExtractor:
    """
    Extract behavior features from market data.

    These features capture market microstructure and agent behavior
    that may predict future price movements beyond pure time series.
    """

    def __init__(self, lookback_window: int = 10):
        """
        Initialize feature extractor.

        Parameters:
        -----------
        lookback_window : int
            Number of periods to look back for calculating features
        """
        self.lookback_window = lookback_window

        # Historical data for feature calculation
        self.order_flow_history: List[float] = []
        self.spread_history: List[float] = []
        self.volume_history: List[float] = []
        self.price_history: List[float] = []
        self.depth_history: List[Tuple[float, float]] = []  # (bid_depth, ask_depth)

        # Agent behavior history
        self.mm_inventory_history: List[float] = []
        self.informed_activity_history: List[float] = []
        self.arb_pressure_history: List[float] = []

        # Sentiment history
        self.sentiment_history: List[float] = []

    def update_market_data(self,
                          order_book: Dict,
                          trades: List,
                          agents: Dict,
                          current_price: float,
                          timestamp: int) -> None:
        """
        Update historical data with current market state.

        Parameters:
        -----------
        order_book : Dict
            Current order book state
        trades : List
            Recent trades
        agents : Dict
            Agent states
        current_price : float
            Current market price
        timestamp : int
            Current timestamp
        """
        # Calculate order flow
        buy_volume = sum(t.quantity for t in trades if t.buyer_id != 'market')
        sell_volume = sum(t.quantity for t in trades if t.seller_id != 'market')
        total_volume = buy_volume + sell_volume

        if total_volume > 0:
            order_flow_imbalance = (buy_volume - sell_volume) / total_volume
        else:
            order_flow_imbalance = 0.0

        self.order_flow_history.append(order_flow_imbalance)

        # Calculate spread
        if order_book and 'bids' in order_book and 'asks' in order_book:
            if order_book['bids'] and order_book['asks']:
                best_bid = order_book['bids'][0][0]
                best_ask = order_book['asks'][0][0]
                spread = best_ask - best_bid

                # Calculate depth
                bid_depth = sum(qty for _, qty, _ in order_book['bids'])
                ask_depth = sum(qty for _, qty, _ in order_book['asks'])
            else:
                spread = 0.0
                bid_depth = 0.0
                ask_depth = 0.0
        else:
            spread = 0.0
            bid_depth = 0.0
            ask_depth = 0.0

        self.spread_history.append(spread)
        self.depth_history.append((bid_depth, ask_depth))

        # Volume
        self.volume_history.append(total_volume)

        # Price
        self.price_history.append(current_price)

        # Agent behavior
        mm_inventory = sum(
            sum(agent.positions.values())
            for agent in agents.values()
            if 'MM' in agent.agent_id
        )
        self.mm_inventory_history.append(mm_inventory)

        informed_activity = sum(
            len(agent.positions)
            for agent in agents.values()
            if 'Trader' in agent.agent_id
        )
        self.informed_activity_history.append(informed_activity)

        arb_pressure = sum(
            abs(sum(agent.positions.values()))
            for agent in agents.values()
            if 'Arb' in agent.agent_id
        )
        self.arb_pressure_history.append(arb_pressure)

        # Sentiment (buy orders / total orders)
        if total_volume > 0:
            sentiment = buy_volume / total_volume
        else:
            sentiment = 0.5  # Neutral
        self.sentiment_history.append(sentiment)

    def extract_features(self, timestamp: int) -> BehaviorFeatures:
        """
        Extract behavior features from historical data.

        Parameters:
        -----------
        timestamp : int
            Current timestamp

        Returns:
        --------
        BehaviorFeatures: Extracted features
        """
        # Ensure we have enough history
        if len(self.order_flow_history) < 2:
            return self._get_default_features(timestamp)

        # Order flow features
        order_flow_imbalance = self._safe_last(self.order_flow_history, 0.0)
        order_flow_momentum = self._calculate_change(self.order_flow_history)

        # Spread features
        bid_ask_spread = self._safe_last(self.spread_history, 0.0)
        spread_change = self._calculate_change(self.spread_history)
        current_price = self._safe_last(self.price_history, 100.0)
        relative_spread = bid_ask_spread / current_price if current_price > 0 else 0.0

        # Depth features
        if self.depth_history:
            bid_depth, ask_depth = self.depth_history[-1]
            total_depth = bid_depth + ask_depth
            depth_imbalance = ((bid_depth - ask_depth) / total_depth
                             if total_depth > 0 else 0.0)
        else:
            bid_depth = ask_depth = depth_imbalance = 0.0

        # Volume features
        trading_volume = self._safe_last(self.volume_history, 0.0)
        volume_change = self._calculate_change(self.volume_history)
        volume_volatility = self._calculate_volatility(self.volume_history)

        # Agent behavior features
        market_maker_inventory = self._safe_last(self.mm_inventory_history, 0.0)
        informed_trader_activity = self._safe_last(self.informed_activity_history, 0.0)
        arbitrage_pressure = self._safe_last(self.arb_pressure_history, 0.0)

        # Sentiment features
        bullish_sentiment = self._safe_last(self.sentiment_history, 0.5)
        sentiment_change = self._calculate_change(self.sentiment_history)

        # Price impact
        price_impact = self._calculate_price_impact()

        return BehaviorFeatures(
            order_flow_imbalance=order_flow_imbalance,
            order_flow_momentum=order_flow_momentum,
            bid_ask_spread=bid_ask_spread,
            spread_change=spread_change,
            relative_spread=relative_spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_imbalance=depth_imbalance,
            trading_volume=trading_volume,
            volume_change=volume_change,
            volume_volatility=volume_volatility,
            market_maker_inventory=market_maker_inventory,
            informed_trader_activity=informed_trader_activity,
            arbitrage_pressure=arbitrage_pressure,
            bullish_sentiment=bullish_sentiment,
            sentiment_change=sentiment_change,
            price_impact=price_impact,
            timestamp=timestamp
        )

    def get_feature_vector(self, features: BehaviorFeatures) -> np.ndarray:
        """
        Convert features to numpy array.

        Parameters:
        -----------
        features : BehaviorFeatures
            Behavior features

        Returns:
        --------
        np.ndarray: Feature vector
        """
        return np.array([
            features.order_flow_imbalance,
            features.order_flow_momentum,
            features.bid_ask_spread,
            features.spread_change,
            features.relative_spread,
            features.bid_depth,
            features.ask_depth,
            features.depth_imbalance,
            features.trading_volume,
            features.volume_change,
            features.volume_volatility,
            features.market_maker_inventory,
            features.informed_trader_activity,
            features.arbitrage_pressure,
            features.bullish_sentiment,
            features.sentiment_change,
            features.price_impact
        ])

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get names of all features."""
        return [
            'order_flow_imbalance',
            'order_flow_momentum',
            'bid_ask_spread',
            'spread_change',
            'relative_spread',
            'bid_depth',
            'ask_depth',
            'depth_imbalance',
            'trading_volume',
            'volume_change',
            'volume_volatility',
            'market_maker_inventory',
            'informed_trader_activity',
            'arbitrage_pressure',
            'bullish_sentiment',
            'sentiment_change',
            'price_impact'
        ]

    def _safe_last(self, history: List, default: float) -> float:
        """Safely get last element from history."""
        return history[-1] if history else default

    def _calculate_change(self, history: List) -> float:
        """Calculate recent change in a metric."""
        if len(history) < 2:
            return 0.0
        return history[-1] - history[-2]

    def _calculate_volatility(self, history: List) -> float:
        """Calculate volatility of a metric."""
        if len(history) < self.lookback_window:
            return 0.0
        recent = history[-self.lookback_window:]
        return np.std(recent) if len(recent) > 1 else 0.0

    def _calculate_price_impact(self) -> float:
        """Calculate price impact per unit volume."""
        if len(self.price_history) < 2 or len(self.volume_history) < 1:
            return 0.0

        price_change = abs(self.price_history[-1] - self.price_history[-2])
        recent_volume = self.volume_history[-1]

        if recent_volume > 0:
            return price_change / recent_volume
        return 0.0

    def _get_default_features(self, timestamp: int) -> BehaviorFeatures:
        """Get default features when not enough history."""
        return BehaviorFeatures(
            order_flow_imbalance=0.0,
            order_flow_momentum=0.0,
            bid_ask_spread=0.0,
            spread_change=0.0,
            relative_spread=0.0,
            bid_depth=0.0,
            ask_depth=0.0,
            depth_imbalance=0.0,
            trading_volume=0.0,
            volume_change=0.0,
            volume_volatility=0.0,
            market_maker_inventory=0.0,
            informed_trader_activity=0.0,
            arbitrage_pressure=0.0,
            bullish_sentiment=0.5,
            sentiment_change=0.0,
            price_impact=0.0,
            timestamp=timestamp
        )


def analyze_feature_importance(features_history: List[BehaviorFeatures],
                               price_changes: np.ndarray) -> pd.DataFrame:
    """
    Analyze which behavior features are most predictive of price changes.

    Parameters:
    -----------
    features_history : List[BehaviorFeatures]
        Historical behavior features
    price_changes : np.ndarray
        Corresponding price changes

    Returns:
    --------
    pd.DataFrame: Feature importance scores
    """
    from sklearn.ensemble import RandomForestRegressor

    # Convert features to matrix
    feature_matrix = np.array([
        BehaviorFeatureExtractor.get_feature_vector(
            BehaviorFeatureExtractor(), f
        )
        for f in features_history
    ])

    # Train random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(feature_matrix, price_changes)

    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': BehaviorFeatureExtractor.get_feature_names(),
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance_df
