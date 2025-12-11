"""
Layer 7: Portfolio Construction Layer
====================================

Trade execution and portfolio management system.

This layer handles the actual execution of trading decisions from Layer 5,
including order management, transaction cost optimization, and portfolio
rebalancing.

Key Functions:
- Order generation and execution
- Transaction cost optimization
- Portfolio rebalancing
- Execution algorithms (VWAP, TWAP, etc.)
- Settlement and cash management
- Performance tracking
"""

from .trade_executor import TradeExecutor, ExecutionOrder, ExecutionResult

__all__ = [
    'TradeExecutor',
    'ExecutionOrder',
    'ExecutionResult'
]

