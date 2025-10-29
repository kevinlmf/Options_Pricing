"""
Real-Time Options Trading Monitoring API

FastAPI-based monitoring system with:
- REST API for risk metrics and order checking
- WebSocket for real-time updates
- Integration with BitcoinRiskController
- Live Greeks and P&L tracking

Endpoints:
- GET /api/risk/status - Current risk status
- POST /api/risk/check-order - Pre-trade risk check
- POST /api/orders/submit - Submit order (with risk check)
- WS /ws/risk - Real-time risk updates
- WS /ws/pnl - Real-time P&L updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio
import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.bitcoin_risk_controller import (
    BitcoinRiskController,
    BitcoinRiskLimits,
    OrderProposal,
    RiskCheckResult,
    VolatilityRegime
)


# ============================================================================
# Pydantic Models for API
# ============================================================================

class OrderRequest(BaseModel):
    """Order request model"""
    symbol: str = Field(..., example="BTC")
    option_type: str = Field(..., example="call", pattern="^(call|put)$")
    strike: float = Field(..., gt=0, example=45000)
    expiry: float = Field(..., gt=0, example=0.25)
    quantity: int = Field(..., gt=0, example=10)
    direction: str = Field(..., example="buy", pattern="^(buy|sell)$")
    underlying_price: float = Field(..., gt=0, example=45000)
    volatility: float = Field(..., gt=0, lt=5, example=0.75)
    risk_free_rate: float = Field(default=0.05, ge=0, le=1)


class RiskStatusResponse(BaseModel):
    """Risk status response"""
    timestamp: str
    portfolio_value: float
    volatility_regime: str
    current_iv: float
    current_greeks: Dict[str, float]
    current_var: Optional[float]
    current_cvar: Optional[float]
    adjusted_limits: Dict[str, float]
    utilization: Dict[str, float]
    statistics: Dict[str, Any]


class OrderCheckResponse(BaseModel):
    """Order check response"""
    status: str
    approved: bool
    reasons: List[str]
    risk_metrics: Dict[str, float]
    proposed_metrics: Dict[str, float]
    limit_utilization: Dict[str, float]
    volatility_regime: str
    timestamp: str


class PortfolioUpdate(BaseModel):
    """Portfolio state update"""
    positions: List[Dict[str, Any]]
    greeks: Dict[str, float]
    returns: List[float]
    current_iv: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    controller_active: bool


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            'risk': [],
            'pnl': [],
            'orders': []
        }

    async def connect(self, websocket: WebSocket, channel: str):
        """Accept and register new WebSocket connection"""
        await websocket.accept()
        if channel in self.active_connections:
            self.active_connections[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str):
        """Remove WebSocket connection"""
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)

    async def broadcast(self, message: dict, channel: str):
        """Broadcast message to all connections in channel"""
        if channel not in self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn, channel)

    async def send_personal(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
        except Exception:
            pass


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Bitcoin Options Real-Time Monitor",
    description="Real-time risk monitoring and order management API for Bitcoin options",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
manager = ConnectionManager()
risk_controller: Optional[BitcoinRiskController] = None
update_interval = 1.0  # seconds


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize risk controller on startup"""
    global risk_controller

    # Initialize with default Bitcoin risk limits
    limits = BitcoinRiskLimits(
        base_max_var=100000,
        base_max_cvar=150000,
        base_max_delta=100,
        base_max_gamma=30,
        base_max_vega=1000,
        base_max_theta=2000,
        max_position_concentration=0.20,
        dynamic_adjustment=True
    )

    risk_controller = BitcoinRiskController(
        risk_limits=limits,
        portfolio_value=1000000,
        var_method='historical',
        cvar_method='historical'
    )

    # Initialize with sample data
    sample_greeks = {
        'delta': 0.0,
        'gamma': 0.0,
        'vega': 0.0,
        'theta': 0.0,
        'rho': 0.0
    }
    sample_returns = np.random.normal(0.001, 0.03, 100).tolist()

    risk_controller.update_portfolio_state(
        positions=[],
        greeks=sample_greeks,
        returns=sample_returns,
        current_iv=0.60
    )

    print("✓ Risk Controller initialized")
    print("✓ API server ready")


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        controller_active=risk_controller is not None
    )


@app.get("/api/risk/status", response_model=RiskStatusResponse)
async def get_risk_status():
    """Get current risk status"""
    if risk_controller is None:
        raise HTTPException(status_code=503, detail="Risk controller not initialized")

    report = risk_controller.generate_risk_report()

    return RiskStatusResponse(
        timestamp=report['timestamp'],
        portfolio_value=report['portfolio_value'],
        volatility_regime=report['volatility_regime'],
        current_iv=report['current_iv'],
        current_greeks=report['current_greeks'],
        current_var=report['current_var'],
        current_cvar=report['current_cvar'],
        adjusted_limits=report['adjusted_limits'],
        utilization=report['utilization'],
        statistics=report['statistics']
    )


@app.post("/api/risk/check-order", response_model=OrderCheckResponse)
async def check_order(order: OrderRequest):
    """
    Perform pre-trade risk check on proposed order.
    Returns approval status and detailed risk analysis.
    """
    if risk_controller is None:
        raise HTTPException(status_code=503, detail="Risk controller not initialized")

    # Convert to OrderProposal
    proposal = OrderProposal(
        symbol=order.symbol,
        option_type=order.option_type,
        strike=order.strike,
        expiry=order.expiry,
        quantity=order.quantity,
        direction=order.direction,
        underlying_price=order.underlying_price,
        volatility=order.volatility,
        risk_free_rate=order.risk_free_rate
    )

    # Perform risk check
    result = risk_controller.check_order(proposal)

    # Broadcast to WebSocket subscribers
    await manager.broadcast({
        'type': 'order_check',
        'data': {
            'order': order.dict(),
            'result': {
                'status': result.status.value,
                'approved': result.approved,
                'reasons': result.reasons
            }
        },
        'timestamp': datetime.now().isoformat()
    }, 'orders')

    return OrderCheckResponse(
        status=result.status.value,
        approved=result.approved,
        reasons=result.reasons,
        risk_metrics=result.risk_metrics,
        proposed_metrics=result.proposed_metrics,
        limit_utilization=result.limit_utilization,
        volatility_regime=result.volatility_regime,
        timestamp=result.timestamp.isoformat()
    )


@app.post("/api/orders/submit")
async def submit_order(order: OrderRequest, background_tasks: BackgroundTasks):
    """
    Submit order for execution (with automatic risk check).
    Order is rejected if it violates risk limits.
    """
    if risk_controller is None:
        raise HTTPException(status_code=503, detail="Risk controller not initialized")

    # Convert to OrderProposal
    proposal = OrderProposal(
        symbol=order.symbol,
        option_type=order.option_type,
        strike=order.strike,
        expiry=order.expiry,
        quantity=order.quantity,
        direction=order.direction,
        underlying_price=order.underlying_price,
        volatility=order.volatility,
        risk_free_rate=order.risk_free_rate
    )

    # Perform risk check
    result = risk_controller.check_order(proposal)

    if not result.approved:
        # Broadcast rejection
        await manager.broadcast({
            'type': 'order_rejected',
            'data': {
                'order': order.dict(),
                'reasons': result.reasons
            },
            'timestamp': datetime.now().isoformat()
        }, 'orders')

        raise HTTPException(
            status_code=400,
            detail={
                'message': 'Order rejected by risk controller',
                'reasons': result.reasons,
                'risk_check': {
                    'status': result.status.value,
                    'limit_utilization': result.limit_utilization
                }
            }
        )

    # Order approved - execute (simulation here)
    execution_result = {
        'order_id': f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'status': 'executed',
        'order': order.dict(),
        'execution_price': order.strike * 0.1,  # Simulated
        'timestamp': datetime.now().isoformat()
    }

    # Broadcast execution
    await manager.broadcast({
        'type': 'order_executed',
        'data': execution_result,
        'timestamp': datetime.now().isoformat()
    }, 'orders')

    return {
        'success': True,
        'execution': execution_result,
        'risk_check': {
            'status': result.status.value,
            'reasons': result.reasons
        }
    }


@app.post("/api/portfolio/update")
async def update_portfolio(update: PortfolioUpdate):
    """Update portfolio state for risk calculations"""
    if risk_controller is None:
        raise HTTPException(status_code=503, detail="Risk controller not initialized")

    risk_controller.update_portfolio_state(
        positions=update.positions,
        greeks=update.greeks,
        returns=update.returns,
        current_iv=update.current_iv
    )

    # Broadcast update
    await manager.broadcast({
        'type': 'portfolio_updated',
        'data': {
            'greeks': update.greeks,
            'current_iv': update.current_iv
        },
        'timestamp': datetime.now().isoformat()
    }, 'risk')

    return {'success': True, 'message': 'Portfolio updated'}


@app.get("/api/risk/max-order-size")
async def get_max_order_size(
    symbol: str,
    option_type: str,
    strike: float,
    expiry: float,
    direction: str,
    underlying_price: float,
    volatility: float,
    risk_metric: str = "all"
):
    """Calculate maximum order size within risk limits"""
    if risk_controller is None:
        raise HTTPException(status_code=503, detail="Risk controller not initialized")

    template = OrderProposal(
        symbol=symbol,
        option_type=option_type,
        strike=strike,
        expiry=expiry,
        quantity=1,
        direction=direction,
        underlying_price=underlying_price,
        volatility=volatility
    )

    max_size = risk_controller.get_max_order_size(template, risk_metric=risk_metric)

    return {
        'max_quantity': max_size,
        'risk_metric': risk_metric,
        'volatility_regime': risk_controller.risk_limits.volatility_regime.value
    }


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws/risk")
async def websocket_risk_updates(websocket: WebSocket):
    """
    WebSocket for real-time risk metric updates.
    Sends risk status every second.
    """
    await manager.connect(websocket, 'risk')

    try:
        # Send initial status
        if risk_controller:
            report = risk_controller.generate_risk_report()
            await websocket.send_json({
                'type': 'risk_update',
                'data': report,
                'timestamp': datetime.now().isoformat()
            })

        # Keep connection alive and send updates
        while True:
            await asyncio.sleep(update_interval)

            if risk_controller:
                report = risk_controller.generate_risk_report()
                await websocket.send_json({
                    'type': 'risk_update',
                    'data': report,
                    'timestamp': datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, 'risk')
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, 'risk')


@app.websocket("/ws/pnl")
async def websocket_pnl_updates(websocket: WebSocket):
    """
    WebSocket for real-time P&L updates.
    Sends portfolio value and P&L every second.
    """
    await manager.connect(websocket, 'pnl')

    try:
        # Send updates
        while True:
            await asyncio.sleep(update_interval)

            if risk_controller:
                # Simulate P&L calculation
                pnl_data = {
                    'portfolio_value': risk_controller.portfolio_value,
                    'daily_pnl': np.random.normal(1000, 5000),  # Simulated
                    'greeks': risk_controller.current_greeks,
                    'var': risk_controller.current_var,
                    'cvar': risk_controller.current_cvar
                }

                await websocket.send_json({
                    'type': 'pnl_update',
                    'data': pnl_data,
                    'timestamp': datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, 'pnl')
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, 'pnl')


@app.websocket("/ws/orders")
async def websocket_order_updates(websocket: WebSocket):
    """
    WebSocket for real-time order updates.
    Receives order check notifications.
    """
    await manager.connect(websocket, 'orders')

    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        manager.disconnect(websocket, 'orders')
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, 'orders')


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("Bitcoin Options Real-Time Monitoring API")
    print("=" * 70)
    print("\nStarting server...")
    print("\nEndpoints:")
    print("  REST API:    http://localhost:8000")
    print("  API Docs:    http://localhost:8000/docs")
    print("  Health:      GET  http://localhost:8000/")
    print("  Risk Status: GET  http://localhost:8000/api/risk/status")
    print("  Check Order: POST http://localhost:8000/api/risk/check-order")
    print("  Submit Order: POST http://localhost:8000/api/orders/submit")
    print("\nWebSocket:")
    print("  Risk Updates: ws://localhost:8000/ws/risk")
    print("  P&L Updates:  ws://localhost:8000/ws/pnl")
    print("  Order Updates: ws://localhost:8000/ws/orders")
    print("=" * 70)
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
