"""
Multi-Agent Volatility Modeling
================================

Two approaches for extracting factors that explain market volatility:

1. Observable Factors (Recommended for Production)
   - Extract factors directly from real market data
   - Directly observable and verifiable
   - Suitable for production trading systems

2. Latent Factors (For Research)
   - Extract factors from simulated agent behavior
   - Useful for theoretical research
   - Factors emerge from multi-agent interactions
"""

from .agent_volatility import (
    MultiAgentVolatilityExtractor,
    AgentState,
    MarketState,
    MarketMakerAgent,
    ArbitrageurAgent,
    TrendFollowerAgent,
    FundamentalInvestorAgent
)

from .observable_factor_extractor import (
    ObservableFactorExtractor,
    ObservableFactors
)

from .factor_modeling import (
    MultiAgentFactorModeler,
    AgentConfiguration
)

from .factor_importance_analyzer import (
    FactorImportanceAnalyzer,
    FactorImportance,
    AgentFactorRelationship,
    FactorAnalysisResult
)

from .unified_factor_pipeline import (
    UnifiedFactorPipeline,
    UnifiedFactorResult
)

from .agent_physical_integration import (
    AgentPhysicalIntegration,
    IntegratedVolatilityPrediction
)

from .physical_model_constraint import (
    PhysicalModelConstraint,
    JumpDiffusionModel,
    JumpDiffusionParams,
    PhysicalModelFit
)

__all__ = [
    # Agent-Physical Integration (Recommended) ⭐⭐⭐
    'AgentPhysicalIntegration',
    'IntegratedVolatilityPrediction',
    'PhysicalModelConstraint',
    'JumpDiffusionModel',
    'JumpDiffusionParams',
    'PhysicalModelFit',
    
    # Unified Pipeline
    'UnifiedFactorPipeline',
    'UnifiedFactorResult',
    
    # Observable factors
    'ObservableFactorExtractor',
    'ObservableFactors',
    
    # Latent factors (research)
    'MultiAgentVolatilityExtractor',
    'AgentState',
    'MarketState',
    'MarketMakerAgent',
    'ArbitrageurAgent',
    'TrendFollowerAgent',
    'FundamentalInvestorAgent',
    
    # Factor modeling
    'MultiAgentFactorModeler',
    'AgentConfiguration',
    
    # Factor importance analysis
    'FactorImportanceAnalyzer',
    'FactorImportance',
    'AgentFactorRelationship',
    'FactorAnalysisResult'
]

