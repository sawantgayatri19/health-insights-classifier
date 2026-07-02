import os
from typing import Dict, List, Optional

class HealthExplainer:
    def __init__(self, model=None, api_key: Optional[str] = None):
        self.model = model
        self.risk_levels = {
            "low": {"range": (0, 30), "label": "Low Risk"},
            "moderate": {"range": (31, 70), "label": "Moderate Risk"},
            "high": {"range": (71, 100), "label": "High Risk"}
        }
    
    def _get_risk_level(self, score: float) -> Dict:
        for level, details in self.risk_levels.items():
            if details["range"][0] <= score <= details["range"][1]:
                return {"level": level, **details}
        return self.risk_levels["high"]
    
    def generate_explanation(self, risk_score: float, health_factors: List[str], user_data: Optional[Dict] = None) -> Dict:
        risk_info = self._get_risk_level(risk_score)
        explanation = f"Your health assessment indicates {risk_info['label'].lower()}. Key factors: {', '.join(health_factors)}."
        
        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_info["label"],
            "explanation": explanation,
            "contributing_factors": health_factors,
            "recommendations": ["Consult healthcare provider", "Monitor key factors"]
        }
