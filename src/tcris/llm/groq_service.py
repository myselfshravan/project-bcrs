"""Groq LLM service for AI-powered explanations."""

from groq import Groq
from typing import Dict, Any, Optional
from loguru import logger
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Try to find and load .env file
try:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        # Fallback: try project root relative to this file
        project_root = Path(__file__).parent.parent.parent
        dotenv_path = project_root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
except Exception as e:
    logger.warning(f"Could not load .env file: {e}")


class GroqService:
    """Service for generating AI explanations using Groq API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq service.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not provided and GROQ_API_KEY not in environment")

        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        self.temperature = 0.3  # Lower temp for medical consistency

        # Simple cache to avoid repeated API calls
        self._cache = {}

    def _generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from Groq API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response

        Returns:
            Generated text
        """
        # Check cache
        cache_key = f"{prompt[:100]}_{max_tokens}"
        if cache_key in self._cache:
            logger.debug("Using cached response")
            return self._cache[cache_key]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical AI assistant helping explain bladder cancer "
                            "recurrence predictions. Provide clear, accurate, patient-friendly "
                            "explanations. Always include appropriate medical disclaimers."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )

            result = response.choices[0].message.content.strip()
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._get_fallback_message()

    def _get_fallback_message(self) -> str:
        """Return fallback message when API fails."""
        return (
            "⚠️ AI explanation temporarily unavailable. Please consult with your "
            "healthcare provider for interpretation of these results.\n\n"
            "**Disclaimer**: This tool is for research purposes only and should not "
            "be used for clinical decision-making without physician consultation."
        )

    def explain_prediction(
        self,
        baseline_tumors: int,
        baseline_size: float,
        treatment: str,
        time_horizon: int,
        cox_risk: float,
        recurrence_prob: float,
        risk_level: str
    ) -> str:
        """Generate plain-language explanation of prediction.

        Args:
            baseline_tumors: Number of tumors at baseline
            baseline_size: Tumor size in cm
            treatment: Treatment type
            time_horizon: Prediction time in months
            cox_risk: Cox model risk score
            recurrence_prob: Probability of recurrence
            risk_level: Risk category (Low/Medium/High)

        Returns:
            Plain-language explanation
        """
        prompt = f"""A patient with bladder cancer has the following characteristics:
- Number of tumors: {baseline_tumors}
- Tumor size: {baseline_size} cm
- Treatment: {treatment}

Our AI model predicts:
- Risk Score: {cox_risk:.3f}
- {time_horizon}-month recurrence probability: {recurrence_prob:.1%}
- Risk Level: {risk_level}

Please provide a clear, 2-3 paragraph explanation that:
1. Explains what these numbers mean in simple terms
2. Discusses the key factors influencing this prediction
3. Provides context about what this means for the patient

End with a medical disclaimer that this is for research/educational purposes only."""

        return self._generate(prompt, max_tokens=600)

    def generate_clinical_report(
        self,
        baseline_tumors: int,
        baseline_size: float,
        treatment: str,
        time_horizon: int,
        cox_risk: float,
        recurrence_prob: float,
        risk_level: str,
        additional_features: Dict[str, Any]
    ) -> str:
        """Generate EHR-ready clinical summary.

        Args:
            baseline_tumors: Number of tumors
            baseline_size: Tumor size
            treatment: Treatment type
            time_horizon: Prediction time
            cox_risk: Cox risk score
            recurrence_prob: Recurrence probability
            risk_level: Risk category
            additional_features: Additional patient features

        Returns:
            Clinical report text
        """
        features_str = "\n".join([f"- {k}: {v}" for k, v in additional_features.items()])

        prompt = f"""Generate a clinical report summary for EHR documentation:

PATIENT PROFILE:
- Baseline tumors: {baseline_tumors}
- Tumor size: {baseline_size} cm
- Treatment: {treatment}

ADDITIONAL FEATURES:
{features_str}

AI MODEL OUTPUT:
- Cox Risk Score: {cox_risk:.3f}
- {time_horizon}-month recurrence probability: {recurrence_prob:.1%}
- Risk Stratification: {risk_level}

Please generate a structured clinical report with:
1. SUMMARY: One-line overview
2. RISK ASSESSMENT: Interpretation of scores
3. KEY FACTORS: Main drivers of prediction
4. CLINICAL IMPLICATIONS: What this means for monitoring/treatment
5. DISCLAIMER: Research tool limitations

Use professional medical language suitable for EHR documentation."""

        return self._generate(prompt, max_tokens=800)

    def explain_treatment_choice(
        self,
        patient_profile: Dict[str, Any],
        treatments: Dict[str, Dict[str, float]],
        recommended: str
    ) -> str:
        """Generate treatment recommendation rationale.

        Args:
            patient_profile: Patient characteristics
            treatments: Dict of treatment -> metrics
            recommended: Recommended treatment

        Returns:
            Treatment explanation
        """
        profile_str = "\n".join([f"- {k}: {v}" for k, v in patient_profile.items()])

        treatments_str = []
        for tx, metrics in treatments.items():
            treatments_str.append(
                f"- {tx}: Risk={metrics['risk']:.3f}, "
                f"24-mo survival={metrics['survival']:.1%}"
            )
        treatments_str = "\n".join(treatments_str)

        prompt = f"""For a bladder cancer patient with:
{profile_str}

Our counterfactual analysis compared treatments:
{treatments_str}

Recommended: {recommended}

Please provide a clear explanation that:
1. Explains why {recommended} is recommended for THIS specific patient
2. Compares the expected outcomes across treatments
3. Discusses what factors made the difference
4. Provides patient-friendly context for shared decision-making

Keep it conversational and empowering for the patient. End with appropriate disclaimer."""

        return self._generate(prompt, max_tokens=600)

    def explain_treatment_simple(
        self,
        recommended: str,
        risk_reduction: float,
        survival_improvement: float
    ) -> str:
        """Generate simple treatment explanation for dashboard.

        Args:
            recommended: Recommended treatment
            risk_reduction: % risk reduction vs worst
            survival_improvement: % survival improvement vs worst

        Returns:
            Simple explanation
        """
        prompt = f"""A bladder cancer patient's optimal treatment is {recommended}.

Compared to other options:
- Risk reduction: {risk_reduction:.1f}%
- Survival improvement: {survival_improvement:.1f}%

In 2-3 sentences, explain what this means in simple, encouraging terms for the patient.
Focus on the benefit and reassurance. Keep it brief and positive."""

        return self._generate(prompt, max_tokens=200)
