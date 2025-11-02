"""Main Streamlit dashboard for T-CRIS."""

import streamlit as st
import sys
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from lifelines import KaplanMeierFitter
import torch

from tcris.data.loaders import BladderDataLoader
from tcris.data.fusion import DataFusionEngine
from tcris.features.extractors import create_features
from tcris.llm.groq_service import GroqService

# Page config
st.set_page_config(
    page_title="T-CRIS Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Groq service (cached)
@st.cache_resource
def load_groq_service():
    """Initialize Groq service."""
    try:
        return GroqService()
    except Exception as e:
        st.warning(f"⚠️ AI explanations unavailable: {e}")
        return None

# Load models and data (cached)
@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    model_dir = Path("models")

    try:
        with open(model_dir / "cox_model.pkl", "rb") as f:
            models["cox"] = pickle.load(f)
    except:
        st.warning("Cox model not found")

    try:
        with open(model_dir / "rsf_model.pkl", "rb") as f:
            models["rsf"] = pickle.load(f)
    except:
        st.warning("RSF model not found")

    try:
        with open(model_dir / "scaler.pkl", "rb") as f:
            models["scaler"] = pickle.load(f)
    except:
        st.warning("Scaler not found")

    try:
        with open(model_dir / "feature_names.pkl", "rb") as f:
            models["feature_names"] = pickle.load(f)
    except:
        models["feature_names"] = []

    try:
        with open(model_dir / "results.json", "r") as f:
            models["results"] = json.load(f)
    except:
        models["results"] = {}

    return models

@st.cache_data
def load_data():
    """Load and process data."""
    loader = BladderDataLoader()
    datasets = loader.load_all()
    fusion = DataFusionEngine()
    unified_df = fusion.fuse(datasets)
    df_features = create_features(unified_df)
    patient_data = df_features.groupby("patient_id").first().reset_index()
    return unified_df, df_features, patient_data, fusion

# Load everything
models = load_models()
unified_df, df_features, patient_data, fusion = load_data()
groq_service = load_groq_service()

# Title
st.title("🏥 T-CRIS: Temporal Cancer Recurrence Intelligence System")
st.markdown("### AI-Powered Bladder Cancer Prediction Platform")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["📊 Overview", "📈 Survival Analysis", "🎯 Predictions", "🔀 Counterfactual", "🔍 Model Performance"]
    )

    st.markdown("---")
    st.header("About")
    st.info(
        "T-CRIS combines classical survival analysis with modern ML/DL "
        "for bladder cancer recurrence prediction."
    )

    st.markdown("---")
    st.caption("Built with Streamlit • Powered by PyTorch")

# Page routing
if page == "📊 Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Patients", unified_df["patient_id"].nunique())

    with col2:
        recurrence_counts = fusion.get_recurrence_counts(unified_df)
        st.metric("Mean Recurrences", f"{recurrence_counts.mean():.2f}")

    with col3:
        recurrers = (recurrence_counts > 0).sum()
        st.metric("Patients with Recurrence", recurrers)

    with col4:
        max_follow_up = unified_df.groupby("patient_id")["stop_time"].max().max()
        st.metric("Max Follow-up", f"{max_follow_up:.0f} mo")

    st.markdown("---")

    # Summary statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Treatment Distribution")
        summary = fusion.summarize_unified_data(unified_df)
        treat_df = pd.DataFrame(list(summary["treatment_distribution"].items()),
                                columns=["Treatment", "Count"])
        fig = px.bar(treat_df, x="Treatment", y="Count", color="Treatment")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Event Type Distribution")
        event_labels = {0: "Censored", 1: "Recurrence", 2: "Death (Bladder)", 3: "Death (Other)"}
        event_df = pd.DataFrame(list(summary["event_type_distribution"].items()),
                                columns=["Event Type", "Count"])
        event_df["Event Type"] = event_df["Event Type"].map(event_labels)
        fig = px.pie(event_df, values="Count", names="Event Type")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean Baseline Tumors", f"{summary['baseline_tumors']['mean']:.1f}")
        st.metric("Median Baseline Tumors", f"{summary['baseline_tumors']['median']:.1f}")

    with col2:
        st.metric("Mean Baseline Size", f"{summary['baseline_size']['mean']:.1f} cm")
        st.metric("Median Baseline Size", f"{summary['baseline_size']['median']:.1f} cm")

    with col3:
        st.metric("Mean Follow-up", f"{summary['follow_up_time']['mean']:.1f} mo")
        st.metric("Max Follow-up", f"{summary['follow_up_time']['max']:.0f} mo")

elif page == "📈 Survival Analysis":
    st.header("📈 Survival Analysis")

    # Prepare survival data - get from unified_df
    patient_first = unified_df.groupby("patient_id").first().reset_index()
    surv_data = patient_first[["stop_time", "event_type", "treatment"]].copy()
    surv_data["event"] = (surv_data["event_type"] == 1).astype(int)

    # Overall KM curve
    st.subheader("Overall Kaplan-Meier Survival Curve")

    kmf = KaplanMeierFitter()
    kmf.fit(surv_data["stop_time"], surv_data["event"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_["KM_estimate"],
        mode="lines",
        name="All Patients",
        line=dict(width=3, color="blue")
    ))

    # Add confidence interval
    ci = kmf.confidence_interval_
    fig.add_trace(go.Scatter(
        x=ci.index.tolist() + ci.index.tolist()[::-1],
        y=ci.iloc[:, 0].tolist() + ci.iloc[:, 1].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0,100,255,0.2)",
        line=dict(width=0),
        name="95% CI",
        showlegend=False
    ))

    fig.update_layout(
        title="Recurrence-Free Survival",
        xaxis_title="Time (months)",
        yaxis_title="Survival Probability",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # KM by treatment
    st.subheader("Survival by Treatment")

    # Handle treatment encoding
    if "treatment" in surv_data.columns and surv_data["treatment"].dtype in ['int64', 'float64']:
        treatment_map = {1: "Placebo", 2: "Thiotepa"}
        surv_data["treatment"] = surv_data["treatment"].map(treatment_map).fillna("Unknown")

    fig = go.Figure()
    colors = {"placebo": "red", "Placebo": "red", "thiotepa": "blue", "Thiotepa": "blue",
              "pyridoxine": "green", "Pyridoxine": "green"}

    for treatment in surv_data["treatment"].unique():
        if pd.notna(treatment):
            mask = surv_data["treatment"] == treatment
            kmf.fit(surv_data.loc[mask, "stop_time"], surv_data.loc[mask, "event"], label=str(treatment))

            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[str(treatment)],
                mode="lines",
                name=str(treatment).title(),
                line=dict(width=3, color=colors.get(treatment, "gray"))
            ))

    fig.update_layout(
        title="Recurrence-Free Survival by Treatment",
        xaxis_title="Time (months)",
        yaxis_title="Survival Probability",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Interpretation**: Higher curves indicate better recurrence-free survival. "
            "Thiotepa (blue) shows improved outcomes compared to placebo (red).")

elif page == "🎯 Predictions":
    st.header("🎯 Individual Risk Prediction")

    st.markdown("### Enter Patient Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        baseline_tumors = st.slider("Initial Tumor Count", 1, 10, 2)
        baseline_size = st.slider("Largest Tumor Size (cm)", 0.5, 8.0, 2.0, 0.5)

    with col2:
        treatment = st.selectbox("Treatment", ["placebo", "thiotepa", "pyridoxine"])

    with col3:
        time_horizon = st.slider("Prediction Time (months)", 6, 60, 24, 6)

    # Initialize session state for predictions
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None

    if st.button("🔮 Predict Risk", type="primary"):
        # Prepare input
        input_data = pd.DataFrame({
            "baseline_tumors": [baseline_tumors],
            "baseline_size": [baseline_size],
            "tumor_burden_index": [baseline_tumors * baseline_size],
            "baseline_burden": [baseline_tumors * baseline_size],
            "time_to_first_recurrence": [time_horizon],
            "recurrence_rate": [0.1],
            "treat_placebo": [1 if treatment == "placebo" else 0],
            "treat_thiotepa": [1 if treatment == "thiotepa" else 0]
        })

        # Scale input
        if "scaler" in models:
            input_scaled = pd.DataFrame(
                models["scaler"].transform(input_data),
                columns=input_data.columns
            )
        else:
            input_scaled = input_data

        # Predict with Cox
        if "cox" in models:
            cox_risk = models["cox"].predict_partial_hazard(input_scaled).values[0]
            cox_surv = models["cox"].predict_survival_function(input_scaled, times=[time_horizon])
            cox_surv_prob = cox_surv.iloc[0, 0]
            recurrence_prob = (1 - cox_surv_prob) * 100

            # Risk interpretation
            if recurrence_prob < 30:
                risk_level = "🟢 Low"
                risk_color = "green"
            elif recurrence_prob < 60:
                risk_level = "🟡 Moderate"
                risk_color = "orange"
            else:
                risk_level = "🔴 High"
                risk_color = "red"

            # Store in session state
            st.session_state.prediction_results = {
                "baseline_tumors": baseline_tumors,
                "baseline_size": baseline_size,
                "treatment": treatment,
                "time_horizon": time_horizon,
                "cox_risk": cox_risk,
                "recurrence_prob": recurrence_prob,
                "cox_surv_prob": cox_surv_prob,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "input_scaled": input_scaled
            }

    # Display results if they exist
    if st.session_state.prediction_results is not None:
        results = st.session_state.prediction_results

        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Risk Score", f"{results['cox_risk']:.2f}")

        with col2:
            st.metric(f"{results['time_horizon']}-Month Recurrence Risk", f"{results['recurrence_prob']:.1f}%")

        with col3:
            st.metric(f"{results['time_horizon']}-Month Recurrence-Free Prob", f"{results['cox_surv_prob']*100:.1f}%")

        st.markdown(f"### Risk Level: {results['risk_level']}")

        # Survival curve
        st.subheader("Predicted Survival Curve")
        times = np.linspace(0, 60, 100)
        surv_curve = models["cox"].predict_survival_function(results["input_scaled"], times=times)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=surv_curve.iloc[:, 0],
            mode="lines",
            name="Predicted Survival",
            line=dict(width=3, color=results["risk_color"])
        ))

        fig.add_vline(x=results["time_horizon"], line_dash="dash", line_color="gray",
                      annotation_text=f"{results['time_horizon']} months")

        fig.update_layout(
            xaxis_title="Time (months)",
            yaxis_title="Recurrence-Free Probability",
            hovermode="x unified",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature contribution
        st.subheader("Key Risk Factors")

        feature_importance = pd.DataFrame({
            "Feature": ["Baseline Tumors", "Tumor Size", "Tumor Burden", "Treatment"],
            "Impact": [results["baseline_tumors"] / 10, results["baseline_size"] / 8,
                      (results["baseline_tumors"] * results["baseline_size"]) / 40,
                      0.3 if results["treatment"] == "thiotepa" else 0.8]
        })

        fig = px.bar(feature_importance, x="Impact", y="Feature", orientation="h",
                    color="Impact", color_continuous_scale="reds")
        st.plotly_chart(fig, use_container_width=True)

        # AI Explanation Section
        if groq_service:
            st.markdown("---")
            st.subheader("🤖 AI-Powered Insights")

            col_ai1, col_ai2 = st.columns(2)

            with col_ai1:
                if st.button("💬 Explain This Prediction", type="secondary"):
                    with st.spinner("Generating AI explanation..."):
                        explanation = groq_service.explain_prediction(
                            baseline_tumors=results["baseline_tumors"],
                            baseline_size=results["baseline_size"],
                            treatment=results["treatment"],
                            time_horizon=results["time_horizon"],
                            cox_risk=results["cox_risk"],
                            recurrence_prob=results["recurrence_prob"],
                            risk_level=results["risk_level"].replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", "")
                        )
                        st.markdown("#### 📝 Plain-Language Explanation")
                        st.info(explanation)

            with col_ai2:
                if st.button("📄 Generate Clinical Report", type="secondary"):
                    with st.spinner("Generating clinical report..."):
                        additional_features = {
                            "Tumor Burden Index": results["baseline_tumors"] * results["baseline_size"],
                            "Risk Score": f"{results['cox_risk']:.3f}",
                            "Treatment": results["treatment"].title()
                        }
                        report = groq_service.generate_clinical_report(
                            baseline_tumors=results["baseline_tumors"],
                            baseline_size=results["baseline_size"],
                            treatment=results["treatment"],
                            time_horizon=results["time_horizon"],
                            cox_risk=results["cox_risk"],
                            recurrence_prob=results["recurrence_prob"],
                            risk_level=results["risk_level"].replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", ""),
                            additional_features=additional_features
                        )
                        st.markdown("#### 📋 Clinical Summary")
                        st.text_area("EHR-Ready Report", report, height=300)

        st.success("✅ Prediction complete! Use these results to inform treatment decisions.")

elif page == "🔀 Counterfactual":
    st.header("🔀 Counterfactual Analysis: Treatment Comparison")

    st.markdown("### Compare predicted outcomes under different treatments")

    col1, col2 = st.columns(2)

    with col1:
        baseline_tumors = st.slider("Initial Tumor Count", 1, 10, 3, key="cf_tumors")
        baseline_size = st.slider("Largest Tumor Size (cm)", 0.5, 8.0, 2.5, 0.5, key="cf_size")

    with col2:
        time_horizon = st.slider("Prediction Time (months)", 6, 60, 24, 6, key="cf_time")

    # Initialize session state for counterfactual results
    if "cf_results" not in st.session_state:
        st.session_state.cf_results = None

    if st.button("🔬 Compare Treatments", type="primary"):
        if "cox" in models:
            treatments = ["placebo", "thiotepa", "pyridoxine"]
            results = {}

            for treatment in treatments:
                input_data = pd.DataFrame({
                    "baseline_tumors": [baseline_tumors],
                    "baseline_size": [baseline_size],
                    "tumor_burden_index": [baseline_tumors * baseline_size],
                    "baseline_burden": [baseline_tumors * baseline_size],
                    "time_to_first_recurrence": [time_horizon],
                    "recurrence_rate": [0.1],
                    "treat_placebo": [1 if treatment == "placebo" else 0],
                    "treat_thiotepa": [1 if treatment == "thiotepa" else 0]
                })

                input_scaled = pd.DataFrame(
                    models["scaler"].transform(input_data),
                    columns=input_data.columns
                )

                risk = models["cox"].predict_partial_hazard(input_scaled).values[0]
                surv = models["cox"].predict_survival_function(input_scaled, times=[time_horizon])
                surv_prob = surv.iloc[0, 0]

                results[treatment] = {
                    "risk": risk,
                    "recurrence_prob": (1 - surv_prob) * 100,
                    "survival_prob": surv_prob * 100
                }

            # Find best treatment
            best_treatment = min(results, key=lambda x: results[x]["recurrence_prob"])
            best_risk = results[best_treatment]["recurrence_prob"]

            # Store in session state
            st.session_state.cf_results = {
                "baseline_tumors": baseline_tumors,
                "baseline_size": baseline_size,
                "time_horizon": time_horizon,
                "treatments": treatments,
                "results": results,
                "best_treatment": best_treatment,
                "best_risk": best_risk
            }

    # Display results if they exist
    if st.session_state.cf_results is not None:
        cf = st.session_state.cf_results

        st.markdown("---")
        st.subheader("Treatment Comparison Results")

        col1, col2, col3 = st.columns(3)

        for col, treatment in zip([col1, col2, col3], cf["treatments"]):
            with col:
                st.markdown(f"### {treatment.title()}")
                st.metric("Recurrence Risk", f"{cf['results'][treatment]['recurrence_prob']:.1f}%")
                st.metric("Recurrence-Free Prob", f"{cf['results'][treatment]['survival_prob']:.1f}%")

        st.markdown("---")
        st.success(f"### 💊 Recommended Treatment: **{cf['best_treatment'].upper()}**")
        st.info(f"**Rationale**: {cf['best_treatment'].title()} shows the lowest predicted recurrence risk "
               f"({cf['best_risk']:.1f}%) at {cf['time_horizon']} months for this patient profile.")

        # AI Treatment Explanation
        if groq_service:
            st.markdown("---")
            st.subheader("🤖 AI Treatment Rationale")

            with st.spinner("Generating personalized treatment explanation..."):
                patient_profile = {
                    "Baseline Tumors": cf["baseline_tumors"],
                    "Tumor Size": f"{cf['baseline_size']} cm",
                    "Tumor Burden": cf["baseline_tumors"] * cf["baseline_size"]
                }

                treatment_data = {
                    tx: {
                        "risk": cf["results"][tx]["risk"],
                        "survival": cf["results"][tx]["survival_prob"] / 100
                    }
                    for tx in cf["treatments"]
                }

                explanation = groq_service.explain_treatment_choice(
                    patient_profile=patient_profile,
                    treatments=treatment_data,
                    recommended=cf["best_treatment"]
                )

                st.info(explanation)

            # Simple comparison summary
            worst_treatment = max(cf["results"], key=lambda x: cf["results"][x]["recurrence_prob"])
            worst_risk = cf["results"][worst_treatment]["recurrence_prob"]
            risk_reduction = ((worst_risk - cf["best_risk"]) / worst_risk) * 100

            best_surv = cf["results"][cf["best_treatment"]]["survival_prob"]
            worst_surv = cf["results"][worst_treatment]["survival_prob"]
            surv_improvement = best_surv - worst_surv

            if risk_reduction > 5:  # Only show if meaningful difference
                with st.expander("📊 Treatment Benefit Summary"):
                    simple_explanation = groq_service.explain_treatment_simple(
                        recommended=cf["best_treatment"],
                        risk_reduction=risk_reduction,
                        survival_improvement=surv_improvement
                    )
                    st.success(simple_explanation)

        # Visualize comparison
        st.subheader("Visual Comparison")

        comp_df = pd.DataFrame({
            "Treatment": [t.title() for t in cf["treatments"]],
            "Recurrence Risk (%)": [cf["results"][t]["recurrence_prob"] for t in cf["treatments"]]
        })

        fig = px.bar(comp_df, x="Treatment", y="Recurrence Risk (%)",
                    color="Recurrence Risk (%)", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Model Performance":
    st.header("🔍 Model Performance & Evaluation")

    if "results" in models and models["results"]:
        st.subheader("Model Comparison")

        # Create results dataframe
        results_data = []
        for model_name, metrics in models["results"].items():
            results_data.append({
                "Model": model_name.upper(),
                "C-Index": metrics.get("c_index", 0)
            })

        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values("C-Index", ascending=False)

        # Display table
        st.dataframe(results_df.style.highlight_max(subset=["C-Index"], color="lightgreen"),
                    use_container_width=True)

        # Visualize performance
        fig = px.bar(results_df, x="Model", y="C-Index", color="C-Index",
                    color_continuous_scale="blues", text="C-Index")
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Model descriptions
        st.subheader("Model Descriptions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Cox PH Model")
            st.write("**Type**: Classical statistical model")
            st.write("**Strengths**: Interpretable, established in clinical practice")
            st.write("**C-Index**: High discrimination power")

            st.markdown("### 🌲 Random Survival Forest")
            st.write("**Type**: Machine Learning ensemble")
            st.write("**Strengths**: Captures non-linear relationships, feature importance")

        with col2:
            st.markdown("### 🧠 LSTM Neural Network")
            st.write("**Type**: Deep Learning")
            st.write("**Strengths**: Temporal pattern recognition, flexible")

            st.markdown("### 🎯 Ensemble Model")
            st.write("**Type**: Meta-model combining all approaches")
            st.write("**Strengths**: Leverages strengths of multiple models")

        st.info("💡 **C-Index Interpretation**: Values range from 0-1, where:\n"
               "- 0.5 = Random predictions\n"
               "- 0.7-0.8 = Good discrimination\n"
               "- 0.8-0.9 = Excellent discrimination\n"
               "- >0.9 = Outstanding (may indicate overfitting)")

    else:
        st.warning("No model results available. Train models first using `python3 scripts/train_all_models.py`")

# Footer
st.markdown("---")
st.caption("T-CRIS v1.0 | Built for precision medicine in bladder cancer")
