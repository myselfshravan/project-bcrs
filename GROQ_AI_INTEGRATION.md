# 🤖 Groq AI Integration - T-CRIS Enhancement

**Date Added**: November 3, 2025
**Status**: ✅ Fully Operational
**Model**: Llama 3.3 70B Versatile (via Groq API)

---

## 📋 Overview

T-CRIS now features **AI-powered clinical explanations** powered by Groq's ultra-fast LLM inference platform, using the Llama 3.3 70B model. This integration provides plain-language explanations of predictions and treatment recommendations, making the system more accessible to patients and clinicians.

---

## ✨ New Features

### 1. **Prediction Explainer** (🎯 Predictions Page)

Located on the Predictions page, after viewing a patient's risk prediction.

**Two AI-powered buttons:**

#### 💬 Explain This Prediction
- Generates a **2-3 paragraph plain-language explanation** of the prediction
- Explains what the risk scores mean in simple terms
- Discusses key factors influencing the prediction
- Provides patient-friendly context
- Includes medical disclaimer

**Example Output:**
```
Let's break down what these numbers mean. The Risk Score of 0.850 indicates
a moderate-to-high likelihood of bladder cancer recurrence. The 24-month
recurrence probability of 45.2% means there's approximately a 45% chance
the cancer will return within 2 years...

The key factors influencing this prediction are:
- Number of tumors (3): Multiple tumors increase recurrence risk
- Tumor size (2.5 cm): Larger tumors correlate with higher risk
- Treatment (placebo): Active treatments like thiotepa can reduce risk

[Medical disclaimer...]
```

#### 📄 Generate Clinical Report
- Creates an **EHR-ready clinical summary**
- Structured format suitable for medical records
- Professional medical language
- Includes risk assessment, key factors, clinical implications
- Ready to copy-paste into documentation

**Example Output:**
```
SUMMARY: 45-year-old patient with 3 baseline tumors, largest 2.5cm,
on placebo treatment. Moderate recurrence risk predicted.

RISK ASSESSMENT: Cox PH model indicates 0.850 risk score with 45.2%
24-month recurrence probability...

KEY FACTORS:
- Tumor burden index: 7.5 (tumors × size)
- Treatment: Placebo (no active intervention)
...

[Full structured report]
```

### 2. **Treatment Rationale** (🔀 Counterfactual Page)

Located on the Counterfactual Analysis page, automatically generated after treatment comparison.

**Features:**
- **Automatic explanation** after comparing treatments
- Explains WHY a specific treatment is recommended for THIS patient
- Compares expected outcomes across all treatments
- Patient-friendly language for shared decision-making
- Quantifies treatment benefits

**Additional Component:**
- **📊 Treatment Benefit Summary** (expandable)
  - Shows risk reduction percentage vs. worst treatment
  - Shows survival improvement percentage
  - Brief, encouraging summary for patients

**Example Output:**
```
For your specific case with 3 tumors and 2.5cm size, thiotepa is
recommended because it offers the best balance of efficacy and tolerability.

Compared to placebo, thiotepa is expected to:
- Reduce your recurrence risk by 25.5%
- Improve your 24-month survival probability by 15.3%

The model predicts this benefit is particularly strong for patients
with your tumor burden profile...

[Patient-friendly explanation continues...]
```

---

## 🛠️ Technical Implementation

### Architecture

```
Dashboard (Streamlit)
    ↓
GroqService (src/tcris/llm/groq_service.py)
    ↓
Groq API (Llama 3.3 70B)
    ↓
LLM Response
```

### Key Components

1. **`src/tcris/llm/groq_service.py`** - Main service class
   - `GroqService` class with initialization and caching
   - `explain_prediction()` - Generates prediction explanations
   - `generate_clinical_report()` - Creates EHR-ready summaries
   - `explain_treatment_choice()` - Full treatment rationale
   - `explain_treatment_simple()` - Brief treatment benefit summary

2. **`dashboard/app.py`** - Integration into Streamlit
   - Groq service initialization (cached)
   - Prediction page: Two AI buttons with spinner feedback
   - Counterfactual page: Automatic explanation + expandable summary

3. **`.env`** - Configuration
   - `GROQ_API_KEY` - API key for Groq service

4. **`.gitignore`** - Security
   - `.env` file excluded from version control

### Configuration

**Model Settings:**
- Model: `llama-3.3-70b-versatile`
- Temperature: `0.3` (lower for medical consistency)
- Max tokens: `200-800` depending on task
- Caching: Simple in-memory cache to avoid repeated API calls

**Environment Variables:**
```bash
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Usage

### For Developers

```python
from tcris.llm.groq_service import GroqService

# Initialize (automatically loads from .env)
groq = GroqService()

# Generate prediction explanation
explanation = groq.explain_prediction(
    baseline_tumors=3,
    baseline_size=2.5,
    treatment="placebo",
    time_horizon=24,
    cox_risk=0.85,
    recurrence_prob=45.2,
    risk_level="Moderate"
)

# Generate clinical report
report = groq.generate_clinical_report(
    baseline_tumors=3,
    baseline_size=2.5,
    treatment="placebo",
    time_horizon=24,
    cox_risk=0.85,
    recurrence_prob=45.2,
    risk_level="Moderate",
    additional_features={"Tumor Burden": 7.5}
)

# Explain treatment choice
treatment_explanation = groq.explain_treatment_choice(
    patient_profile={"Tumors": 3, "Size": "2.5 cm"},
    treatments={"placebo": {"risk": 0.9, "survival": 0.6},
                "thiotepa": {"risk": 0.7, "survival": 0.75}},
    recommended="thiotepa"
)
```

### For Dashboard Users

1. **Launch Dashboard:**
   ```bash
   python3 -m streamlit run dashboard/app.py
   ```

2. **On Predictions Page:**
   - Enter patient characteristics
   - Click "Predict Risk"
   - Scroll down to "🤖 AI-Powered Insights"
   - Click "💬 Explain This Prediction" for plain-language explanation
   - Click "📄 Generate Clinical Report" for EHR-ready summary

3. **On Counterfactual Page:**
   - Enter patient characteristics
   - Click "Compare Treatments"
   - Scroll down to see automatic "🤖 AI Treatment Rationale"
   - Expand "📊 Treatment Benefit Summary" for brief summary

---

## 🔒 Security & Privacy

### API Key Protection
- API key stored in `.env` file (not committed to git)
- `.gitignore` configured to exclude `.env`
- Environment variable loading via `python-dotenv`

### Medical Safety
- All outputs include medical disclaimers
- Clearly labeled as research/educational tool
- Emphasizes need for physician consultation
- No automated clinical decisions

### Data Privacy
- No patient data sent to Groq API beyond what's needed for explanation
- Simple caching prevents repeated API calls
- No persistent storage of API responses

---

## 📊 Benefits

### For Patients
1. **Plain-language explanations** make predictions understandable
2. **Treatment rationales** support shared decision-making
3. **Encouraging, positive framing** while maintaining accuracy
4. **Empowering information** to discuss with physicians

### For Clinicians
1. **EHR-ready reports** save documentation time
2. **Structured summaries** highlight key factors
3. **Treatment comparisons** support evidence-based decisions
4. **Professional language** suitable for medical records

### For Researchers
1. **Novel integration** of LLM with survival analysis
2. **Demonstration** of AI interpretability in medical context
3. **Template** for adding AI explanations to ML models
4. **Production-ready** implementation pattern

---

## 🧪 Testing

### Automated Test
```bash
python3 scripts/test_groq.py
```

**Tests:**
1. ✅ Groq service initialization
2. ✅ Prediction explanation generation
3. ✅ Treatment explanation generation
4. ✅ API connectivity and error handling

**Expected Output:**
```
Testing Groq LLM service...
1. Initializing Groq service...
   ✓ Service initialized successfully
2. Testing prediction explanation...
   ✓ Generated explanation
3. Testing treatment explanation...
   ✓ Generated treatment explanation

✅ All Groq integration tests PASSED!
```

### Manual Testing
1. Launch dashboard
2. Go to Predictions page
3. Generate prediction
4. Click AI buttons
5. Verify explanations are clear and accurate

---

## 🎯 Performance

### Speed
- **Groq inference**: ~1-2 seconds per request (ultra-fast!)
- **Llama 3.3 70B**: High-quality outputs with minimal latency
- **Caching**: Instant response for repeated queries

### Cost
- Groq offers generous free tier
- Pay-as-you-go pricing for production
- Caching reduces API call volume

### Reliability
- Error handling with graceful fallbacks
- Fallback message if API unavailable
- Clear error logging with loguru

---

## 🔮 Future Enhancements

### Potential Additions
1. **Multi-modal explanations**: Add visualizations to text
2. **Interactive Q&A**: Let users ask follow-up questions
3. **Personalized language**: Adjust medical terminology level
4. **Multi-language support**: Translate explanations
5. **Voice output**: Text-to-speech for accessibility
6. **Explanation history**: Track and compare past explanations

### Technical Improvements
1. **Prompt optimization**: Fine-tune prompts for better outputs
2. **Streaming responses**: Show text as it's generated
3. **Better caching**: Use Redis or similar for persistence
4. **A/B testing**: Compare different explanation styles
5. **Feedback collection**: Let users rate explanations

---

## 📚 References

- **Groq**: https://groq.com/
- **Llama 3.3**: Meta's latest large language model
- **Groq Python SDK**: https://github.com/groq/groq-python

---

## ✅ Integration Checklist

- [x] Install groq package
- [x] Create GroqService class
- [x] Implement explanation methods
- [x] Add to dashboard (Predictions page)
- [x] Add to dashboard (Counterfactual page)
- [x] Configure .env file
- [x] Update .gitignore
- [x] Write tests
- [x] Test integration
- [x] Update documentation

---

## 🎊 Summary

The Groq AI integration transforms T-CRIS from a prediction platform into a **complete clinical decision support system** with:

- ✅ Ultra-fast LLM inference (Groq)
- ✅ State-of-the-art language model (Llama 3.3 70B)
- ✅ Patient-friendly explanations
- ✅ Clinician-ready documentation
- ✅ Production-ready implementation
- ✅ Full error handling and security

**This enhancement significantly increases the system's value for presentation and real-world deployment.**

---

*Last Updated: November 3, 2025 - 02:08 AM*
