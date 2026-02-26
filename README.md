AI Governance Copilot for Horticulture Stabilization
Overview
This project builds an AI-driven Governance Copilot for district-level horticulture price stabilization.
The system predicts crop price crash risk across districts and provides economically justified intervention recommendations based on:
Multi-crop crash probability
Estimated farmer income loss
Fiscal cost comparison
Hybrid policy trigger logic
The goal is to support data-driven decision-making for agricultural price stabilization at scale.
Problem Statement
Horticulture crops such as Onion, Tomato, and Potato frequently experience sharp price crashes due to:
Sudden arrival spikes
Climate-driven supply shocks
Demand–supply mismatches
These crashes cause significant farmer income loss and often require reactive government intervention.
Current systems lack:
Early warning capability
District-level risk scoring
Economic justification modeling
Scalable automation
This project addresses these gaps.
System Architecture
1.Data Layer
Automated Agmarknet API ingestion
Multi-crop support (Onion, Tomato, Potato)
District-level daily price & arrival data
Crop-specific configuration
2. Intelligence Layer
Crop-aware crash detection logic
Smoothed future average minimum calculation
Multi-class severity labeling:
0 → Stable
1 → Moderate
2 → Severe
Multi-class XGBoost model
Class imbalance handling
3. Governance Layer
Hybrid intervention trigger combining:
Severe crash probability threshold
Expected economic loss estimation
Fiscal cost comparison
Intervention is triggered only when:
Probability threshold AND Expected loss > Fiscal cost
This ensures economically justified decisions.
4. GenAI Copilot Layer (Planned via Amazon Bedrock)
District-level policy advisory generation
Crash explanation in plain language
What-if scenario reasoning
Budget-aware intervention summaries
Repository Structure
ai-governance-copilot/
│
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── model_experiments.ipynb
│
└── src/
    ├── ingestion.py
    ├── preprocessing.py
    ├── labeling.py
    ├── model.py
    └── hybrid_trigger.py
Current Progress
Multi-crop API ingestion validated
Crop-aware crash labeling implemented
Multi-class severity framework operational
Hybrid fiscal trigger logic implemented
Project modularized for AWS deployment
AWS Deployment Plan
Amazon S3 → Data lake storage
AWS Lambda → Scheduled ingestion
Amazon SageMaker → Model training & endpoint deployment
Amazon Bedrock (Claude 3 Sonnet) → Governance Copilot reasoning
Next Milestones
AWS pipeline deployment
Bedrock integration
Climate feature integration
Time-series cross-validation
Dashboard layer