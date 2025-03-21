#Reflexion

**Reflexion** is an AI-powered smart mirror system designed to detect early signs of cognitive decline and dementia in elderly individuals through continuous monitoring of speech patterns, facial expressions, and interaction patterns. The system provides personalized risk assessments and recommendations for users, caregivers, and healthcare providers.


This README will lay out the general principles behind Reflexion's design and some avenues to be further pursued. 

# Current Gaps in Existing Senior Healthcare Solutions
1. Limited Contextual & Personalized Health Alerts
- Wearables are able to provide alerts, but can trigger false positives easily such as mistaking sitting down quickly for a fall
- AI models lack personalized baselines for each senior's normal activity, leading to unnecessary interventions or missed detections.

2. Poor Early Detection of Cognitive & Functional Decline
- Current solutions mainly react to major health events such as falls or hospitalizations, instead of predicting gradual decline in cognitive or motor functions.
- Dementia and frailty progression are difficult to track through wearables alone.

3. Limited Social & Behavioral insights for Holistic Care
- Loneliness and depression impact seniors' health, but current solutions don't track changes in social behavior over time
- AI models don't integrate conversational or sentiment analysis from virtual assistants to flag mental health risks

# Problem Statement
*How can we leverage AI and InterSystems IRIS Vector Search to detect early signs of cognitive decline in seniors through monitoring signs observable on a daily basis, enalbing timely intervention and independent aging?*

# Key Principles

Reflexion exists as a microphone, camera and speaker system attached to a bathroom mirror. Through the system, elderly individuals are able to converse with a Generative AI-powered chatbot-like feature, which simultaneously provides companionsihop to ward off loneliness while monitoring the elderly individual for warning signs of dementia. Early signs of Mild Cognitive Impairment (MCI) can be detected in factors like simpler sentence structures, language changes, hesitation in speech, or changes in facial expressions, which can be detected by Reflexion's sensors.

Using a LLM in the background provided with existing data from dementia datasets as well as the individual's previous interactions, the patient's current risk level is determined. Based on the risk level, the individual will receive recommendations in Reflexion's companion app, from daily brain training and cognitive exercises at low risk, to informing caregivers and recommending a doctor's appointment at high risk. 

# Datasets
Dementia, being a commonly investigated topic, has several datasets that can be leveraged for research (though not all are accessible as of this time due to membership requirements and other similar constraints). 

- https://www.kaggle.com/datasets/shashwatwork/dementia-prediction-dataset/data - Dementia detection based on speech
- https://dementia.talkbank.org/ - DementiaBank, a global resource on dementia
- https://www.i2lab.info/resources - i2Lab by Keio University, including several resources such as detection of depression and dementia from vocal inflections and facial expressions

## Features

- **Real-time Cognitive Assessment**
  - Speech pattern analysis (hesitation, coherence, complexity)
  - Facial expression monitoring (confusion, stress, attention)
  - Interaction pattern tracking
  - Micro-expression detection

- **Smart Risk Assessment**
  - Vector-based similarity search using InterSystems IRIS
  - Historical trend analysis
  - Personalized risk level categorization
  - Adaptive recommendation system

- **Healthcare Integration**
  - Caregiver notification system
  - Healthcare provider dashboard
  - Longitudinal cognitive health tracking
  - Early intervention alerts

## Technology Stack

- **Backend**
  - FastAPI
  - Python 3.8+
  - InterSystems IRIS
  - TensorFlow/PyTorch
  - OpenCV
  - MediaPipe
  - Librosa

- **AI/ML Components**
  - Speech Recognition
  - Natural Language Processing
  - Computer Vision
  - Sentiment Analysis
  - Vector Similarity Search

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CSP-PC/reflexion.git
cd reflexion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up InterSystems IRIS:
- Install InterSystems IRIS
- Configure connection settings in `reflexion/utils/iris_vector_store.py`

## Usage

1. Start the API server:
```bash
uvicorn reflexion.api.main:app --reload
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

3. Integrate with smart mirror hardware:
- Configure video input
- Set up audio capture
- Connect to the API endpoints

## API Endpoints

- `POST /analyze/interaction`: Submit video and audio for analysis
- `GET /user/{user_id}/history`: Retrieve user's assessment history
- `GET /user/{user_id}/trends`: Get cognitive health trends
- `POST /alert/caregiver`: Send alerts to caregivers

## Risk Levels and Recommendations

### Low Risk
- Continue daily cognitive exercises
- Engage in social activities
- Try brain training games

### Moderate Risk
- Schedule cognitive assessment
- Increase exercise frequency
- Join support groups

### High Risk
- Immediate medical consultation
- Daily monitoring
- Caregiver support

reflexion/
├── data/
│   ├── dementia_speech/
│   │   └── dementia_dataset.csv
│   └── facial_expressions/
│       └── ck_plus/
