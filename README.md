# Reflexion

**Reflexion** is a passive AI-assisted solution aimed at detecting early signs of dementia in elderly individuals.

Due to a miscommunication regarding the required deliverables, there is currently no source code -- however, this README will lay out the general principles behind Reflexion's design and some avenues to be further pursued. 

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


