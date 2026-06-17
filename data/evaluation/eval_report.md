# Noema Dataset Evaluation

No fine-tuning was performed. This report evaluates deterministic routing, emotion classification, source selection, safety, and response behavior.

## Metrics

| Metric | Result |
|---|---:|
| `intent_accuracy` | 77.74% |
| `emotion_accuracy` | 0.00% |
| `internet_routing_accuracy` | 98.30% |
| `research_routing_accuracy` | 100.00% |
| `safety_routing_accuracy` | 100.00% |
| `advice_answer_rate` | 100.00% |
| `decision_answer_rate` | 90.35% |
| `over_reflection_rate` | 0.00% |
| `bad_phrase_rate` | 0.00% |
| `generic_decision_template_rate` | 0.00% |
| `false_crisis_rate` | 0.00% |
| `mixed_intent_success_rate` | 100.00% |
| `distortion_detection_rate` | 100.00% |
| `relationship_routing_accuracy` | 100.00% |
| `business_routing_accuracy` | 100.00% |
| `casual_chat_success_rate` | 100.00% |
| `humanization_score` | 100.00% |
| `research_humanization_score` | 100.00% |
| `symptom_education_score` | 100.00% |
| `identity_depth_score` | 100.00% |
| `response_variety_score` | 100.00% |
| `narrative_memory_score` | 100.00% |
| `offline_language_routing_accuracy` | 100.00% |
| `internet_suppression_accuracy` | 100.00% |
| `casual_rapport_success` | 100.00% |
| `slang_understanding_success` | 100.00% |
| `wrong_internet_trigger_rate` | 0.00% |
| `wrong_safety_trigger_rate` | 0.00% |
| `hybrid_knowledge_engine_success` | 100.00% |
| `longform_response_success_rate` | 90.55% |
| `therapy_retrieval_success_rate` | 100.00% |

## Required Targets

- [x] `over_reflection_rate_below_5_percent`
- [x] `over_reflection_rate_below_3_percent`
- [x] `advice_answer_rate_above_90_percent`
- [x] `decision_answer_rate_above_90_percent`
- [x] `safety_routing_accuracy_100_percent`
- [x] `generic_decision_template_rate_below_5_percent`
- [x] `false_crisis_rate_below_1_percent`
- [x] `false_crisis_rate_below_0_5_percent`
- [x] `mixed_intent_success_rate_above_80_percent`
- [x] `distortion_detection_rate_above_90_percent`
- [x] `relationship_routing_accuracy_above_90_percent`
- [x] `business_routing_accuracy_above_90_percent`
- [x] `casual_chat_success_rate_above_90_percent`
- [x] `longform_response_success_rate_above_85_percent`
- [x] `humanization_score_above_90_percent`
- [x] `research_humanization_score_above_90_percent`
- [x] `symptom_education_score_above_90_percent`
- [x] `identity_depth_score_above_90_percent`
- [x] `response_variety_score_above_60_percent`
- [x] `narrative_memory_score_above_90_percent`
- [x] `offline_language_routing_accuracy_above_90_percent`
- [x] `internet_suppression_accuracy_above_95_percent`
- [x] `casual_rapport_success_above_90_percent`
- [x] `slang_understanding_success_above_85_percent`
- [x] `wrong_internet_trigger_rate_below_5_percent`
- [x] `wrong_safety_trigger_rate_below_0_5_percent`
- [x] `hybrid_knowledge_engine_success_100_percent`

## Notes

- Emotion datasets are evaluated only as input-to-label classifiers.
- HH-RLHF and CounselChat are not counted as routing ground truth.
- Failure examples are stored in the JSON report for iteration.
