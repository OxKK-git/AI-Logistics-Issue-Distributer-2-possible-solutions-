# Logistics Incident Triage - Run Report

Generated: 2026-10-25 07:53:21

## 1. Run summary

| Item | Value |
|---|---|
| Dataset | `logistics_incident_reports.csv` |
| Incidents processed | 620 |
| LLM fallback | loaded models/qwen2.5-7b-instruct.Q4_K_M.gguf in 35.01 s |
| CPU threads | 8 |
| Rule confidence rule | top score >= 5 and lead >= 2 |

## 2. Speed

| Metric | Value |
|---|---|
| Total classification time | 879780.87 ms |
| Throughput | 0.7 incidents/s |
| Average per incident | 1419.001 ms |
| Time spent in rule engine | 172.137 ms (277.6 us per incident) |
| LLM calls | 199 |
| Time spent in LLM | 879605.61 ms |
| Average LLM latency | 4420.13 ms |
| Slowest LLM call | 6584.02 ms |
| Model load time (one-off) | 35.01 s |
| Estimated time if EVERY incident used the LLM | 2740.5 s (speed-up of 3.1x from the hybrid routing) |

## 3. Routing

| Path | Incidents | Share |
|---|---|---|
| Rules (confident) | 421 | 67.9% |
| LLM fallback | 199 | 32.1% |
| Rules (low confidence, LLM unavailable) | 0 | 0.0% |
| Unclassified (no keyword hit, LLM unavailable) | 0 | 0.0% |

Unclassified incidents default to **Urgent** so a human reviews them (fail-safe).

## 4. Category accuracy

**Overall: 599 / 620 = 96.6%**

| Path | Correct | Total | Accuracy |
|---|---|---|---|
| Rules (confident) | 415 | 421 | 98.6% |
| LLM fallback | 184 | 199 | 92.5% |

### Per-category precision / recall

| Category | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Theft/Security | 124 | 98.3% | 93.5% | 0.959 |
| Cold Chain/Temperature | 124 | 93.9% | 99.2% | 0.965 |
| Delay/Transit | 124 | 98.3% | 92.7% | 0.954 |
| Mispick/Order Error | 124 | 96.9% | 99.2% | 0.980 |
| Damage/Handling | 124 | 96.1% | 98.4% | 0.972 |

### Confusion matrix (rows = true category, columns = predicted)

| True \ Pred | THF | CLD | DLY | MSP | DMG | Unclassified |
|---|---|---|---|---|---|---|
| Theft/Security | 116 | 0 | 2 | 3 | 3 | 0 |
| Cold Chain/Temperature | 0 | 123 | 0 | 0 | 1 | 0 |
| Delay/Transit | 0 | 8 | 115 | 1 | 0 | 0 |
| Mispick/Order Error | 0 | 0 | 0 | 123 | 1 | 0 |
| Damage/Handling | 2 | 0 | 0 | 0 | 122 | 0 |

## 5. Urgency triage

**Overall: 596 / 620 = 96.1%**

| True \ Pred | Urgent | Standard |
|---|---|---|
| Urgent | 294 | 16 |
| Standard | 8 | 302 |

- **Urgent recall** (urgent incidents correctly flagged urgent): 94.8%
- **Missed urgent incidents** (the costly error): 16
- **False alarms** (standard flagged urgent): 8

| True category | Urgency correct | Total | Accuracy |
|---|---|---|---|
| Theft/Security | 116 | 124 | 93.5% |
| Cold Chain/Temperature | 123 | 124 | 99.2% |
| Delay/Transit | 111 | 124 | 89.5% |
| Mispick/Order Error | 124 | 124 | 100.0% |
| Damage/Handling | 122 | 124 | 98.4% |

## 6. Why the rules decided (top matched keywords)

- **Theft/Security**: `yard` (7), `security` (5), `parked` (5), `overnight` (5), `missing` (5), `gate` (5), `stol*` (4), `police` (4)
- **Cold Chain/Temperature**: `frozen` (50), `reefer` (35), `cold` (27), `temperature*` (24), `unit` (22), `minus` (22), `ice` (14), `fresh` (12)
- **Delay/Transit**: `hours` (21), `deliver*` (18), `late` (17), `tomorrow` (16), `delay*` (12), `arrive*` (12), `shipment` (11), `minutes` (11)
- **Mispick/Order Error**: `wrong` (44), `customer` (36), `order` (33), `picker*` (32), `ordered` (23), `received` (22), `instead of` (21), `label*` (18)
- **Damage/Handling**: `carton*` (68), `pallet*` (58), `cartons` (56), `boxes` (27), `crush*` (24), `corner*` (21), `dent*` (20), `box` (20)

## 7. Misclassified categories

Showing 21 of 21.

| ID | True | Predicted | Path | Text |
|---|---|---|---|---|
| THF-011 | THF | DMG | rules | Our delivery from the carrier arrived with the seal intact but 18 cartons of tablets are missing from the m... |
| THF-037 | THF | DLY | rules | While parked at the border queue, someone climbed onto the roof and cut the tarp. I found missing cartons o... |
| THF-044 | THF | DMG | rules | Customer reports a shipment arrived with the shrink wrap redone and 7 cartons of video game consoles replac... |
| THF-059 | THF | DLY | llm | Dispatch reports that the driver of unit 417 has stopped responding and the trailer door alert is active. L... |
| DLY-010 | DLY | CLD | llm | The driver reports engine overheating and he has pulled over. Eight pallets of vaccines destined for a rura... |
| DLY-024 | DLY | CLD | llm | The ferry crossing was cancelled due to high winds and our truck with temperature-sensitive chocolate canno... |
| MSP-030 | MSP | DMG | rules | I opened the carton expecting 24 cans of tomato paste and found 24 cans of tomato puree. Packaging looks si... |
| THF-080 | THF | MSP | llm | The pallet of printer toner is not where the system says it is. Bay 9 is empty and the cycle count last wee... |
| THF-099 | THF | MSP | llm | Pallet scan shows 40 cartons of beauty products received, but the actual receiving count was 32, and the mi... |
| THF-102 | THF | DMG | llm | We got a complaint that 30 percent of the cartons of gaming controllers contain bricks wrapped in bubble wr... |
| THF-114 | THF | MSP | llm | The container yard at the port reported that one of our units was released to a driver whose ID does not ma... |
| CLD-098 | CLD | DMG | rules | Customer reports that the cheese consignment smelled sour and the pallets were wet inside the wrap. The tra... |
| DLY-070 | DLY | CLD | llm | Heavy snow closed the pass and 12 trucks are waiting. The cargo includes fresh milk for a school meal progr... |
| DLY-072 | DLY | CLD | llm | The truck engine seized outside the city limits. The chilled sushi for tonight's restaurant opening has 3 h... |
| DLY-075 | DLY | MSP | llm | The ambulance equipment shipment was left behind at the air hub because the manifest was wrong. The county ... |
| DLY-076 | DLY | CLD | llm | My trailer's brake lines failed and the truck has been towed. The load is human blood samples for a transpl... |
| DLY-082 | DLY | CLD | llm | Driver hours are exhausted and the truck is at a layby 30 km from the destination. The load is donor tissue... |
| DLY-089 | DLY | CLD | rules | I had to stop for a repair on the road, and the vaccines will miss the cold chain window if I do not get a ... |
| DLY-091 | DLY | CLD | llm | The airline overbooked the freight and our cargo of insulin cartridges was offloaded. The pharmacy distribu... |
| DMG-076 | DMG | THF | llm | The garment boxes were chewed through at the corners by rodents at the carrier's warehouse. There are dropp... |
| DMG-124 | DMG | THF | llm | Rodents chewed through the sacks of lentils stored at the warehouse. Several sacks have holes and grains on... |

## 8. Missed urgent incidents

Showing 16 of 16.

| ID | Predicted category | Text |
|---|---|---|
| THF-011 | Damage/Handling | Our delivery from the carrier arrived with the seal intact but 18 cartons of tablets are missing from the m... |
| THF-037 | Delay/Transit | While parked at the border queue, someone climbed onto the roof and cut the tarp. I found missing cartons o... |
| THF-044 | Damage/Handling | Customer reports a shipment arrived with the shrink wrap redone and 7 cartons of video game consoles replac... |
| THF-059 | Delay/Transit | Dispatch reports that the driver of unit 417 has stopped responding and the trailer door alert is active. L... |
| DLY-007 | Delay/Transit | The carrier informed us that our shipment of wedding decor is delayed until next week, but the event is thi... |
| THF-080 | Mispick/Order Error | The pallet of printer toner is not where the system says it is. Bay 9 is empty and the cycle count last wee... |
| THF-099 | Mispick/Order Error | Pallet scan shows 40 cartons of beauty products received, but the actual receiving count was 32, and the mi... |
| THF-102 | Damage/Handling | We got a complaint that 30 percent of the cartons of gaming controllers contain bricks wrapped in bubble wr... |
| THF-114 | Mispick/Order Error | The container yard at the port reported that one of our units was released to a driver whose ID does not ma... |
| CLD-098 | Damage/Handling | Customer reports that the cheese consignment smelled sour and the pallets were wet inside the wrap. The tra... |
| DLY-066 | Delay/Transit | The truck taking livestock vaccines to the farm cooperative is stuck behind an overturned lorry and the vac... |
| DLY-075 | Mispick/Order Error | The ambulance equipment shipment was left behind at the air hub because the manifest was wrong. The county ... |
| DLY-084 | Delay/Transit | The truck with copper cathodes for the smelter is two days late and the smelter has run out of feed materia... |
| DLY-090 | Delay/Transit | The truck delivering election material to polling stations is held up by a flooded road. The vote is tomorr... |
| DLY-092 | Delay/Transit | A pileup shut the expressway and I am stuck for hours with a truckload of cut roses for Valentine's Day at ... |
| DLY-094 | Delay/Transit | The truck delivering the replacement pump to the flooded mine broke its axle. Water is rising in the lower ... |

## 9. Notes and limitations

- True categories come from the `incident_id` prefix (THF/CLD/DLY/MSP/DMG); true urgency from the CSV column.
- The keyword weights were tuned while looking at this same dataset, so the rule accuracy above is optimistic. Measure on separately written reports before trusting it.
- The LLM only picks the category (one forward pass, answer letters A-E). Urgency is always decided by the rules.
- Low-confidence rule results are sent to the LLM; without a loaded model they are used anyway and flagged `rules-lowconf`.
