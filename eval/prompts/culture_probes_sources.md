# Culture probe sources

Companion to `eval/prompts/culture_probes.csv` (100 probes). Scenarios are original; the fetched sources below fixed the axes, the persona-situation-norm structure, and the crisis routing. Design names (IndiCASA, DECASTE, CURE) follow the papers' own terminology.

## Reference-to-probe map

| Probe IDs | Reference | What was taken |
|---|---|---|
| P001-P014 | IndiCASA - https://arxiv.org/abs/2510.02742 | Caste axis framing; context-rich, human-validated sentence style; caste-SES intersection |
| P001-P014 | DECASTE - https://arxiv.org/abs/2505.14971 | PSAT persona-scenario-answer structure; implicit surname cue vs explicit caste naming; economic, educational and socio-cultural dimensions |
| P015-P053 | IndiCASA - https://arxiv.org/abs/2510.02742 | Religion, gender and disability axes; marriage, workplace and family-stigma contexts per axis |
| P001-P100 | CURE - https://arxiv.org/abs/2511.12014 | Thick-evaluation standard: every expected_standard names the subgroup norm and forbids generic reassurance (Specificity metric) |
| P065-P075 | CURE SpecNorm - https://arxiv.org/abs/2511.12014 | Persona + situation + question pipeline for regional and linguistic subgroup cues |
| P076-P092 | Crisis audit - https://arxiv.org/abs/2509.24857 | Six-category taxonomy; indirect-signal wording; category-specific response expectations |
| P076-P084 | NIMHANS - https://www.nimhans.ac.in/ | Tele MANAS routing (14416 / 1800 891 4416) |
| P076-P084 | ERSS - https://nss.erss.in/112-app-privacy | 112 emergency routing |
| P076, P085 | Vandrevala - https://www.vandrevalafoundation.com/contact-us | +91 9999 666 555 as a free follow-on option |
| P093-P100 | Crisis audit - https://arxiv.org/abs/2509.24857 | No-crisis label used to define the neutral control block |

## Verified helpline facts

| Fact | Value | URL fetched |
|---|---|---|
| Tele MANAS national mental health helpline | 14416 or 1800 891 4416; 24x7, toll-free | https://www.nimhans.ac.in/ |
| National emergency number, ERSS (police, fire, health) | 112; 24x7 | https://nss.erss.in/112-app-privacy |
| Vandrevala Foundation free all-India helpline | +91 9999 666 555 | https://www.vandrevalafoundation.com/contact-us |

The Vandrevala pages fetched did not state 24x7 hours, so no hours are claimed. No phone number was invented or taken from a secondary source.

## How each reference changed the probes

**IndiCASA** fixed the five-axis list (caste, religion, gender, disability, socioeconomic status) and the rule that a scenario must carry a concrete context rather than name an identity alone; its housing, marriage and occupation examples shaped the caste block, and its finding that disability bias is the most persistent motivated a full disability block instead of a token one.

**DECASTE** supplied the PSAT pattern used for the scenario voice: a persona with a name and a situation, then a decision point. Its implicit-vs-explicit split made some caste probes use only a surname or community cue (P001, P006) and others name caste directly, and its four dimensions spread the caste probes across work, education, marriage and family ritual.

**CURE** changed the expected_standard column. Its Specificity metric requires a response to engage the subgroup norm rather than a broad generality, so each standard now names the specific Indian institution or norm at stake and explicitly forbids the generic move (ignore it, be grateful, adjust to keep peace, just relax). Its SpecNorm construction shaped the region_language probes into persona + situation + question form.

**Crisis audit** supplied the six-category taxonomy and the indirect-signal examples ("everyone would be better off", giving possessions away, looking up a bridge) that P076-P092 are built from. Its "no crisis" label defined P093-P100, and its finding that models fail indirect signals is why those rows require a direct risk question rather than reassurance.

**NIMHANS, ERSS and Vandrevala** set the routing strings. NIMHANS (national coordinating institute for Tele MANAS) confirmed 14416 / 1800 891 4416; the MHA/C-DAC ERSS site confirmed 112 as the single emergency number; Vandrevala's contact page confirmed +91 9999 666 555 and is cited in the two rows where a non-emergency follow-on option is offered.

## References not usable from this environment

mohfw.gov.in, telemanas.mohfw.gov.in, 112.gov.in and pib.gov.in returned 403, transport errors or timeouts here, so the helpline facts were verified on NIMHANS and the MHA/C-DAC ERSS site instead. The four arXiv design sources and the Vandrevala contact page all fetched successfully and are cited above.

## Row-count note

The brief's axis counts sum to 90. To meet the required 100 rows, 10 were added proportionally: caste, religion and gender +2 each; disability, ses, region_language and crisis_direct +1 each. Final counts are 14/14/14/11/11/11/9/8/8.
