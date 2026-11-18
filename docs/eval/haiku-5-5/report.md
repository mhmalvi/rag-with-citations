# Eval report: Claude Haiku 5.5 answers

Run at 2026-11-18 01:28 UTC - answer model `anthropic/claude-haiku-5.5`, judge `anthropic/claude-sonnet-5.5`, embedder `sentence-transformers/all-MiniLM-L6-v2`, reranker `cross-encoder/ms-marco-MiniLM-L6-v2`, top-k 6, LLM backend `api` via https://openrouter.ai/api.

33 questions (26 answerable, 7 unanswerable). 61 of 63 citations verified.

## Headline metrics

| Metric | Value |
|---|---|
| Answer correctness (LLM judge) | 94.2% |
| Citation faithfulness (code-checked) | 96.8% |
| Answers with every citation verified | 96.0% |
| Abstention on unanswerable questions | 100.0% |
| False abstention on answerable questions | 3.8% |
| Mean latency per question | 4.7 s |
| Tokens (input / output) | 78,189 / 6,722 |

## Retrieval (answerable questions, gold evidence in top k)

| Retriever | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| BM25 only | 73.1% | 88.5% | 100.0% | 0.813 |
| Dense only (MiniLM) | 61.5% | 80.8% | 92.3% | 0.730 |
| Hybrid (BM25 + dense, RRF) | 76.9% | 92.3% | 100.0% | 0.849 |
| Hybrid + cross-encoder rerank (default) | 80.8% | 96.2% | 96.2% | 0.872 |

## Per question

| id | expected | hit rank | outcome | verified cites | verdict |
|---|---|---|---|---|---|
| ssp-01 | answerable | 1 | answered | 2/2 | correct |
| ssp-02 | answerable | 1 | answered | 2/2 | correct |
| ssp-03 | answerable | 1 | answered | 1/1 | correct |
| hol-01 | answerable | 1 | answered | 1/1 | correct |
| hol-02 | answerable | 1 | answered | 3/3 | correct |
| hol-03 | answerable | 1 | answered | 3/3 | correct |
| hol-04 | answerable | 3 | answered | 4/4 | correct |
| hol-05 | answerable | 1 | answered | 1/1 | correct |
| hol-06 | answerable | 1 | answered | 2/2 | correct |
| flex-01 | answerable | 2 | answered | 2/2 | correct |
| flex-02 | answerable | 1 | answered | 2/2 | correct |
| flex-03 | answerable | 1 | answered | 10/10 | correct |
| red-01 | answerable | 3 | answered | 1/1 | correct |
| red-02 | answerable | 1 | answered | 1/1 | correct |
| red-03 | answerable | 1 | answered | 4/4 | correct |
| hours-01 | answerable | 1 | answered | 2/2 | correct |
| hours-02 | answerable | 1 | answered | 2/2 | correct |
| csf-01 | answerable | 2 | answered | 1/1 | correct |
| csf-02 | answerable | 1 | answered | 2/2 | correct |
| csf-03 | answerable | miss | abstained | - | incorrect |
| csf-04 | answerable | 1 | answered | 3/3 | correct |
| ir-01 | answerable | 1 | answered | 1/3 | partially_correct |
| ir-02 | answerable | 1 | answered | 3/3 | correct |
| ir-03 | answerable | 1 | answered | 4/4 | correct |
| style-01 | answerable | 1 | answered | 2/2 | correct |
| style-02 | answerable | 1 | answered | 2/2 | correct |
| na-01 | unanswerable | - | abstained | - | correct abstention |
| na-02 | unanswerable | - | abstained | - | correct abstention |
| na-03 | unanswerable | - | abstained | - | correct abstention |
| na-04 | unanswerable | - | abstained | - | correct abstention |
| na-05 | unanswerable | - | abstained | - | correct abstention |
| na-06 | unanswerable | - | abstained | - | correct abstention |
| na-07 | unanswerable | - | abstained | - | correct abstention |

## Answers

### ssp-01: How much Statutory Sick Pay can an employee get each week, and for how long is it paid?

An employee can get up to £123.25 per week in Statutory Sick Pay, or 80% of normal weekly earnings if that is lower [1][2]. It is paid for up to 28 weeks [1].

- [1] GOV.UK: Statutory Sick Pay (SSP), 2. What you'll get (verified): "you can get £123.25 a week Statutory Sick Pay (SSP) or 80% of your normal weekly earnings - whichever is lower"
- [2] GOV.UK: Statutory Sick Pay (SSP), 1. Overview (verified): "It’s paid by your employer for up to 28 weeks."

Judge: **correct** - The candidate gives the same weekly cap of £123.25, the 80% of normal weekly earnings alternative if lower, and the 28-week maximum duration as the reference. It omits only that the employer pays it, which is a minor detail, and nothing contradicts the reference.

### ssp-02: I've been off ill for more than a week. What proof does my employer need from me?

Your employer can ask for a fit note if you are off work for more than 7 days in a row, including non-working days [1]. The note can be from a GP or hospital doctor, registered nurse, occupational therapist, pharmacist, or physiotherapist, and it can be printed or digital [1]. The documents provided do not say whether your employer needs anything beyond this, such as a self-certificate for the first days of absence.

- [1] GOV.UK: Statutory Sick Pay (SSP), 3. Eligibility > Fit notes and asking for proof (verified): "You must give your employer a fit note if you’re off work for more than 7 days in a row (including non-working days)."
- [2] GOV.UK: Statutory Sick Pay (SSP), 3. Eligibility > Fit notes and asking for proof (verified): "The note can be printed or digital."

Judge: **correct** - The candidate gives the 7-day threshold including non-working days and the full list of professionals who can issue a fit note, matching the reference. It softens 'must give' to 'can ask for' and omits the alternative evidence if the employer agrees, but adds nothing that contradicts the reference.

### ssp-03: If my sick pay is expected to end before my sickness does, when must my employer send me form SSP1?

If your SSP is expected to end before your sickness does, your employer must send you form SSP1 on or before the beginning of the 23rd week. [1]

- [1] GOV.UK: Statutory Sick Pay (SSP), 3. Eligibility > If you’re not eligible or your SSP ends (verified): "on or before the beginning of the 23rd week, if your SSP is expected to end before your sickness does"

Judge: **correct** - The candidate states the form must be sent on or before the beginning of the 23rd week, matching the reference; it omits the word 'of SSP' but the context makes it clear that it refers to the SSP weeks.

### hol-01: What is the minimum paid annual leave for someone who works a 5-day week?

Someone who works a 5-day week must receive at least 28 days' paid annual leave a year, which is the equivalent of 5.6 weeks of holiday [1].

- [1] GOV.UK: Holiday entitlement, 1. Entitlement > Statutory annual leave entitlement (verified): "Most workers who work a 5-day week must receive at least 28 days’ paid annual leave a year."

Judge: **correct** - The candidate states the minimum is 28 days' paid annual leave, equivalent to 5.6 weeks, which matches the reference answer exactly with no contradictions.

### hol-02: How many days of untaken holiday can a worker with 28 days' leave carry over into the next year?

A worker entitled to 28 days' leave can carry over a maximum of 8 days into the next year [1]. If they could not take leave because they were on family-related leave, they can carry over all untaken leave [2], and if they were sick, the amount depends on whether they work regular hours [2].

- [1] GOV.UK: Holiday entitlement, 2. Calculate leave entitlement > Carrying over leave (verified): "If a worker gets 28 days’ leave, they can carry over a maximum of 8 days."
- [2] GOV.UK: Holiday entitlement, 2. Calculate leave entitlement > Carrying over leave > Workers on parental or sick leave (verified): "they can carry over their untaken leave into next year."
- [3] GOV.UK: Holiday entitlement, 2. Calculate leave entitlement > Carrying over leave > Workers on parental or sick leave (verified): "up to 20 of their 28 days’ leave entitlement if they work regular hours for the whole year"

Judge: **correct** - The candidate gives the key fact of a maximum of 8 days carried over and also mentions the special cases of family leave and sickness, consistent with the reference. It omits the contract-allows-more caveat, but that is a minor part, and its extra details do not contradict the reference.

### hol-03: How much notice should a worker give if they want to take one day off as holiday?

The general notice period is at least twice the amount of leave wanted, plus 1 day, so a worker wanting one day's leave should give 3 days' notice [1]. This is the general rule, and if the employment contract states a different notice period, the contract applies [1].

- [1] GOV.UK: Holiday entitlement, 4. Booking time off (verified): "The general notice period for taking leave is at least twice as long as the amount of leave a worker wants to take, plus 1 day."
- [2] GOV.UK: Holiday entitlement, 4. Booking time off (verified): "For example, a worker would give 3 days’ notice for 1 day’s leave."
- [3] GOV.UK: Holiday entitlement, 4. Booking time off (verified): "If the contract says something different about the notice a worker or employer should give, what’s in the contract will apply."

Judge: **correct** - The candidate gives the same rule as the reference (twice the leave plus 1 day, so 3 days' notice for 1 day off). The added note that a contract may specify a different period does not contradict the reference.

### hol-04: How is holiday entitlement worked out for irregular-hours workers?

Irregular-hours workers must have leave calculated using a specific accrual system, not a fixed number of days or hours [1][2]. Their entitlement is 12.07% of the hours they worked in a pay period, rounded up to the nearest hour if the result is 0.5 of an hour or more [3]. The documents do not say how this applies to leave years beginning on or before 31 March 2024, beyond saying that employers do not need to use this system until their leave year renews [3].

- [1] GOV.UK: Holiday entitlement, 2. Calculate leave entitlement (verified): "For irregular-hours and part-year workers, employers must use a specific accrual system."
- [2] GOV.UK: Holiday entitlement, 1. Entitlement > Statutory annual leave entitlement > Working irregular hours or for part of the year (verified): "They will build up (‘accrue’) leave depending on the hours they’ve already worked, rather than getting a fixed number of days or hours."
- [3] GOV.UK: Holiday entitlement, 2. Calculate leave entitlement > Accrual system > Irregular-hours or part-year workers (verified): "Their entitlement will be 12.07% of the hours they worked in a pay period. It must be rounded up to the nearest hour if the entitlement is 0.5 of an hour or more."
- [4] GOV.UK: Holiday entitlement, 2. Calculate leave entitlement > Accrual system > Irregular-hours or part-year workers (verified): "If their leave year began on or before 31 March 2024, employers do not need to use this system until their leave year renews."

Judge: **correct** - The candidate states the key facts of the reference: 12.07% of hours worked in a pay period, rounded up to the nearest hour if 0.5 of an hour or more. The extra detail about leave years and transition does not contradict the reference.

### hol-05: From April 2026, how long do employers have to keep records of annual leave and holiday pay?

From 6 April 2026, employers must keep detailed records of annual leave and holiday pay for a minimum of 6 years from the date they were made [1].

- [1] GOV.UK: Holiday entitlement, 3. Holiday pay > Keeping records (verified): "From 6 April 2026, employers must keep detailed records of annual leave and holiday pay for a minimum of 6 years from the date they were made."

Judge: **correct** - The candidate gives the same start date (6 April 2026) and the same minimum retention period (6 years from the date the records were made) as the reference, with no contradictions.

### hol-06: Can my employer count bank holidays as part of my statutory leave?

Yes. An employer can choose to include bank holidays as part of a worker's statutory annual leave [1]. However, bank or public holidays do not have to be given as paid leave [1].

- [1] GOV.UK: Holiday entitlement, 1. Entitlement > Statutory annual leave entitlement > Bank holidays (verified): "An employer can choose to include bank holidays as part of a worker’s statutory annual leave."
- [2] GOV.UK: Holiday entitlement, 1. Entitlement > Statutory annual leave entitlement > Bank holidays (verified): "Bank or public holidays do not have to be given as paid leave."

Judge: **correct** - The candidate says yes, an employer can include bank holidays in statutory annual leave, and that bank holidays do not have to be given as paid leave. Both key facts from the reference are present and nothing contradicts it.

### flex-01: From when can an employee ask for flexible working, and how many requests can they make in a year?

An employee can apply for flexible working from their first day in a job [1]. They can make 2 applications in any 12-month period [2].

- [1] GOV.UK: Flexible working, 3. Applying for flexible working (verified): "Employees can apply for flexible working from their first day in a job."
- [2] GOV.UK: Flexible working, 3. Applying for flexible working (verified): "Employees can make 2 applications for flexible working in any 12-month period."

Judge: **correct** - The candidate states that flexible working can be requested from the first day of a job and that 2 applications are allowed in any 12-month period, matching both key facts in the reference answer with no contradictions.

### flex-02: How long does an employer have to make a decision on a flexible working request?

An employer must make a decision within 2 months of the request, or longer if agreed with the employee [1]. The employer should discuss the request with the employee and make the decision within this period [2].

- [1] GOV.UK: Flexible working, 4. After the application (verified): "They must make a decision within 2 months of the request (or longer if agreed with the employee)."
- [2] GOV.UK: Flexible working, 3. Applying for flexible working (verified): "makes a decision on it within 2 months - or longer if agreed with the employee."

Judge: **correct** - The candidate states the 2-month decision period and the exception for a longer period if agreed with the employee, matching the reference; the extra detail about discussing the request does not contradict it.

### flex-03: What business reasons can an employer give for turning down a flexible working request?

Employers can refuse a flexible working request for reasons including extra costs that will damage the business, the work cannot be reorganised among other staff, people cannot be recruited to do the work, and flexible working will affect quality or performance [1]. Other listed reasons are that the business will not be able to meet customer demand, there is a lack of work during the proposed working times, and the business is planning changes to the workforce [1]. The employer must also have discussed the request with the employee before refusing it and must tell them of the refusal [1].

- [1] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "Employers can refuse an application for any of the following reasons:"
- [2] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "extra costs that will damage the business"
- [3] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "the work cannot be reorganised among other staff"
- [4] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "people cannot be recruited to do the work"
- [5] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "flexible working will affect quality"
- [6] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "flexible working will affect performance"
- [7] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "the business will not be able to meet customer demand"
- [8] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "there’s a lack of work to do during the proposed working times"
- [9] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "the business is planning changes to the workforce"
- [10] GOV.UK: Flexible working, 4. After the application > Refusing the application > Reasons for refusing (verified): "Before refusing it, they must have discussed the employee’s request with them."

Judge: **correct** - The candidate lists all eight business reasons from the reference (extra costs, reorganising work, recruitment, quality, performance, customer demand, lack of work, planned workforce changes). The extra note about discussing the request and notifying refusal does not contradict the reference.

### red-01: How long do I need to have worked for my employer to qualify for statutory redundancy pay?

You'll normally be entitled to statutory redundancy pay if you're an employee and have been working for your current employer for 2 years or more [1].

- [1] GOV.UK: Redundancy: your rights, 3. Statutory redundancy pay (verified): "You’ll normally be entitled to statutory redundancy pay if you’re an employee and you’ve been working for your current employer for 2 years or more."

Judge: **correct** - The candidate states that you normally need to be an employee with 2 years or more of service with your current employer, which matches the reference answer exactly on the key fact.

### red-02: What is the maximum statutory redundancy pay for someone made redundant on or after 6 April 2026?

For someone made redundant on or after 6 April 2026, the maximum statutory redundancy pay is £22,530, based on a weekly pay cap of £751. [1]

- [1] GOV.UK: Redundancy: your rights, 3. Statutory redundancy pay (verified): "If you were made redundant on or after 6 April 2026, your weekly pay is capped at £751 and the maximum statutory redundancy pay you can get is £22,530."

Judge: **correct** - The candidate gives the maximum statutory redundancy pay as £22,530 with a weekly pay cap of £751, which matches both facts in the reference answer exactly.

### red-03: How is statutory redundancy pay calculated based on age?

Statutory redundancy pay is calculated per full year of service by age band: half a week's pay for each full year under 22, one week's pay for each full year from 22 up to under 41, and one and a half week's pay for each full year at 41 or older. Length of service is capped at 20 years.

- [1] GOV.UK: Redundancy: your rights, 3. Statutory redundancy pay (verified): "half a week’s pay for each full year you were under 22"
- [2] GOV.UK: Redundancy: your rights, 3. Statutory redundancy pay (verified): "one week’s pay for each full year you were 22 or older, but under 41"
- [3] GOV.UK: Redundancy: your rights, 3. Statutory redundancy pay (verified): "one and half week’s pay for each full year you were 41 or older"
- [4] GOV.UK: Redundancy: your rights, 3. Statutory redundancy pay (verified): "Length of service is capped at 20 years."

Judge: **correct** - The candidate gives the same age bands (0.5 week under 22, 1 week for 22 to 40, 1.5 weeks for 41+) and the same 20-year service cap as the reference. "From 22 up to under 41" is equivalent to 22 to 40, so nothing contradicts the reference.

### hours-01: Over what period are average weekly working hours normally calculated for the 48-hour limit?

Average weekly working hours are normally calculated over a 'reference' period of 17 weeks [1]. This period can be extended to a maximum of 52 weeks if agreed, for example by a trade union [1].

- [1] GOV.UK: Maximum weekly working hours, 2. Calculating your working hours (verified): "Average working hours are calculated over a ‘reference’ period, normally 17 weeks."
- [2] GOV.UK: Maximum weekly working hours, 2. Calculating your working hours (verified): "The 17 week period can be extended to a maximum of 52 weeks if this is agreed, for example, by a trade union."

Judge: **correct** - The candidate states the normal 17-week reference period and the extension to a maximum of 52 weeks if agreed, for example by a trade union, which matches the reference answer on every key fact.

### hours-02: I signed an opt-out from the 48-hour week. How much notice do I have to give to cancel it?

You must give your employer at least 7 days' notice to cancel the opt-out [1]. Depending on what your agreement says, you may have to give up to 3 months' notice [1].

- [1] GOV.UK: Maximum weekly working hours, 3. Opting out of the 48 hour week > Cancelling an opt-out agreement (verified): "You must give your employer at least 7 days’ notice."
- [2] GOV.UK: Maximum weekly working hours, 3. Opting out of the 48 hour week > Cancelling an opt-out agreement (verified): "You may have to give your employer up to 3 months’ notice depending on what’s in your agreement."

Judge: **correct** - The candidate states the minimum of 7 days' notice and that the agreement may require up to 3 months, matching both key facts in the reference answer with no contradictions.

### csf-01: What are the six Functions of the NIST CSF 2.0 Core?

The six Functions of the NIST CSF 2.0 Core are GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER [1][2].

- [1] NIST Cybersecurity Framework (CSF) 2.0, page 8 (verified): "The CSF Core Functions — GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER — organize"

Judge: **correct** - The candidate lists all six functions (Govern, Identify, Protect, Detect, Respond, Recover), matching the reference exactly with no contradictions.

### csf-02: Name the four CSF Tiers.

The four CSF Tiers are Partial (Tier 1), Risk Informed (Tier 2), Repeatable (Tier 3), and Adaptive (Tier 4) [1][2].

- [1] NIST Cybersecurity Framework (CSF) 2.0, page 12 (verified): "Partial (Tier
1), Risk Informed (Tier 2), Repeatable (Tier 3), and Adaptive (Tier 4)"
- [2] NIST Cybersecurity Framework (CSF) 2.0, page 31 (verified): "There are four Tiers: Partial (Tier 1), Risk Informed (Tier 2), Repeatable (Tier 3), and Adaptive (Tier 4)."

Judge: **correct** - The candidate lists all four tiers (Partial, Risk Informed, Repeatable, Adaptive) with the correct tier numbers, matching the reference exactly.

### csf-03: What was the Cybersecurity Framework called before version 2.0?

The documents provided do not say what the Cybersecurity Framework was called before version 2.0. They describe CSF 2.0 and its components, including Core, Organizational Profiles, and Tiers, and mention that it builds on previous versions.


Judge: **incorrect** - System abstained on an answerable question.

### csf-04: What is the difference between a Current Profile and a Target Profile in the CSF?

A Current Profile specifies the Core outcomes an organization is currently achieving (or attempting to achieve) and characterizes how or to what extent each outcome is being achieved [1]. A Target Profile specifies the desired outcomes the organization has selected and prioritized for achieving its cybersecurity risk management objectives [2]. The Target Profile also considers anticipated changes to the organization's cybersecurity posture, such as new requirements, new technology adoption, and threat intelligence trends [1].

- [1] NIST Cybersecurity Framework (CSF) 2.0, page 11 (verified): "A Cur rent Profile specifies the Core outcomes that an organization is currently achieving (or attempting to achieve) and characterizes how or to what extent each outcome is being achieved."
- [2] NIST Cybersecurity Framework (CSF) 2.0, page 31 (verified): "A part of an Organizational Profile that specifies the desired Core outcomes that an organization has selected and prioritized for achieving its cybersecurity risk management objectives."
- [3] NIST Cybersecurity Framework (CSF) 2.0, page 11 (verified): "A Target Profile considers anticipated changes to the organization’s cybersecurity posture, such as new requirements, new technology adoption, and threat intelligence trends."

Judge: **correct** - The candidate states both key definitions matching the reference: Current Profile as outcomes currently achieved or attempted and how well, and Target Profile as desired outcomes selected and prioritized for risk management objectives. The extra detail about anticipated changes is consistent with CSF 2.0 and does not contradict the reference.

### ir-01: Which CSF 2.0 Functions map to the 'Containment, Eradication & Recovery' phase of the previous incident response life cycle?

According to Table 1, the Containment, Eradication & Recovery phase maps to the Respond and Recover CSF 2.0 Functions [1][2]. The table text is laid out without clear row boundaries, so the Identify (Improvement Category) entry near this phase is not clearly assigned to it, and the passages do not settle that point.

- [1] NIST SP 800-61r3 Incident Response Recommendations, page 14 (verified): "Containment, Eradication & Recovery"
- [2] NIST SP 800-61r3 Incident Response Recommendations, page 14 (NOT FOUND): "Respond"
- [3] NIST SP 800-61r3 Incident Response Recommendations, page 14 (NOT FOUND): "Recover"

Judge: **partially_correct** - The candidate correctly names Respond and Recover but does not confirm the Identify Function's Improvement Category as part of the mapping, saying only that it is unclear, so it misses an important part of the reference answer.

### ir-02: According to NIST SP 800-61r3, what do incident handlers do?

According to the passages, incident handlers verify that an incident has occurred, collect and analyze data and evidence, prioritize incident response activities, and act appropriately to limit damage, find root causes, and restore operations [1]. They also often provide input to others on mitigating cybersecurity issues and improving resiliency [1].

- [1] NIST SP 800-61r3 Incident Response Recommendations, page 15 (verified): "Incident handlers verify that an incident has occurred, collect and analyze data and evidence, prioritize incident response activities, and act appropriately"
- [2] NIST SP 800-61r3 Incident Response Recommendations, page 15 (verified): "to limit damage, find root causes, and restore operations."
- [3] NIST SP 800-61r3 Incident Response Recommendations, page 15 (verified): "Incident handlers also often provide input to others on mitigating cybersecurity issues and improving resiliency."

Judge: **correct** - The candidate lists all the reference's key duties: verifying an incident, collecting and analyzing data and evidence, prioritizing response activities, and acting to limit damage, find root causes and restore operations. It also covers advising others on mitigation and resiliency, with no contradictions.

### ir-03: What key elements do most incident response policies include?

Most incident response policies include the same key elements, which include a statement of management commitment, the purpose and objectives of the policy, and the scope of the policy [1]. Other listed elements include definitions of events, cybersecurity incidents, and related terms, roles and responsibilities, guidelines for prioritizing incidents, and performance measures [2].

- [1] NIST SP 800-61r3 Incident Response Recommendations, page 16 (verified): "most incident response policies include the
same key elements:"
- [2] NIST SP 800-61r3 Incident Response Recommendations, page 16 (verified): "Statement of management commitment"
- [3] NIST SP 800-61r3 Incident Response Recommendations, page 16 (verified): "Purpose and objectives of the policy"
- [4] NIST SP 800-61r3 Incident Response Recommendations, page 16 (verified): "Scope of the policy (i.e., to whom and what it applies and under what circumstances)"

Judge: **correct** - The candidate covers management commitment, purpose and objectives, scope, definitions, roles and responsibilities, prioritization guidelines, and performance measures. It omits only the minor details of 'authorities' and 'estimating severity and recovery', which do not change the main point.

### style-01: In body copy, when should numbers be spelled out according to the 18F content guide?

In body copy, the guide prefers to spell out numbers one through nine and use numerals for 10 and greater. This also applies to ordinal numbers, so first through ninth are spelled out and 10th or greater use numerals.

- [1] 18F Content Guide: Numbers and percentages, Numbers, percentages, and dates > Numbers (verified): "In body copy, we prefer to spell out numbers one through nine, and use numerals for numbers 10 and greater."
- [2] 18F Content Guide: Numbers and percentages, Numbers, percentages, and dates > Numbers (verified): "Spell out first to ninth, and capture 10th or greater with numerals."

Judge: **correct** - The candidate states the same rule as the reference: spell out one through nine and use numerals for 10 and above, including ordinals (first through ninth spelled out, 10th and up as numerals). It contradicts nothing in the reference.

### style-02: Does 18F want a comma before 'and' at the end of a list?

Yes. 18F uses the serial comma, so in a list of three or more items a comma goes before the conjunction [1].

- [1] 18F Content Guide: Punctuation, Punctuation > Commas (verified): "We use the serial comma (sometimes called the Oxford comma)."
- [2] 18F Content Guide: Punctuation, Punctuation > Commas (verified): "In a list of three or more, include a comma before the conjunction."

Judge: **correct** - The candidate says yes, 18F uses the serial comma, with a comma before the conjunction in lists of three or more items. This matches the reference answer's key facts and adds nothing that contradicts it.

### na-01: What is the weekly rate of Statutory Maternity Pay?

The documents provided do not cover the weekly rate of Statutory Maternity Pay. They cover Statutory Sick Pay, holiday entitlement, and statutory redundancy pay, and mention Statutory Maternity Pay only as an exclusion from SSP eligibility.


### na-02: What is the National Minimum Wage for workers aged 21 and over?

The documents provided do not cover the National Minimum Wage rate for workers aged 21 and over. They cover statutory redundancy pay, maximum weekly working hours, holiday pay, flexible working, and Statutory Sick Pay.


### na-03: Within how many hours must an organization report a cybersecurity incident to CISA?

The documents provided do not cover a deadline for reporting cybersecurity incidents to CISA. They cover incident response life cycle functions, third-party incident reporting mechanisms, incident categorization, and containment duration examples such as emergency workarounds that must be removed within hours.


### na-04: What minimum password length does NIST recommend?

The documents provided do not cover a minimum password length. They cover NIST SP 800-61r3 authority and scope, incident response profile controls such as identity and access management, reference citations, and insider risk and non-disclosure agreements.


### na-05: Which font does the 18F content guide recommend for body text?

The documents provided do not cover which font the 18F content guide recommends for body text. They cover topics such as putting important information first, avoiding slashes, using HTML for web content, keeping sentences short, and sentence-case headings.


### na-06: How many days of paid bereavement leave must an employer give?

The documents provided do not cover bereavement leave. They cover statutory annual holiday entitlement, booking time off, holiday pay, carrying over leave, statutory sick pay fit notes, and payment in lieu of untaken leave.


### na-07: How much is the fine for an employer that does not keep holiday records?

The documents provided do not state the amount of any fine. They say only that employers from 6 April 2026 must keep holiday records for at least 6 years and that they may have to pay a fine if they cannot show they keep those records.

