# Why AI investment looks like a bubble but may just be replacement spending

*Jonathan Rice, European Systemic Risk Board*

Combined capital spending by Amazon, Alphabet, Microsoft, and Meta hit $97 billion in the third quarter of 2025 alone, six times the quarterly run rate of 2018. Meta has guided $115–135 billion for 2026. Headlines frame this as a bubble. But a large share of that spending is not building new capacity. It is replacing hardware that has already become obsolete.

That distinction matters. AI accelerators — the specialised chips that train and serve large models — lose economic value far faster than conventional capital. GPU rental rates have fallen roughly 70 per cent in two years. Secondary-market prices for an H100 purchased eighteen months ago sit well below half the original cost. Taking rental-market and resale evidence together, economic depreciation of AI hardware runs at something like 30–40 per cent per year, an order of magnitude faster than the 6 per cent annual rate used for conventional equipment in standard macro models.

## Most AI spending is standing still

A simple identity makes the point concrete. In any period, gross investment equals replacement of worn-out capital plus net additions to the stock:

*Investment = Replacement + Net additions*

When annual depreciation is 6 per cent, replacement is a modest background flow. When it is 33 per cent, replacement *is* the headline number. In a steady state, a firm that needs to maintain a $100 billion compute stock at 33 per cent annual depreciation must spend $33 billion per year just to stand still. A 20 per cent cut to gross spending in that world wipes out net additions entirely, without the stock falling by more than a few per cent.

Figure 1 shows quarterly capital expenditure for the four largest cloud infrastructure firms from 2018 to mid-2025. The acceleration since 2023 is dramatic. But Figure 2 decomposes that spending into replacement and net additions using a perpetual-inventory method, assuming quarterly depreciation of 9.5 per cent — consistent with an economic life of roughly three years and with the paper's calibration (Rice 2026). The grey area — replacement spending — accounts for roughly half of the total even during the current boom, and would dominate rapidly if capex growth slowed. The implied compute stock, shown on the right axis, is far smoother than the investment flow that produces it.

**Figure 1.** Hyperscaler capital expenditure, 2018–2025
*Source: SEC EDGAR filings (Amazon, Alphabet, Microsoft, Meta Platforms). Quarterly "payments to acquire property, plant and equipment" from cash-flow statements.*

**Figure 2.** Decomposing AI investment: replacement versus net additions
*Source: Author's calculations using perpetual-inventory method applied to combined hyperscaler capex. High depreciation scenario assumes quarterly δ = 0.095 (≈33 per cent annual). Stock lines show sensitivity to alternative assumptions.*

## Two speeds of adjustment

This pattern — volatile investment, smooth stock — is exactly what a growth model with two types of capital predicts when one asset depreciates much faster than the other (Rice 2026). In that framework, the economy adjusts along two speeds. AI capacity reverts to trend with a half-life of about seven quarters, because the high rate of ongoing replacement gives the economy a built-in stabiliser: modest shifts in gross spending translate quickly into stock corrections. Conventional capital, depreciating at 6 per cent annually, adjusts over roughly a decade. The fast mode means that even large swings in measured AI investment can be consistent with an economy that is close to its optimal path.

This has a direct parallel. Communications-equipment investment surged from $62 billion to $135 billion annually between 1996 and 2000, then collapsed. The episode is often cited as pure speculative excess. But telecom switching equipment also depreciated rapidly, and much of the late-1990s investment was replacing earlier-generation hardware. The AI cycle has the same structural feature, with shorter hardware lives and correspondingly larger replacement shares.

## Why this matters for policy

Three implications follow from recognising that AI investment is dominated by replacement spending.

**Measurement.** National statistical offices currently classify AI accelerators alongside general-purpose computing equipment, using accounting depreciation schedules of five to six years. If true economic depreciation is closer to three years, measured investment is more volatile than underlying capacity, and productivity attribution is distorted. Germany's 2021 reform, which set computer-hardware useful life to one year for tax purposes, is closer to the mark.

**Financial stability.** Financial institutions have extended over $200 billion in data-centre-related debt. If the underlying hardware collateral depreciates at 33 per cent per year, it could enter negative equity within two to three years of financing. The Bank of England has already opened an inquiry into data-centre lending structures. The framework here suggests that lenders should distinguish clearly between the long-lived structure (the building) and the short-lived hardware inside it, because their depreciation profiles differ by an order of magnitude.

**Fiscal policy.** When replacement flows are large, small changes in the effective user cost of compute — whether through depreciation allowances, energy levies, or regulatory surcharges — move the steady-state stock substantially, and transitions are fast. In the calibrated model, a two-percentage-point change in quarterly depreciation shifts the optimal AI stock by around a fifth and alters welfare by 0.36 per cent in consumption-equivalent terms. Compute taxation is high-leverage and fast-acting, which makes it a powerful but also a risky policy instrument.

## A better way to read the capex numbers

The next time a hyperscaler announces a surge — or a cut — in AI spending, ask not what it implies for the *flow* of investment, but what it implies for the *stock* of compute relative to replacement needs. A spending cut that looks dramatic in flow terms may barely register in the stock, because depreciation would have done most of the work anyway. Conversely, the current boom is building a stock that will require enormous ongoing replacement spending just to maintain. The real question is not whether the spending will slow. It is whether it will slow to below replacement, and for how long.

---

**References**

Rice, J. (2026), "The Macroeconomics of AI Capacity: Insights from a Two-Asset Growth Model", ESRB Working Paper.

**Further reading on VoxEU:**

Acemoglu, D. and P. Restrepo (2019), "[Automation and new tasks: How technology displaces and reinstates labor](https://cepr.org/voxeu/columns/automation-and-new-tasks-how-technology-complements-labor)", VoxEU.org.

Korinek, A. and B. Lockwood (2026), "[The future of tax policy in the age of AI](https://www.brookings.edu/articles/the-future-of-tax-policy-a-public-finance-framework-for-the-age-of-ai/)", Brookings Institution.

*The views expressed are the author's and do not necessarily reflect those of the European Systemic Risk Board.*
