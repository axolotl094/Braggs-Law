Bragg's Law and Absorption of X-rays

Python script that regenerates all figures of the lab report from the raw PHYWE measure data (Cu-anode X-ray tube, 35 kV / 1 mA, GM counter).

Method (short)
Dead-time correction: N = N0 / (1 - τN0), assumed τ = 90 µs
Peak angles: Gaussian + linear background fits; λ = 2d sinθ / n with d(LiF) = 201.4 pm
KBr: d and a zero-angle offset fitted together over orders n = 1-4 of Kα and Kβ
Absorption: weighted fit of ln N vs thickness gives μ; μ/ρ with ρ(Al) = 2.70 g/cm³; log-log fit gives the λ exponent
Assumptions

τ = 90 µs (not given in the write-up); file lif_4mm is taken as the 0.04 mm absorber.

The physical models, mathematical derivations, and experimental data were formulated by me as part of the laboratory coursework. The accompanying Python analysis and plotting scripts were generated using Claude (Anthropic) and independently validated.