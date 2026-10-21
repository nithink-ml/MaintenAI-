# Industrial Bearing Maintenance & Degradation SOP (IMS Benchmark Series)

## 1. Overview of Bearing Health & Vibration Diagnostics
Rexnord ZA-2115 double-row bearing telemetry captures acceleration vibration at 20 kHz.
Key statistical and spectral health indicators:
- **Kurtosis**: Normal < 3.0. Incipient spalling / micro-cracking pushes kurtosis > 4.5.
- **Crest Factor**: Normal 3.0 - 4.5. Values > 6.0 indicate severe shock impacts.
- **Peak-to-Peak Vibration**: Warning threshold 4.0g; Critical shutdown threshold 8.5g.

---

## 2. Bearing Defect Frequencies & Anomaly Mitigation
### Ball Pass Frequency Outer Race (BPFO) & Outer Race Flaking
When IMS anomaly score exceeds **0.80** with elevated spectral energy in the 2.5 kHz - 5.0 kHz band:
1. **Acoustic Lubrication Analysis**: Apply synthetic lithium-complex grease (Grade NLGI 2) via automatic applicator (5 pumps @ 2.5 cc/stroke).
2. **Re-evaluate Vibration Spectrum**: If peak vibration does not decrease by at least 25% within 15 operating cycles, schedule physical bearing exchange.
3. **Emergency Action**: If vibration > 8.0g or anomaly score > 0.90, trigger emergency machine lock-out (LOTO) to avoid shaft seizure.
4. **Work Order Action**: Assign Work Order Category `BEARING_REPLACEMENT`, Part `BRG-REXNORD-ZA2115`.
