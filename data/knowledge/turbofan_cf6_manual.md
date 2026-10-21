# Turbofan Engine Maintenance Manual & Diagnostic Guide (CF6 Series)

## 1. System Overview & Sensor Architecture
The CF6 industrial turbofan turbine engine continuously monitors thermal and mechanical telemetry across operational cycles.
Critical telemetry channels include:
- **T24 / LPC Outlet Temperature**: Normal range 500K - 650K.
- **T30 / HPC Outlet Temperature**: Normal range 1400K - 1600K.
- **P30 / HPC Outlet Pressure**: Normal range 450 - 550 psia.
- **Nf / Core Fan Speed (RPM)**: Normal range 2200 - 2500 RPM.
- **Vibration & Core Accel**: Normal baseline < 2.5 mm/s RMS.

---

## 2. High Pressure Compressor (HPC) Degradation & Low RUL
### Symptoms & Telemetry Traces
When Remaining Useful Life (RUL) drops below **30 cycles**, the following mechanical degradation patterns manifest:
1. Progressive rise in LPC/HPC exhaust gas temperature differential.
2. Decreased HPC pressure ratio relative to fuel flow.
3. Elevated vibration amplitudes in bearing cavities 2 and 3.

### Standard Operating Procedure (SOP-TF-04)
1. **Immediate Inspection**: When RUL < 15 cycles, schedule immediate borescope inspection of the high-pressure turbine nozzle guide vanes (NGV) and HPC stage 5-8 blades.
2. **Derate Engine Power**: Restrict maximum continuous thrust to 85% until scheduled overhaul.
3. **Lubrication Filter Check**: Check the scavenge oil filter for metallic debris or magnetic chip detector (MCD) activation.
4. **Work Order Action**: Issue Work Order Category `EMERGENCY_OVERHAUL`, Part Reference `KIT-CF6-TURBINE-SEALS-V2`.

---

## 3. Sensor Drift & Calibration Protocol
### Fault Code: CODE-SENS-DRIFT
If anomaly score exceeds 0.70 while machine vibration remains nominal (< 1.5 mm/s), verify sensor transducer impedance:
- Calibrate P30 pressure transducer using digital manometer.
- Replace thermocouple harness if resistance exceeds 12.5 Ohms.
