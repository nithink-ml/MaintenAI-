# CNC Milling Machine Diagnostic & Troubleshooting Manual (AI4I Standard)

## 1. Overview of Automated Failure Modes
Factory milling centers track tool wear, air/process temperature, rotational torque, and spindle RPM.
The platform classifies 5 deterministic physical failure modes:

1. **Tool Wear Failure (TWF)**: Excessive mechanical wear on carbide tool inserts exceeding 200 minutes cumulative cut time.
2. **Heat Dissipation Failure (HDF)**: Insufficient coolant dissipation where $\Delta T = T_{process} - T_{air} < 8.6\text{K}$ at spindle speeds $< 1380\text{ RPM}$.
3. **Power Failure (PWF)**: Motor mechanical power $P = \tau \times \omega$ falls outside the permissible 3500 W - 9000 W operating envelope.
4. **Overstrain Failure (OSF)**: Product of cumulative tool wear and cutting torque exceeds structural yield limits ($> 11000 \text{ min}\cdot\text{Nm}$).
5. **Random Failure (RNF)**: Uncorrelated electrical surge or external workpiece slip.

---

## 2. Corrective Maintenance Procedures

### Troubleshooting Heat Dissipation Failure (HDF)
1. **Coolant Pressure Verification**: Inspect high-pressure coolant pump delivery (minimum 20 bar).
2. **Chip Clearance**: Flush milling cavity of aluminum/steel swarf blocking heat transfer channels.
3. **Spindle Feed Rate Adjustment**: Reduce feed rate by 20% until $\Delta T$ stabilizes above 10K.
4. **Work Order Category**: `COOLANT_SYSTEM_FLUSH`, Task: `CLEAN_HEAT_EXCHANGER`.

### Troubleshooting Tool Wear Failure (TWF) & Overstrain (OSF)
1. **Index Tool Insert**: Rotate indexable carbide insert to fresh cutting edge.
2. **Torque Check**: Re-zero spindle torque sensor offset.
3. **Work Order Category**: `TOOLING_REPLACE`, Part: `INSERT-CARBIDE-CNMG120408`.
