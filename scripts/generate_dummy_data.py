"""Generate dummy clinical data for testing."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

# Data directory
DATA_DIR = Path("data/dummy")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class ClinicalDataGenerator:
    """Generate dummy clinical trial data."""

    def __init__(self, study_id: str = "STUDY001", n_subjects: int = 100):
        """Initialize generator."""
        self.study_id = study_id
        self.n_subjects = n_subjects
        self.random = random.Random(42)
        self.np_random = np.random.RandomState(42)

        # Generate subject IDs
        self.subject_ids = [f"{study_id}{i:03d}" for i in range(1, n_subjects + 1)]

    def generate_demographics(self) -> List[Dict]:
        """Generate DM (Demographics) data."""
        data = []

        for subject_id in self.subject_ids:
            age = self.np_random.randint(18, 85)
            sex = self.random.choice(["M", "F"])
            race = self.random.choice(["WHITE", "BLACK", "ASIAN", "HISPANIC"])
            ethnicity = self.random.choice(
                ["HISPANIC OR LATINO", "NOT HISPANIC OR LATINO"]
            )

            data.append(
                {
                    "STUDYID": self.study_id,
                    "USUBJID": subject_id,
                    "SUBJID": subject_id[-3:],
                    "SITEID": f"SITE{self.random.randint(1, 10):02d}",
                    "BRTHDTC": (datetime.now() - timedelta(days=age * 365)).strftime(
                        "%Y-%m-%d"
                    ),
                    "AGE": age,
                    "AGEU": "YEARS",
                    "SEX": sex,
                    "RACE": race,
                    "ETHNIC": ethnicity,
                    "COUNTRY": "USA",
                }
            )

        return data

    def generate_vitals(self) -> List[Dict]:
        """Generate VS (Vitals Signs) data."""
        data = []

        for subject_id in self.subject_ids:
            # Generate 3-5 vital signs measurements per subject
            n_measurements = self.random.randint(3, 5)

            for i in range(n_measurements):
                visit_day = self.random.choice([1, 8, 15, 29, 57, 85])

                data.append(
                    {
                        "STUDYID": self.study_id,
                        "USUBJID": subject_id,
                        "VSSEQ": i + 1,
                        "VSTESTCD": self.random.choice(
                            [
                                "SYSBP",
                                "DIABP",
                                "PULSE",
                                "TEMP",
                                "RESP",
                                "WEIGHT",
                                "HEIGHT",
                            ]
                        ),
                        "VSSTRESN": self._generate_vital_value(),
                        "VSORRES": str(self._generate_vital_value()),
                        "VSORRESU": self._get_vital_unit(),
                        "VISITNUM": self.random.choice([1, 2, 3, 4, 5, 6]),
                        "VISIT": self.random.choice(
                            [
                                "SCREENING",
                                "BASELINE",
                                "WEEK 2",
                                "WEEK 4",
                                "WEEK 8",
                                "WEEK 12",
                            ]
                        ),
                        "VSDTC": (datetime.now() + timedelta(days=visit_day)).strftime(
                            "%Y-%m-%d"
                        ),
                    }
                )

        return data

    def _generate_vital_value(self) -> float:
        """Generate realistic vital sign value."""
        vitals = {
            "SYSBP": (110, 140),
            "DIABP": (70, 90),
            "PULSE": (60, 100),
            "TEMP": (36.5, 37.5),
            "RESP": (12, 20),
            "WEIGHT": (50, 100),
            "HEIGHT": (150, 190),
        }

        # Random vital type
        vital_type = self.random.choice(list(vitals.keys()))
        min_val, max_val = vitals[vital_type]

        return round(self.np_random.uniform(min_val, max_val), 1)

    def _get_vital_unit(self) -> str:
        """Get unit for vital sign."""
        units = {
            "SYSBP": "mmHg",
            "DIABP": "mmHg",
            "PULSE": "bpm",
            "TEMP": "C",
            "RESP": "bpm",
            "WEIGHT": "kg",
            "HEIGHT": "cm",
        }
        return units.get(self._generate_vital_value(), "unit")

    def generate_lab_results(self) -> List[Dict]:
        """Generate LB (Laboratory) data."""
        data = []

        lab_tests = [
            ("ALT", "U/L", (10, 50)),
            ("AST", "U/L", (10, 40)),
            ("CREAT", "umol/L", (50, 120)),
            ("GLUC", "mmol/L", (4.0, 7.0)),
            ("HGB", "g/L", (120, 160)),
            ("PLAT", "10^9/L", (150, 400)),
            ("WBC", "10^9/L", (4.0, 11.0)),
            ("SODIUM", "mmol/L", (135, 145)),
            ("POTASS", "mmol/L", (3.5, 5.0)),
            ("CALCIUM", "mmol/L", (2.1, 2.6)),
        ]

        for subject_id in self.subject_ids:
            # Generate lab results for each test
            for test_code, unit, (min_val, max_val) in lab_tests:
                # Generate 2-4 measurements per test
                n_measurements = self.random.randint(2, 4)

                for i in range(n_measurements):
                    visit_day = self.random.choice([1, 15, 29, 57, 85])

                    data.append(
                        {
                            "STUDYID": self.study_id,
                            "USUBJID": subject_id,
                            "LBSEQ": i + 1,
                            "LBTESTCD": test_code,
                            "LBTEST": f"Laboratory Test {test_code}",
                            "LBCAT": "CHEMISTRY",
                            "LBORRES": str(
                                round(self.np_random.uniform(min_val, max_val), 2)
                            ),
                            "LBORRESU": unit,
                            "LBSTRESC": str(
                                round(self.np_random.uniform(min_val, max_val), 2)
                            ),
                            "LBSTRESN": round(
                                self.np_random.uniform(min_val, max_val), 2
                            ),
                            "LBSTRESU": unit,
                            "VISITNUM": self.random.choice([1, 2, 3, 4, 5, 6]),
                            "VISIT": self.random.choice(
                                [
                                    "SCREENING",
                                    "BASELINE",
                                    "WEEK 2",
                                    "WEEK 4",
                                    "WEEK 8",
                                    "WEEK 12",
                                ]
                            ),
                            "LBDTC": (
                                datetime.now() + timedelta(days=visit_day)
                            ).strftime("%Y-%m-%d"),
                        }
                    )

        return data

    def generate_adverse_events(self) -> List[Dict]:
        """Generate AE (Adverse Events) data."""
        data = []

        ae_terms = [
            "HEADACHE",
            "NAUSEA",
            "DIZZINESS",
            "FATIGUE",
            "DIARRHEA",
            "RASH",
            "INSOMNIA",
            "ANXIETY",
            "COUGH",
            "FEVER",
        ]

        for subject_id in self.subject_ids:
            # 30% of subjects have adverse events
            if self.random.random() < 0.3:
                n_events = self.random.randint(1, 3)

                for i in range(n_events):
                    ae_term = self.random.choice(ae_terms)
                    severity = self.random.choice(["MILD", "MODERATE", "SEVERE"])
                    relationship = self.random.choice(
                        ["RELATED", "NOT RELATED", "POSSIBLY RELATED"]
                    )

                    data.append(
                        {
                            "STUDYID": self.study_id,
                            "USUBJID": subject_id,
                            "AESEQ": i + 1,
                            "AETERM": ae_term,
                            "AELLT": ae_term,
                            "AEDECOD": ae_term,
                            "AEBODSYS": self._get_body_system(ae_term),
                            "AESEV": severity,
                            "AEREL": relationship,
                            "AEACN": self.random.choice(["Y", "N"]),
                            "AEOUT": self.random.choice(
                                ["RECOVERED", "RECOVERING", "NOT RECOVERED"]
                            ),
                            "VISITNUM": self.random.choice([1, 2, 3, 4, 5, 6]),
                            "VISIT": self.random.choice(
                                [
                                    "SCREENING",
                                    "BASELINE",
                                    "WEEK 2",
                                    "WEEK 4",
                                    "WEEK 8",
                                    "WEEK 12",
                                ]
                            ),
                            "AESTDTC": (
                                datetime.now()
                                + timedelta(days=self.random.randint(1, 85))
                            ).strftime("%Y-%m-%d"),
                            "AEENDTC": (
                                datetime.now()
                                + timedelta(days=self.random.randint(2, 90))
                            ).strftime("%Y-%m-%d"),
                        }
                    )

        return data

    def _get_body_system(self, ae_term: str) -> str:
        """Get body system for adverse event."""
        body_systems = {
            "HEADACHE": "NERVOUS SYSTEM",
            "NAUSEA": "GASTROINTESTINAL",
            "DIZZINESS": "NERVOUS SYSTEM",
            "FATIGUE": "GENERAL",
            "DIARRHEA": "GASTROINTESTINAL",
            "RASH": "SKIN",
            "INSOMNIA": "NERVOUS SYSTEM",
            "ANXIETY": "PSYCHIATRIC",
            "COUGH": "RESPIRATORY",
            "FEVER": "GENERAL",
        }
        return body_systems.get(ae_term, "GENERAL")

    def generate_adam_adsl(self) -> List[Dict]:
        """Generate ADaM ADSL (Subject Level Analysis Dataset)."""
        data = []

        for i, subject_id in enumerate(self.subject_ids):
            # Random treatment assignment
            treatment = self.random.choice(["PLACEBO", "DRUG A", "DRUG B"])

            # Random dates
            rand_date = datetime.now() - timedelta(days=self.random.randint(100, 200))

            data.append(
                {
                    "STUDYID": self.study_id,
                    "USUBJID": subject_id,
                    "SUBJID": subject_id[-3:],
                    "SITEID": f"SITE{self.random.randint(1, 10):02d}",
                    "TRT01P": treatment,
                    "TRT01A": treatment,
                    "TRTSDT": rand_date.strftime("%Y-%m-%d"),
                    "TRTEDT": (rand_date + timedelta(days=84)).strftime("%Y-%m-%d"),
                    "TRTDURD": 84,
                    "AGE": self.np_random.randint(18, 85),
                    "AGEU": "YEARS",
                    "SEX": self.random.choice(["M", "F"]),
                    "RACE": self.random.choice(["WHITE", "BLACK", "ASIAN", "HISPANIC"]),
                    "ETHNIC": self.random.choice(
                        ["HISPANIC OR LATINO", "NOT HISPANIC OR LATINO"]
                    ),
                    "HEIGHTBL": round(self.np_random.uniform(150, 190), 1),
                    "HEIGHTBLU": "cm",
                    "WEIGHTBL": round(self.np_random.uniform(50, 100), 1),
                    "WEIGHTBLU": "kg",
                    "BMIBL": round(self.np_random.uniform(18, 35), 1),
                    "SAFFL": "Y",
                    "PENFL": self.random.choice(["Y", "N"]),
                    "ITTFL": "Y",
                    "EFFFL": self.random.choice(["Y", "N"]),
                    "COMPFL": self.random.choice(["Y", "N"]),
                    "COMP8FL": self.random.choice(["Y", "N"]),
                    "COMP16FL": self.random.choice(["Y", "N"]),
                    "DSRAFFL": self.random.choice(["Y", "N"]),
                    "DSRRAFL": self.random.choice(["Y", "N"]),
                }
            )

        return data

    def generate_adam_adlb(self) -> List[Dict]:
        """Generate ADaM ADLB (Laboratory Analysis) data."""
        data = []

        lab_params = [
            ("ALT", "U/L", "ALT"),
            ("AST", "U/L", "AST"),
            ("CREAT", "umol/L", "CREATININE"),
            ("GLUC", "mmol/L", "GLUCOSE"),
            ("HGB", "g/L", "HEMOGLOBIN"),
        ]

        for subject_id in self.subject_ids:
            for param_code, unit, param_name in lab_params:
                # Baseline value
                baseline = round(self.np_random.uniform(20, 80), 2)

                # Post-baseline values
                for visit in [1, 2, 3, 4, 5]:
                    if visit == 1:
                        # Baseline
                        aval = baseline
                        ablfl = "Y"
                    else:
                        # Post-baseline with some variation
                        change = self.np_random.uniform(-10, 10)
                        aval = round(baseline + change, 2)
                        ablfl = "N"

                    data.append(
                        {
                            "STUDYID": self.study_id,
                            "USUBJID": subject_id,
                            "ADSEQ": len(data) + 1,
                            "TRT01P": self.random.choice(
                                ["PLACEBO", "DRUG A", "DRUG B"]
                            ),
                            "TRT01A": self.random.choice(
                                ["PLACEBO", "DRUG A", "DRUG B"]
                            ),
                            "PARAMCD": param_code,
                            "PARAM": param_name,
                            "PARCAT1": "LABORATORY",
                            "AVAL": aval,
                            "AVALC": str(aval),
                            "BASE": baseline,
                            "BASETYPE": "LAST",
                            "ABLFL": ablfl,
                            "CHG": round(aval - baseline, 2) if visit > 1 else 0,
                            "PCHG": (
                                round((aval - baseline) / baseline * 100, 1)
                                if visit > 1 and baseline != 0
                                else 0
                            ),
                            "AVISITNUM": visit,
                            "AVISIT": self._get_visit_name(visit),
                            "ADT": (
                                datetime.now() + timedelta(days=visit * 14)
                            ).strftime("%Y-%m-%d"),
                            "ADTM": (
                                datetime.now() + timedelta(days=visit * 14)
                            ).strftime("%Y-%m-%dT%H:%M:%S"),
                            "DTYPE": "OBSERVATION",
                        }
                    )

        return data

    def _get_visit_name(self, visit_num: int) -> str:
        """Get visit name by number."""
        visits = {1: "BASELINE", 2: "WEEK 2", 3: "WEEK 4", 4: "WEEK 8", 5: "WEEK 12"}
        return visits.get(visit_num, f"WEEK {visit_num * 2}")

    def generate_sap_document(self) -> str:
        """Generate Statistical Analysis Plan document."""
        sap_content = f"""
# Statistical Analysis Plan - {self.study_id}

## 1. Introduction
This document outlines the statistical analysis plan for clinical trial {self.study_id}.

## 2. Study Objectives
### 2.1 Primary Objective
To evaluate the efficacy of Drug A compared to placebo in improving patient outcomes.

### 2.2 Secondary Objectives
- To evaluate the safety profile of Drug A
- To assess the effect on laboratory parameters
- To evaluate quality of life measures

## 3. Study Design
- Phase: II/III
- Design: Randomized, double-blind, placebo-controlled
- Sample Size: {self.n_subjects} subjects
- Randomization: 1:1:1 (Placebo:Drug A:Drug B)
- Duration: 12 weeks

## 4. Analysis Populations
### 4.1 Safety Population
All subjects who receive at least one dose of study medication.

### 4.2 Intent-to-Treat (ITT) Population
All subjects who are randomized and receive at least one dose.

### 4.3 Per Protocol (PP) Population
Subset of ITT population without major protocol violations.

## 5. Efficacy Analyses
### 5.1 Primary Endpoint
Change from baseline in primary efficacy parameter at Week 12.

### 5.2 Statistical Methods
- ANCOVA model with treatment as factor and baseline as covariate
- Two-sided 95% confidence intervals
- p-value < 0.05 for statistical significance

## 6. Safety Analyses
### 6.1 Adverse Events
- Summarize by treatment group
- Calculate incidence rates
- Compare between groups using chi-square test

### 6.2 Laboratory Parameters
- Summarize changes from baseline
- Identify shifts in toxicity grades
- Clinical significance assessment

## 7. Data Handling
### 7.1 Missing Data
- Primary analysis: Last Observation Carried Forward (LOCF)
- Sensitivity analysis: Mixed Effects Model

### 7.2 Outliers
- Pre-defined outlier criteria
- Sensitivity analyses excluding outliers

## 8. Software
- SAS Version 9.4
- R Version 4.2.0
- Validation procedures per SOP

## 9. Appendices
### 9.1 Data Definitions
- SDTM domain specifications
- ADaM dataset structures
- Variable definitions

### 9.2 Programming Standards
- CDISC compliance
- Validation checks
- Documentation requirements
"""
        return sap_content

    def generate_protocol_document(self) -> str:
        """Generate Protocol document."""
        protocol_content = f"""
# Clinical Protocol - {self.study_id}

## Title
A Randomized, Double-Blind, Placebo-Controlled Study of Drug A in Indication

## Sponsor
Pharma Corporation

## Phase
Phase II/III

## Objectives
### Primary
To evaluate the efficacy and safety of Drug A compared to placebo.

### Secondary
- To assess dose-response relationship
- To evaluate quality of life
- To assess long-term safety

## Study Design
- Type: Interventional
- Assignment: Randomized
- Masking: Double-blind
- Control: Placebo
- Allocation: Randomized

## Population
### Inclusion Criteria
1. Age 18-85 years
2. Diagnosis of indication
3. Able to provide informed consent
4. Willing to comply with study procedures

### Exclusion Criteria
1. Prior exposure to study drug
2. Significant comorbidities
3. Pregnant or breastfeeding
4. Participation in another trial

## Interventions
### Drug A
- Dose: 100mg daily
- Route: Oral
- Duration: 12 weeks

### Placebo
- Matching placebo
- Same schedule as active drug

## Assessments
### Efficacy
- Primary endpoint assessed at Week 12
- Secondary endpoints at Weeks 4, 8, 12
- Follow-up at Week 16

### Safety
- Adverse events throughout study
- Laboratory parameters at each visit
- Vital signs at each visit
- ECG at baseline and Week 12

## Sample Size
- Total subjects: {self.n_subjects}
- Placebo: {self.n_subjects // 3}
- Drug A: {self.n_subjects // 3}
- Drug B: {self.n_subjects // 3}

## Statistical Considerations
### Analysis Sets
- Intent-to-Treat (ITT)
- Per Protocol (PP)
- Safety

### Hypothesis Testing
- Primary hypothesis: H0: No difference vs H1: Difference exists
- Two-sided alpha = 0.05
- Power = 80%

## Ethics
- IRB/IEC approval required
- GCP compliance
- Informed consent mandatory
- Subject confidentiality protected

## References
1. ICH E6(R2) Good Clinical Practice
2. ICH E9 Statistical Principles
3. CDISC Standards
"""
        return protocol_content

    def save_data(self):
        """Save all generated data to files."""
        # Generate SDTM data
        dm_data = self.generate_demographics()
        vs_data = self.generate_vitals()
        lb_data = self.generate_lab_results()
        ae_data = self.generate_adverse_events()

        # Generate ADaM data
        adsl_data = self.generate_adam_adsl()
        adlb_data = self.generate_adam_adlb()

        # Generate documents
        sap_content = self.generate_sap_document()
        protocol_content = self.generate_protocol_document()

        # Save SDTM files
        self._save_json(dm_data, f"{self.study_id}_DM.json")
        self._save_json(vs_data, f"{self.study_id}_VS.json")
        self._save_json(lb_data, f"{self.study_id}_LB.json")
        self._save_json(ae_data, f"{self.study_id}_AE.json")

        # Save ADaM files
        self._save_json(adsl_data, f"{self.study_id}_ADSL.json")
        self._save_json(adlb_data, f"{self.study_id}_ADLB.json")

        # Save documents
        self._save_markdown(sap_content, f"{self.study_id}_SAP.md")
        self._save_markdown(protocol_content, f"{self.study_id}_Protocol.md")

        # Save as CSV for variety
        self._save_csv(dm_data, f"{self.study_id}_DM.csv")
        self._save_csv(lb_data, f"{self.study_id}_LB.csv")

        print(f"Generated dummy clinical data for study {self.study_id}")
        print(f"Files saved to: {DATA_DIR}")

    def _save_json(self, data: List[Dict], filename: str):
        """Save data as JSON."""
        with open(DATA_DIR / filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_csv(self, data: List[Dict], filename: str):
        """Save data as CSV."""
        df = pd.DataFrame(data)
        df.to_csv(DATA_DIR / filename, index=False)

    def _save_markdown(self, content: str, filename: str):
        """Save document as Markdown."""
        with open(DATA_DIR / filename, "w") as f:
            f.write(content)


if __name__ == "__main__":
    # Generate data for multiple studies
    studies = ["STUDY001", "STUDY002", "STUDY003"]

    for study_id in studies:
        generator = ClinicalDataGenerator(study_id=study_id, n_subjects=50)
        generator.save_data()

    print("\nGenerated dummy data for 3 studies")
    print("Each study includes:")
    print("- SDTM domains: DM, VS, LB, AE")
    print("- ADaM datasets: ADSL, ADLB")
    print("- Documents: SAP, Protocol")
    print("- Formats: JSON, CSV, Markdown")
