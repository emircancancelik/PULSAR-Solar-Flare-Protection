# PULSAR: Autonomous UAV GNSS Interference & Space Weather Risk Estimation System

PULSAR is a production-ready, hardware-constrained solar alert and GNSS risk estimation system engineered for Unmanned Aerial Vehicles (UAVs). The system mathematically segregates space weather anomalies (e.g., geomagnetic storms, ionospheric scintillation, TEC anomalies) from intentional electronic warfare (jamming and spoofing). To prevent catastrophic downstream failures caused by model overconfidence, PULSAR integrates a Platt Scaling calibration layer to minimize Expected Calibration Error (ECE) and executes a deterministic decision policy optimized for expected operational cost minimization.

## 1. Architectural Design & Methodology
The system architecture is decoupled into three primary asynchronous, non-blocking layers: Data Ingestion, Calibrated Risk Classification, and Deterministic Policy Evaluation.
Probabilistic Risk Differentiation & Calibration
Raw classification outputs or logit matrices from deep learning architectures (e.g., ConvLSTM, XGBoost) often suffer from miscalibration. PULSAR enforces strict statistical validation by passing raw network inferences through a sigmoid-based Platt Scaling calibration layer. The pipeline classifies environmental telemetry into four mutually exclusive domain states:
Nominal: Stable signal propagation without external atmospheric or adversarial degradation.
Space Weather Anomaly: Ionospheric delays, Total Electron Content (TEC) spikes, and solar wind-induced geomagnetic disruptions.
Jamming: Intentional RF power injection leading to a degraded noise floor, captured via SNR and C/N 

​
Spoofing: Coordinated adversarial signal manipulation exhibiting anomalous Doppler shifts and artificial time-synchronization vectors.
Deterministic Decision Policy
Once calibrated state probabilities are derived, the system evaluates an immutable operational cost matrix to transition the UAV's navigational stack.
​	

Where a∈A represents the available control actions, s∈S represents the latent environmental states, P(s∣x) is the strictly calibrated state probability vector, and Cost(a,s) denotes the operational penalty matrix. The engine automatically executes the action that achieves global cost minimization.


## 2. Repository Structure & Domain Modules

The repository layout is organized into specialized domain modules to isolate data extraction, predictive modeling, deterministic tracking, and execution lifecycles:

PULSAR-Solar-System/
│
├── agent/                  # Multi-agent telemetry monitoring and processing logic
├── archive/                # Legacy baseline configurations and historical code versions
├── core/                   # Platt Scaling calibration, optimization engines, and mathematical validation
├── data/                   # Local storage for telemetry caches and operational logs (Git ignored)
├── data_pipeline/          # Asynchronous, non-blocking ingestion layers for NOAA SWPC and GFZ Potsdam APIs
├── models/                 # Serialized model weights (XGBoost/ConvLSTM) and calibration parameters
├── pulsar_env/             # Local isolated Python virtual environment dependencies
├── solarsystem/            # Solar flare tracking, TEC anomalies, and space weather vector computations
├── utils/                  # Domain-specific helpers, statistical evaluators, and structured log formatters
│
├── .gitignore              # Project-wide exclusion tracking rules
├── app.py                  # High-level entry point or serving API layer
├── launcher.py             # System bootstrap sequence and multi-process orchestrator
├── LICENSE                 # System distribution and licensing rights
├── main.py                 # Central asynchronous execution loop and runtime manager
├── noaa_texbat_train2.py   # Training script for model optimization using NOAA and TEXBAT datasets
└── README.md               # Technical system architecture documentation

## 3. Operational Cost Matrix Formulation
The decision policy operates on a predefined penalty matrix designed to mathematically balance the trade-offs between false alarms and undetected malicious spoofing/jamming attacks:
Control Action / True State (s)	Nominal	Space Weather	Jamming	Spoofing
CONTINUE_NOMINAL_NAVIGATION	0.0	50.0	200.0	500.0
SWITCH_TO_VISUAL_ODOMETRY	20.0	10.0	30.0	40.0
EXECUTE_FAILSAFE_AUTONOMOUS_LANDING	100.0	80.0	70.0	60.0

## 4. Production Standards & Engineering Constraints
Type Safety: Absolute enforcement of Python type hinting coupled with runtime schema verification via Pydantic V2.
Structured Logging: Standard print() statements are prohibited. All operational events, network anomalies, and policy evaluations are streamed as structured, machine-readable entries via the native logging utility.
Granular Exception Handling: Avoids generic except Exception blocks. Network I/O failures isolate aiohttp.ClientError, while hardware telemetric timeouts specifically catch asyncio.TimeoutError.

## 5. Installation & Deployment
Prerequisites
The core execution layer requires Python 3.10 or higher. Install the necessary mathematical modeling and async networking dependencies using the following package configuration:

## pip install pydantic aiohttp scikit-learn numpy scipy xgboost


Initializing the System
The orchestrator executes a bootstrap sequence on startup to fit the calibration parameters before starting the continuous real-time execution loop:

## python main.py
