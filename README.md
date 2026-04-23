# PULSAR-Solar-Flare-Protection
PULSAR: Autonomous UAV GNSS Interference & Space Weather Risk Estimation System
PULSAR is an edge-compatible MLOps architecture designed to autonomously differentiate between natural space weather anomalies (geomagnetic storms, TEC anomalies) and adversarial electronic warfare (jamming, spoofing).

Instead of relying on raw, uncalibrated model confidences, PULSAR evaluates strictly calibrated risk scores to feed a deterministic, hardware-constrained decision policy. It fuses localized GNSS time-series data with global space weather contextual variables to probabilistically classify signal disruptions and execute optimal mitigation strategies based on expected cost minimization.

1. System Architecture and Data Fusion
The pipeline ingests and synchronizes disparate data streams, operating under strict fault-tolerance constraints (utilizing Google Colab checkpointing during the training phase).

GNSS Signal Time Series: SNR, C/N₀, and Doppler shift variance extracted via georinex.

Space Weather Context (APIs): Kp index, F10.7 solar flux, IMF Bz, and solar wind velocity sourced real-time from NOAA SWPC and GFZ Potsdam.

Adversarial Datasets: Spoofing and jamming signatures integrated from the TEXBAT dataset.

Ephemeris Data: NASA CDDIS for satellite positioning context.

2. ML Pipeline & Target Classification
The inference engine relies on an ensemble and temporal sequence architecture (XGBoost, Random Forest, 1D-CNN/ConvLSTM) to classify the GNSS state space into a discrete set:

Y={Nominal,Space_Weather,Jamming,Spoofing}
The models process statistical features extracted from raw IF-level injection tests, utilizing spectral analysis methods (Welch PSD, STFT) to isolate adversarial signatures from ionospheric scintillation.

3. Probability Calibration (The Core Constraint)
Raw neural network outputs are often overconfident. PULSAR strictly mandates probability calibration using Platt Scaling and Isotonic Regression. The reliability of the model is quantified using the Expected Calibration Error (ECE):
​	
  represents the prediction bins, acc is the true accuracy, and conf is the predicted confidence. A payload is rejected if the ECE exceeds the pre-defined safety threshold.

4. Deterministic Hardware Decision Policy
The calibrated probabilities P(y∣X) do not directly control the UAV. They are fed into a deterministic decision matrix that calculates the Expected Cost C(a) for every possible hardware action a∈A (e.g., Switch to Visual Odometry, Trigger Failsafe, Maintain GPS).

Where L(a,y) is the static loss/penalty matrix defined for taking action a when the true environmental state is y. The system strictly selects the action that minimizes C(a).

5. Technical Stack & Deployment
Core Logic: Python 3.10+, SciPy, scikit-learn.

Deep Learning: PyTorch (1D-CNN, ConvLSTM).

Tree Ensembles: XGBoost, Random Forest.

Signal Processing: georinex (RINEX parsing), Welch PSD, STFT.

Local Initialization & Testing

To run the spectral analysis and model calibration pipeline locally:

# Install strict dependencies
pip install -r requirements.txt

# Run the ingestion and alignment pipeline
python -m pulsar.data.fusion_pipeline

# Execute calibration and print ECE metrics
python -m pulsar.models.evaluate_calibration
