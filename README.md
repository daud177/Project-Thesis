# Optimization and Practical Evaluation of an Interactive Learning Framework for Robotic Manipulation of Electric Motors Components

# This repository contains the codebase developed for the thesis project, which explores robotic manipulation of electric motor components using behavior cloning (BC) and interactive learning frameworks. The project focuses on applying and evaluating data augmentation techniques (DAT) to optimize BC and testing pre-trained models and custom architectures.

---

## 📂 Project Structure

### Main Directories and Files

#### `src_dawood_DataAugment`
- **Description:** Contains implementations of behavior cloning with different data augmentation pipelines.
- **Files:**
  - `behaviour_cloning_DAT_1.py` to `behaviour_cloning_DAT_7.py`: Implementations of BC with the seven different data augmentation techniques discussed in the report.
  - `behaviour_cloning_ELA_UPA.py` and `behaviour_cloning_ELA_DV.py`: Episode-Level Augmentations using Uniform Positive Adjustments (UPA) and Dynamic Variability (DV).
  - `behaviour_cloning_FLA_UPA.py` and `behaviour_cloning_FLA_DV.py`: Frame-Level Augmentations using UPA and DV.

#### `src_dawood_efficientnet`
- **Description:** Implements the EfficientNet B2 pre-trained model.
- **Files:**
  - `models.py`: Contains the EfficientNet architecture integrated with the project.

#### `src_dawood_mobilenet`
- **Description:** Implements the MobileNet V2 pre-trained model.
- **Files:**
  - `models.py`: Contains the MobileNet architecture integrated with the project.

#### `src_dawood_Resnet`
- **Description:** Implements the ResNet-18 pre-trained model.
- **Files:**
  - `models.py`: Contains the ResNet-18 architecture integrated with the project.

#### `src_Model_V2_256`
- **Description:** Contains the original `Model_V2_256` architecture, which serves as the baseline from preliminary work by Akash and Tim.
- **Files:**
  - All files for training, evaluation, and environment setup with the original architecture.

### Common Files Across Folders
- **`models.py`:** Defines the model architecture.
- **`utils.py`:** Contains utility functions for data processing and model training.
- **`control_wsg_50.py`:** Handles gripper control for robotic manipulation.
- **`custom_env.py`:** Custom environment definitions for training and evaluation.
- **`evaluate.py`:** Script for evaluating trained models.
- **`feedback_train.py`:** Handles feedback-based training.
- **`human_feedback.py`:** Simulates human feedback for the robot.
- **`move_client.py`:** Handles robot motion commands.
- **`RosRobot.py`:** Interface for ROS-based robotic control.
- **`teleoperation_env.py`:** Enables teleoperation and data recording.
- **`iiwa_py3`:** A package for controlling the KUKA iiwa robot.

---

## 🧪 Implemented Augmentation Techniques

- **Data Augmentation Pipelines (DAT 1–7):** Discussed in the thesis report, each pipeline uses unique augmentation techniques like brightness, contrast, gamma adjustment, noise, sharpening, and blur.
- **Episode-Level Augmentations (ELA):**
  - UPA: Uniform Positive Adjustments.
  - DV: Dynamic Variability.
- **Frame-Level Augmentations (FLA):**
  - UPA: Uniform Positive Adjustments.
  - DV: Dynamic Variability.

Refer to the thesis report for a detailed explanation of augmentation techniques.

---

### How to Run
1. Access the iRobot PC terminal:
   ```bash
   source /home/faps/Workspace/workspace_kuka/devel/setup.bash
   conda activate ceiling_env
   cd CEILing/CEILing256_v2


# Train Model
python src_dawood_DataAugment/behavior_cloning.py --task ".dat file path" --feedback_type "cloning_100" 

# Evaliate Model
python src_dawood_DataAugment/evaluate.py --task "path to trained policy folder" --feedback_type cloning_100



# Features
# Behavior Cloning Framework: Optimized using data augmentation pipelines for improved performance.
# Pre-trained Model Integration: ResNet-18, MobileNet V2, and EfficientNet B2 implementations for robotic manipulation.
# Baseline Comparison: Includes original Model_V2_256 architecture for benchmarking.
# Notes and Limitations
# Due to time constraints, CEILing implementation was not tested; the focus was on BC.
# Pre-trained models failed to pick up a stator in the multiple stator use case, but DAT pipelines demonstrated significant performance improvements.