import torch
import pandas as pd
import numpy as np
from hierarchical_discriminators import HierarchicalDiscriminatorSystem
from utils import format_row
import logging
import sys
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("discriminator_dashboard.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class DiscriminatorDashboard:
    """
    Dashboard for monitoring and validating discriminator performance
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.discriminators = HierarchicalDiscriminatorSystem(device=self.device)
        self.metrics_history = defaultdict(list)
        self.performance_stats = {}

    def validate_discriminator_health(self):
        """
        Comprehensive health check for all discriminators
        """
        logger.info("=== DISCRIMINATOR HEALTH CHECK ===")

        health_status = {
            "token_discriminator": False,
            "sentence_discriminator": False,
            "row_discriminator": False,
            "feature_discriminators": False,
            "overall_health": False,
        }

        try:
            # Test token discriminator
            test_text = "gender is Female, age is 30"
            token_ids = self.discriminators.tokenizer.encode(
                test_text, return_tensors="pt"
            ).to(self.device)
            token_output = self.discriminators.token_discriminator(token_ids)
            if token_output is not None and token_output.shape[0] > 0:
                health_status["token_discriminator"] = True
                logger.info("✅ Token discriminator: HEALTHY")

            # Test sentence discriminator
            encoding = self.discriminators.tokenizer(
                test_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(self.device)
            sentence_output = self.discriminators.sentence_discriminator(
                encoding["input_ids"], encoding["attention_mask"]
            )
            if sentence_output is not None:
                health_status["sentence_discriminator"] = True
                logger.info("✅ Sentence discriminator: HEALTHY")

            # Test row discriminator
            row_output = self.discriminators.row_discriminator(
                encoding["input_ids"], encoding["attention_mask"]
            )
            if row_output is not None:
                health_status["row_discriminator"] = True
                logger.info("✅ Row discriminator: HEALTHY")

            # Test feature discriminators
            feature_working = 0
            for (
                feature_name,
                discriminator,
            ) in self.discriminators.feature_discriminators.items():
                try:
                    embeddings = self.discriminators.extract_feature_embeddings(
                        test_text, feature_name
                    )
                    if embeddings is not None:
                        feature_output = discriminator(embeddings)
                        if feature_output is not None:
                            feature_working += 1
                except:
                    pass

            if feature_working > 0:
                health_status["feature_discriminators"] = True
                logger.info(
                    f"✅ Feature discriminators: {feature_working}/{len(self.discriminators.feature_discriminators)} HEALTHY"
                )

            # Overall health
            health_status["overall_health"] = all(
                [
                    health_status["token_discriminator"],
                    health_status["sentence_discriminator"],
                    health_status["row_discriminator"],
                    health_status["feature_discriminators"],
                ]
            )

            if health_status["overall_health"]:
                logger.info("🎉 ALL DISCRIMINATORS ARE HEALTHY!")
            else:
                logger.warning("⚠️ Some discriminators need attention")

            return health_status

        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return health_status

    def monitor_discriminator_performance(self, num_samples=100):
        """
        Monitor discriminator performance metrics
        """
        logger.info(f"=== PERFORMANCE MONITORING ({num_samples} samples) ===")

        try:
            # Load test data
            df = pd.read_csv("diabetes.csv")
            real_texts = df.apply(format_row, axis=1).tolist()[:num_samples]

            # Generate synthetic texts
            synthetic_texts = []
            for i in range(num_samples // 2):
                real_text = real_texts[i]
                # Create synthetic by modifying some values
                synthetic_text = real_text.replace("diabetes is 0", "diabetes is 1")
                synthetic_texts.append(synthetic_text)

            # Test real vs synthetic discrimination
            real_scores = []
            synthetic_scores = []

            start_time = time.time()

            for text in real_texts[: num_samples // 2]:
                feedback = self.discriminators.get_multi_level_feedback(text)
                real_scores.append(feedback["row"])

            for text in synthetic_texts:
                feedback = self.discriminators.get_multi_level_feedback(text)
                synthetic_scores.append(feedback["row"])

            elapsed_time = time.time() - start_time

            # Calculate metrics
            real_mean = np.mean(real_scores)
            synthetic_mean = np.mean(synthetic_scores)
            discrimination_score = abs(real_mean - synthetic_mean)

            # Store metrics
            self.metrics_history["real_scores"].extend(real_scores)
            self.metrics_history["synthetic_scores"].extend(synthetic_scores)
            self.metrics_history["discrimination_score"].append(discrimination_score)
            self.metrics_history["processing_time"].append(elapsed_time)

            # Performance stats
            self.performance_stats = {
                "real_mean": real_mean,
                "synthetic_mean": synthetic_mean,
                "discrimination_score": discrimination_score,
                "processing_time": elapsed_time,
                "samples_per_second": num_samples / elapsed_time,
                "real_std": np.std(real_scores),
                "synthetic_std": np.std(synthetic_scores),
            }

            logger.info("📊 Performance Metrics:")
            logger.info(f"  Real data mean score: {real_mean:.3f}")
            logger.info(f"  Synthetic data mean score: {synthetic_mean:.3f}")
            logger.info(f"  Discrimination score: {discrimination_score:.3f}")
            logger.info(f"  Processing time: {elapsed_time:.2f}s")
            logger.info(
                f"  Samples per second: {self.performance_stats['samples_per_second']:.1f}"
            )

            return self.performance_stats

        except Exception as e:
            logger.error(f"❌ Performance monitoring failed: {str(e)}")
            return {}

    def validate_discriminator_consistency(self, num_tests=10):
        """
        Validate discriminator consistency across multiple runs
        """
        logger.info(f"=== CONSISTENCY VALIDATION ({num_tests} tests) ===")

        try:
            test_text = "gender is Female, age is 30, hypertension is 0, heart_disease is 0, smoking_history is never, bmi is 25.5, HbA1c_level is 5.2, blood_glucose_level is 120, diabetes is 0"

            scores = []
            for i in range(num_tests):
                feedback = self.discriminators.get_multi_level_feedback(test_text)
                scores.append(feedback["row"])

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            consistency_score = 1.0 - min(std_score, 1.0)  # Higher is better

            logger.info("📈 Consistency Metrics:")
            logger.info(f"  Mean score: {mean_score:.3f}")
            logger.info(f"  Standard deviation: {std_score:.6f}")
            logger.info(f"  Consistency score: {consistency_score:.3f}")

            if consistency_score > 0.95:
                logger.info("✅ Excellent consistency")
            elif consistency_score > 0.90:
                logger.info("✅ Good consistency")
            elif consistency_score > 0.80:
                logger.info("⚠️ Acceptable consistency")
            else:
                logger.warning("❌ Poor consistency - needs attention")

            return {
                "mean_score": mean_score,
                "std_score": std_score,
                "consistency_score": consistency_score,
            }

        except Exception as e:
            logger.error(f"❌ Consistency validation failed: {str(e)}")
            return {}

    def test_discriminator_robustness(self):
        """
        Test discriminator robustness with edge cases
        """
        logger.info("=== ROBUSTNESS TESTING ===")

        edge_cases = [
            "gender is Female, age is 30",  # Incomplete text
            "invalid format text",  # Malformed text
            "",  # Empty text
            "gender is Female, age is 30, hypertension is 0, heart_disease is 0, smoking_history is never, bmi is 25.5, HbA1c_level is 5.2, blood_glucose_level is 120, diabetes is 0, extra_field is value",  # Extra fields
            "gender is Female, age is 999, hypertension is 2, heart_disease is 5, smoking_history is invalid, bmi is -5, HbA1c_level is 999, blood_glucose_level is -10, diabetes is 3",  # Invalid values
        ]

        robustness_results = {}

        for i, test_case in enumerate(edge_cases):
            try:
                if test_case.strip():  # Skip empty text
                    feedback = self.discriminators.get_multi_level_feedback(test_case)
                    robustness_results[f"case_{i}"] = {
                        "status": "success",
                        "row_score": feedback["row"],
                        "error": None,
                    }
                    logger.info(
                        f"✅ Edge case {i}: Handled successfully (score: {feedback['row']:.3f})"
                    )
                else:
                    robustness_results[f"case_{i}"] = {
                        "status": "skipped",
                        "row_score": None,
                        "error": "Empty text",
                    }
                    logger.info(f"⚠️ Edge case {i}: Skipped (empty text)")
            except Exception as e:
                robustness_results[f"case_{i}"] = {
                    "status": "failed",
                    "row_score": None,
                    "error": str(e),
                }
                logger.warning(f"❌ Edge case {i}: Failed - {str(e)}")

        # Calculate robustness score
        successful_cases = sum(
            1 for result in robustness_results.values() if result["status"] == "success"
        )
        total_cases = len(robustness_results)
        robustness_score = successful_cases / total_cases if total_cases > 0 else 0

        logger.info(
            f"📊 Robustness Score: {robustness_score:.2f} ({successful_cases}/{total_cases} cases handled)"
        )

        return robustness_results

    def generate_performance_report(self):
        """
        Generate comprehensive performance report
        """
        logger.info("=== GENERATING PERFORMANCE REPORT ===")

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "health_status": self.validate_discriminator_health(),
            "performance_stats": self.monitor_discriminator_performance(50),
            "consistency": self.validate_discriminator_consistency(),
            "robustness": self.test_discriminator_robustness(),
        }

        # Overall score calculation
        health_score = 1.0 if report["health_status"]["overall_health"] else 0.5
        performance_score = min(
            report["performance_stats"].get("discrimination_score", 0) * 10, 1.0
        )
        consistency_score = report["consistency"].get("consistency_score", 0)
        robustness_score = sum(
            1
            for result in report["robustness"].values()
            if result["status"] == "success"
        ) / len(report["robustness"])

        overall_score = (
            health_score + performance_score + consistency_score + robustness_score
        ) / 4

        report["overall_score"] = overall_score

        logger.info("📋 COMPREHENSIVE REPORT:")
        logger.info(f"  Overall Score: {overall_score:.3f}")
        logger.info(f"  Health Score: {health_score:.3f}")
        logger.info(f"  Performance Score: {performance_score:.3f}")
        logger.info(f"  Consistency Score: {consistency_score:.3f}")
        logger.info(f"  Robustness Score: {robustness_score:.3f}")

        if overall_score > 0.9:
            logger.info("🎉 EXCELLENT: Discriminators are working flawlessly!")
        elif overall_score > 0.8:
            logger.info("✅ GOOD: Discriminators are working well")
        elif overall_score > 0.7:
            logger.info("⚠️ ACCEPTABLE: Some improvements needed")
        else:
            logger.warning("❌ NEEDS ATTENTION: Significant issues detected")

        return report


def main():
    """
    Main function to run discriminator dashboard
    """
    try:
        logger.info("=== STARTING DISCRIMINATOR DASHBOARD ===")

        dashboard = DiscriminatorDashboard()

        # Run comprehensive validation
        report = dashboard.generate_performance_report()

        logger.info("=== DASHBOARD COMPLETED ===")

        return report

    except Exception as e:
        logger.error(f"❌ Dashboard failed: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
