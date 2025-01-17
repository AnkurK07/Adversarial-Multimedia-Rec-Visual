import sys
import io
import re
import json
from src.logger import logging

class RunExp:
    def __init__(self, exp, k=50):
        """
        Initializes the RunExp class.

        Parameters:
        - exp: The experiment object to run (should have a .run() method).
        - k: The cutoff value for Precision and Recall (default is 50).
        """
        self.exp = exp
        self.k = k
        self.metrics = None  # To store extracted metrics

    def run_experiment(self):
        """
        Runs the experiment and extracts Precision@k and Recall@k.

        Returns:
        - metrics: A dictionary containing Precision@k and Recall@k.
        """
        # Step 1: Capture printed output
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        try:
            logging.info("Training ..........")
            # Step 2: Run the experiment
            self.exp.run()
            logging.info("Training Completed !")
        finally:
            # Step 3: Reset stdout
            sys.stdout = old_stdout

        # Step 4: Get the captured output
        output = new_stdout.getvalue()

        # Step 5: Extract Precision@k and Recall@k using regex
        pattern = rf"AMR\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
        match = re.search(pattern, output)

        # Step 6: If metrics are found, store them
        if match:
            self.metrics = {
                f"Precision@{self.k}": float(match.group(1)),
                f"Recall@{self.k}": float(match.group(2))
            }
            print(f"✅ Extracted metrics: {self.metrics}")
        else:
            print("❌ No metrics found in the output.")
            self.metrics = None

        return self.metrics

    def save_metrics(self, file_path):
        """
        Saves the extracted metrics to a JSON file at the given file path.

        Parameters:
        - file_path: Path to save the extracted metrics JSON file.
        """
        if self.metrics:
            with open(file_path, "w") as file:
                json.dump(self.metrics, file, indent=4)
            print(f"✅ Metrics saved to '{file_path}'.")
        else:
            print("❌ No metrics to save. Run the experiment first.")
