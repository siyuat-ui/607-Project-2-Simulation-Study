# Makefile for Engression Simulation Study

.PHONY: all simulate analyze figures clean test

# Run complete simulation pipeline and generate all outputs
all: simulate analyze figures
	@echo "=========================================="
	@echo "Complete pipeline finished!"
	@echo "Results in: results/raw/"
	@echo "Figures in: results/figures/"
	@echo "=========================================="

# Run simulations and save raw results
simulate:
	@echo "=========================================="
	@echo "Running simulation..."
	@echo "=========================================="
	python src/main.py --mode full --replications 10 --sizes 100 500 1000
	@echo "Simulation complete!"

# Process raw results and generate summary statistics
analyze:
	@echo "=========================================="
	@echo "Analyzing results..."
	@echo "=========================================="
	@if [ -d results/raw ]; then \
		python -c "import pandas as pd; import glob; files = glob.glob('results/raw/simulation_results_*.csv'); df = pd.read_csv(max(files)) if files else None; print('\n=== SUMMARY STATISTICS ===\n' + str(df.groupby('generator')[['mmd', 'two_sample_mean_p', 'mean_distance', 'cov_frobenius', 'training_time']].mean()) + '\n\n=== SAMPLE COUNTS ===\n' + str(df.groupby(['generator', 'sample_size']).size())) if df is not None else print('No results found. Run make simulate first.')"; \
	else \
		echo "No results directory found. Run make simulate first."; \
	fi

# Create all visualizations
figures:
	@echo "=========================================="
	@echo "Generating analysis figures..."
	@echo "=========================================="
	python src/analyze_results.py

# Remove generated files
clean:
	@echo "=========================================="
	@echo "Cleaning generated files..."
	@echo "=========================================="
	rm -rf results/raw/*.csv
	rm -rf results/figures/*.png
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Run test suite
test:
	@echo "=========================================="
	@echo "Running test suite..."
	@echo "=========================================="
	pytest tests/ -v