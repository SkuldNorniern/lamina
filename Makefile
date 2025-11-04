# Lamina Compiler Makefile

.PHONY: test test-single build clean benchmark help

# Default target
help:
	@echo "Lamina Compiler Build System"
	@echo "============================="
	@echo ""
	@echo "Available targets:"
	@echo "  test           - Run all test cases"
	@echo "  test-cargo     - Run Cargo integration tests"
	@echo "  test-single    - Run a single test case (specify TEST=filename)"
	@echo "  build          - Build the compiler"
	@echo "  clean          - Clean build artifacts"
	@echo "  benchmark      - Run performance benchmarks"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make test                          # Run all tests"
	@echo "  make test-single TEST=arithmetic  # Run arithmetic test"
	@echo "  make benchmark                     # Run benchmarks"

# Build the compiler
build:
	@echo "🔨 Building Lamina compiler..."
	@cargo build --release

# Run all test cases
test:
	@echo "🧪 Running Lamina test suite..."
	@python3 run_tests.py

# Run cargo tests (integration tests)
test-cargo:
	@echo "🦀 Running Cargo integration tests..."
	@cargo test --test integration_tests

# Run a single test case
test-single:
	@if [ -z "$(TEST)" ]; then \
		echo "❌ Please specify a test case: make test-single TEST=arithmetic"; \
		echo "Available tests:"; \
		python3 run_tests.py --list; \
	else \
		python3 run_tests.py $(TEST).lamina; \
	fi

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean
	@rm -f testcases/*.s
	@rm -f benchmarks/*/*.s
	@rm -f simple_test simple_test.s
	@find . -name "*.s" -type f -delete
	@find . -name "arithmetic" -o -name "loops" -o -name "conditionals" \
		-o -name "functions" -o -name "constants" -o -name "variables" \
		-o -name "simple_const" -o -name "simple_print" -o -name "simple_plus" \
		-type f -delete

# Run performance benchmarks
benchmark:
	@echo "🚀 Running performance benchmarks..."
	@python3 run_benchmark.py

# Development helpers
dev-test: build test

# Check code formatting and linting
check:
	@echo "🔍 Checking code formatting..."
	@cargo fmt --check
	@cargo clippy -- -D warnings

# Format code
fmt:
	@echo "🎨 Formatting code..."
	@cargo fmt
