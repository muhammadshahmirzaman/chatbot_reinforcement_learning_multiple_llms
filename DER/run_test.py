import subprocess
import sys

# Run the test and capture output
result = subprocess.run(
    [sys.executable, "test_integration.py"],
    capture_output=True,
    text=True
)

# Write output to a simple text file with UTF-8 encoding
with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\n\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\n\nEXIT CODE: {result.returncode}\n")

print("Test completed. Results saved to test_results.txt")
print(f"Exit code: {result.returncode}")
